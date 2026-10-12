"""Label construction + LightGBM training with walk-forward validation.

Three data roles, kept strictly separate to avoid fooling ourselves:
  - dev folds: walk-forward train/validate, used only to pick model hyperparams
    and to generate oos_predictions for threshold TUNING.
  - holdout: the final slice of history, never touched by CV or threshold tuning.
    holdout_predictions come from a model that never saw this data — this is the
    one honest number for "would this actually have worked."
  - production_model: retrained on dev+holdout together once validated, so the
    deployed model doesn't throw away the most recent data.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from ai_trader.features.indicators import FEATURE_COLUMNS, build_feature_frame

LABEL_HORIZON = 4  # bars ahead to label direction over

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
}


def label_data(feature_df: pd.DataFrame, horizon: int = LABEL_HORIZON) -> pd.DataFrame:
    """Binary label: does close price `horizon` bars ahead exceed current close?"""
    out = feature_df.copy()
    future_close = out["close"].shift(-horizon)
    out["forward_return"] = future_close / out["close"] - 1
    out["label"] = (out["forward_return"] > 0).astype(int)
    return out.iloc[:-horizon]  # drop rows without a future label


@dataclass
class TrainResult:
    model: lgb.Booster  # production model, trained on dev+holdout — use this to deploy
    metrics: dict
    feature_columns: list[str]
    oos_predictions: pd.DataFrame  # dev-fold CV predictions — for threshold TUNING only
    holdout_predictions: pd.DataFrame  # never-trained-on final slice — for honest FINAL check


def _train_one(train_slice: pd.DataFrame, val_slice: pd.DataFrame | None = None) -> lgb.Booster:
    train_set = lgb.Dataset(train_slice[FEATURE_COLUMNS], label=train_slice["label"])
    if val_slice is not None and not val_slice.empty:
        val_set = lgb.Dataset(val_slice[FEATURE_COLUMNS], label=val_slice["label"], reference=train_set)
        return lgb.train(
            LGB_PARAMS, train_set, num_boost_round=200,
            valid_sets=[val_set], callbacks=[lgb.early_stopping(20, verbose=False)],
        )
    return lgb.train(LGB_PARAMS, train_set, num_boost_round=150)


def walk_forward_train(df: pd.DataFrame, n_splits: int = 5, holdout_frac: float = 0.15) -> TrainResult:
    """Time-ordered walk-forward split on the dev portion; final holdout_frac of
    history is held out entirely until the end."""
    labeled = label_data(build_feature_frame(df)).dropna(subset=FEATURE_COLUMNS)
    labeled = labeled.reset_index(drop=True)

    n = len(labeled)
    if n < 500:
        raise ValueError(f"Not enough labeled rows to train ({n}). Fetch more history.")

    holdout_start = int(n * (1 - holdout_frac))
    dev = labeled.iloc[:holdout_start].reset_index(drop=True)
    holdout = labeled.iloc[holdout_start:].reset_index(drop=True)

    n_dev = len(dev)
    fold_size = n_dev // (n_splits + 1)
    fold_metrics = []
    oos_chunks = []
    last_fold_model = None

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        val_end = min(fold_size * (i + 1), n_dev)
        train_slice = dev.iloc[:train_end]
        val_slice = dev.iloc[train_end:val_end]
        if val_slice.empty:
            continue

        last_fold_model = _train_one(train_slice, val_slice)
        preds = last_fold_model.predict(val_slice[FEATURE_COLUMNS])
        fold_metrics.append(
            {
                "fold": i,
                "accuracy": accuracy_score(val_slice["label"], preds > 0.5),
                "auc": roc_auc_score(val_slice["label"], preds) if val_slice["label"].nunique() > 1 else float("nan"),
                "n_val": len(val_slice),
            }
        )
        oos_chunk = val_slice[["timestamp", "close", "forward_return", "label"]].copy()
        oos_chunk["pred_proba"] = preds
        oos_chunks.append(oos_chunk)

    if last_fold_model is None:
        raise ValueError("Walk-forward training produced no folds — check data size/n_splits.")

    # Model trained on dev only, used ONLY to score the untouched holdout honestly.
    dev_only_model = _train_one(dev)
    holdout_preds = dev_only_model.predict(holdout[FEATURE_COLUMNS])
    holdout_predictions = holdout[["timestamp", "close", "forward_return", "label"]].copy()
    holdout_predictions["pred_proba"] = holdout_preds
    holdout_auc = (
        roc_auc_score(holdout["label"], holdout_preds) if holdout["label"].nunique() > 1 else float("nan")
    )

    # Production model: retrained on everything (dev+holdout) now that validation is done.
    production_model = _train_one(labeled)

    avg_auc = float(np.nanmean([m["auc"] for m in fold_metrics]))
    avg_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
    oos_predictions = pd.concat(oos_chunks, ignore_index=True) if oos_chunks else pd.DataFrame()

    return TrainResult(
        model=production_model,
        metrics={
            "folds": fold_metrics,
            "avg_auc": avg_auc,
            "avg_accuracy": avg_acc,
            "holdout_auc": float(holdout_auc),
            "n_rows": n,
            "n_dev": n_dev,
            "n_holdout": len(holdout),
        },
        feature_columns=FEATURE_COLUMNS,
        oos_predictions=oos_predictions,
        holdout_predictions=holdout_predictions,
    )


def save_model(result: TrainResult, model_dir: Path, tag: str) -> Path:
    version_dir = model_dir / tag
    version_dir.mkdir(parents=True, exist_ok=True)
    result.model.save_model(str(version_dir / "model.txt"))
    with open(version_dir / "meta.json", "w") as f:
        json.dump({"metrics": result.metrics, "feature_columns": result.feature_columns}, f, indent=2)
    latest_pointer = model_dir / "latest.json"
    with open(latest_pointer, "w") as f:
        json.dump({"version": tag}, f)
    return version_dir


def load_latest_model(model_dir: Path):
    latest_pointer = model_dir / "latest.json"
    if not latest_pointer.exists():
        return None, None
    with open(latest_pointer) as f:
        tag = json.load(f)["version"]
    version_dir = model_dir / tag
    model = lgb.Booster(model_file=str(version_dir / "model.txt"))
    with open(version_dir / "meta.json") as f:
        meta = json.load(f)
    return model, meta
