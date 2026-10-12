"""Pushes each scan to Supabase so predictions have a history, and scores
every prior still-open signal against what the price actually did since —
the feedback loop the static dashboard alone can't give you.

No-ops cleanly if SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY aren't set, so
local runs without Supabase configured keep working unchanged.
"""
import logging

import requests

logger = logging.getLogger("supabase_sync")

_TIMEOUT_S = 15


def _headers(service_role_key: str) -> dict:
    return {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json",
    }


def _score_prior_signals(base_url: str, headers: dict, coin: str, horizon: str, live_price: float) -> None:
    """Find this coin/horizon's most recent prior signal that hasn't been scored
    yet, and record what its live_price turned into by now."""
    resp = requests.get(
        f"{base_url}/rest/v1/coin_signals",
        headers=headers,
        params={
            "select": "id,live_price,live_status,created_at",
            "coin": f"eq.{coin}",
            "horizon": f"eq.{horizon}",
            "order": "created_at.desc",
            "limit": 1,
            "offset": 1,  # skip the row we just inserted for this scan
        },
        timeout=_TIMEOUT_S,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return
    prior = rows[0]

    outcome_check = requests.get(
        f"{base_url}/rest/v1/signal_outcomes",
        headers=headers,
        params={"select": "id", "coin_signal_id": f"eq.{prior['id']}", "limit": 1},
        timeout=_TIMEOUT_S,
    )
    outcome_check.raise_for_status()
    if outcome_check.json():
        return  # already scored

    price_then = float(prior["live_price"])
    pct_change = (live_price - price_then) / price_then
    predicted_up = prior["live_status"] in ("BREAKOUT_ACTIVE", "UPTREND_WAITING")
    was_correct = (pct_change > 0) if predicted_up else (pct_change <= 0)

    insert_resp = requests.post(
        f"{base_url}/rest/v1/signal_outcomes",
        headers=headers,
        json={
            "coin_signal_id": prior["id"],
            "price_then": price_then,
            "price_now": live_price,
            "pct_change": pct_change,
            "was_correct": was_correct,
        },
        timeout=_TIMEOUT_S,
    )
    insert_resp.raise_for_status()


def push_scan(scan_data: dict, supabase_url: str, service_role_key: str) -> None:
    base_url = supabase_url.rstrip("/")
    headers = _headers(service_role_key)

    scan_resp = requests.post(
        f"{base_url}/rest/v1/scans",
        headers={**headers, "Prefer": "return=representation"},
        json={"generated_at": scan_data["generated_at"]},
        timeout=_TIMEOUT_S,
    )
    scan_resp.raise_for_status()
    scan_id = scan_resp.json()[0]["id"]

    pushed, skipped, scored_errors = 0, 0, 0
    for coin, horizons in scan_data["coins"].items():
        for horizon_name, data in horizons.items():
            if data.get("status") != "ok":
                skipped += 1
                continue

            m = data["holdout_metrics"]
            row = {
                "scan_id": scan_id,
                "coin": coin,
                "horizon": horizon_name,
                "status": data["status"],
                "edge_validated": data["edge_validated"],
                "live_price": data["live_price"],
                "live_status": data["live_status"],
                "donchian_breakout_level": data.get("donchian_breakout_level"),
                "buy_and_hold_pct": data["buy_and_hold_pct"],
                "entry_lookback": data["params"]["entry_lookback"],
                "exit_lookback": data["params"]["exit_lookback"],
                "chandelier_mult": data["params"]["chandelier_mult"],
                "holdout_num_trades": m.get("num_trades"),
                "holdout_sharpe": m.get("sharpe_annualized"),
                "holdout_total_return_pct": m.get("total_return_pct"),
                "n_candles": data["n_candles"],
            }
            resp = requests.post(f"{base_url}/rest/v1/coin_signals", headers=headers, json=row, timeout=_TIMEOUT_S)
            resp.raise_for_status()
            pushed += 1

            try:
                _score_prior_signals(base_url, headers, coin, horizon_name, data["live_price"])
            except requests.RequestException:
                logger.exception("Failed to score prior signal for %s/%s", coin, horizon_name)
                scored_errors += 1

    logger.info(
        "Supabase sync: scan_id=%s pushed=%d skipped=%d score_errors=%d",
        scan_id, pushed, skipped, scored_errors,
    )
