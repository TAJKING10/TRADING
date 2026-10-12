# AI crypto trading toolkit

A research + execution toolkit for Kraken crypto trading: data pipeline, an
ML classifier kept for research/comparison, a rule-based trend-following
strategy (Donchian breakout + EMA filter) that's what `run`/`signal`/`scan`
actually use live, a persistent risk/kill-switch layer, dry-run-safe order
execution, a daily WhatsApp digest, and a multi-coin dashboard.

The ML classifier (`train`/`backtest` commands) failed honest holdout
validation (see below) and is **not** used by the live loop or daily
digest — those both run the same trend-following logic as the dashboard,
for consistency. It's kept in the repo as a research tool / comparison
baseline, not because it's trusted with real signals.

## Read this first

Every backtest number in this repo comes from a strict methodology: parameters
are tuned on one slice of history ("dev"), then checked against a holdout
slice that tuning never saw. As of the last scan, **0 of 15 coins across 2
timeframes showed a validated edge over simple buy-and-hold** — active
trading lost to just holding, because the test period was a strong bull run
and stop-losses/breakout exits gave back gains that holding would have kept.
That's the honest, current state of the signal quality. The dashboard will
say "NO VALIDATED EDGE" until a strategy actually earns a "VALIDATED EDGE"
label by beating buy-and-hold on data it was never tuned on. Don't trust a
"VALIDATED EDGE" label blindly either — re-check it periodically, markets
change regime.

**DRY_RUN=true by default.** With dry-run on, the bot reads real balances
and prices (if you give it API keys) but never sends a real order — it only
logs/reports what it *would* do. Flip `DRY_RUN=false` only once you've run
`smoke-test` and understand the risk controls below.

## Project layout

```
ai_trader/
  config/      settings.py — all behavior is controlled via .env
  data/        Kraken (live prices/execution) + Binance (deep training history)
  features/    technical indicators + trend-following signals
  model/       LightGBM direction classifier, walk-forward + holdout training
  backtest/    honest backtesters (real fees+slippage) for both strategies
  risk/        position sizing + persistent kill switch
  execution/   order placement wrapper (precision checks, dry-run)
  core/        live trading loop, daily digest, WhatsApp notifier, smoke test
  scan/        multi-coin universe scanner
  web/         static dashboard generator
site/          generated dashboard (index.html + scan_results.json) — this is
               what gets deployed to Vercel
```

## Local setup

```bash
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash; use venv\Scripts\activate.bat for cmd.exe
pip install -r requirements.txt
cp .env.example .env              # defaults are safe (dry-run, no keys needed for most commands)
```

## Commands

| Command | What it does | Needs API keys? |
|---|---|---|
| `python main.py backtest` | ML classifier walk-forward train + honest backtest | No |
| `python main.py trend-backtest` | Donchian breakout strategy backtest | No |
| `python main.py scan` | Multi-coin honest scan, writes `site/` dashboard | No |
| `python main.py train` | Train and save an ML model version to `artifacts/` | No |
| `python main.py signal` | Today's trade recommendation (reads your real balance, places no orders) | Yes |
| `python main.py smoke-test` | Validates order precision/kill-switch plumbing, no real orders | Yes |
| `python main.py run` | Starts the live loop (dry-run or real, per `.env`) | Yes |
| `python main.py reset-risk` | Clears a tripped kill switch after you've investigated why | No |

## Getting Kraken API keys

Kraken account → Settings → API → generate a key with Query Funds + Create/Cancel
Orders permissions. Put them in `.env` as `API_KEY` / `API_SECRET`. Leave
`DRY_RUN=true` while testing — reading balances needs auth, but no order is
ever sent in dry-run.

## Daily WhatsApp digest

One-time setup (from your own phone, not here): WhatsApp `+34 644 59 71 30`
with the message `I allow callmebot to send me messages`. It replies with a
personal API key — put it in `.env` as `WHATSAPP_PHONE` / `WHATSAPP_APIKEY`,
then `python main.py signal` will push the digest to WhatsApp as well as
writing it to `logs/digests/`.

## Deployment: GitHub → Vercel

The **dashboard only** (public market data, no account info, no secrets) is
meant to be deployed. The live trading bot (`run`, `signal`, `smoke-test` —
anything touching real API keys) is NOT deployed anywhere; it keeps running
on your own machine (or a VPS later), because it needs to hold credentials
and persist state continuously, which a public static host can't do safely.

1. Push this repo to GitHub (`.env` is gitignored — your keys never leave
   your machine).
2. In Vercel: "Add New Project" → import the GitHub repo. Framework preset
   "Other" is fine — `vercel.json` already points Vercel at the `site/`
   folder as static output, so no build step runs on Vercel at all.
3. `.github/workflows/scan.yml` re-runs the scan every 6 hours (and on-demand
   via the Actions tab's "Run workflow" button), commits the refreshed
   `site/` folder back to `main`, and Vercel auto-redeploys on every push.
   This workflow needs no secrets — it only reads public market data.

**Known risk:** Binance (used for deep training history) occasionally
geo-blocks certain cloud IP ranges, which could include GitHub's hosted
runners. If the scheduled scan's Actions log shows connection errors instead
of real data, go to the repo's Settings → Secrets and variables → Actions →
Variables and add `TRAINING_DATA_EXCHANGE` set to another exchange (e.g.
`okx` or `bybit`) — no code change needed.

## Safety notes

- Never commit `.env` — it's gitignored. `.env.example` has no real values.
- The kill switch (`risk/risk_manager.py`) persists to `logs/risk_state.json`
  so a crash-and-restart can't silently bypass it. After it trips, you must
  run `reset-risk` and explicitly confirm before trading resumes.
- `DRY_RUN=false` is the only thing that makes real orders possible. Treat
  flipping it as a deliberate, one-time decision, not a default.
