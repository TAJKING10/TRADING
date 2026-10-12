# AI crypto trading toolkit

A research + execution toolkit for Kraken crypto trading: data pipeline, an
ML classifier kept for research/comparison, a rule-based trend-following
strategy (Donchian breakout + EMA filter) that's what `run`/`signal`/`scan`
actually use live, a persistent risk/kill-switch layer, dry-run-safe order
execution, a daily WhatsApp digest, and a multi-coin dashboard with a
Supabase-backed prediction history.

The ML classifier (`train`/`backtest` commands) failed honest holdout
validation (see below) and is **not** used by the live loop or daily
digest — those both run the same trend-following logic as the dashboard,
for consistency. It's kept in the repo as a research tool / comparison
baseline, not because it's trusted with real signals.

## Read this first

Every backtest number in this repo comes from a strict methodology: parameters
are tuned on one slice of history ("dev"), then checked against a holdout
slice that tuning never saw. Costs are modeled at Kraken's real base-tier
spot taker fee (0.80%, not a discount rate) plus slippage, on both entry and
exit. Fills are timed realistically — a signal confirmed at a candle's close
executes at the *next* candle's open, not the same close; the chandelier
stop is checked intrabar (gap-aware) since it's a standing order, not a
close-confirmed signal.

As of the last full scan, **0 of 862 coin/horizon slots across the entire
scanned Kraken universe (431 coins) showed a validated edge over simple
buy-and-hold.** That's the honest, current state of the signal quality —
see the dashboard's live "Statistical reliability" panel for current counts,
including how many coins are highly correlated with BTC (a correlated group
moving together isn't independent evidence, however many coins are in it).
The dashboard will say "NO VALIDATED EDGE" until a strategy actually earns
a "VALIDATED EDGE" label by beating buy-and-hold on data it was never tuned
on. Don't trust a "VALIDATED EDGE" label blindly either — re-check it
periodically, markets change regime.

**What this system has not yet done, to be explicit about scope:**
walk-forward validation across multiple market periods (currently a single
85/15 split), a KuCoin-vs-Kraken historical price divergence check, a
portfolio-level risk model (each coin is still evaluated independently),
forward paper trading, or a head-to-head comparison against the ML model
under identical validation standards. None of this is connected to real
money — `run`/`signal` exist as a separate, unconnected code path (see
below) and require you to explicitly supply API keys and flip `DRY_RUN`.

**DRY_RUN=true by default.** With dry-run on, the bot reads real balances
and prices (if you give it API keys) but never sends a real order — it only
logs/reports what it *would* do. Flip `DRY_RUN=false` only once you've run
`smoke-test` and understand the risk controls below.

## Project layout

```
ai_trader/
  config/      settings.py — all behavior is controlled via .env
  data/        Kraken (live prices/execution) + KuCoin (deep training history —
               not Binance, which geo-blocks GitHub Actions' runner IPs)
  features/    technical indicators + trend-following signals
  model/       LightGBM direction classifier, walk-forward + holdout training
  backtest/    honest backtesters (real Kraken fees+slippage, realistic fill
               timing — next-open fills, intrabar/gap-aware stops) for both
               strategies
  risk/        position sizing + persistent kill switch
  execution/   order placement wrapper (precision checks, dry-run)
  core/        live trading loop, daily digest, WhatsApp notifier, smoke test
  scan/        multi-coin universe scanner — dynamically resolves every
               Kraken/USD pair with KuCoin/USDT history (~431 coins), pushes
               results + per-trade logs + BTC correlation to Supabase
  web/         static dashboard generator — tiers, filters, price sparklines,
               live track record, statistical reliability panel
site/          generated dashboard (index.html + scan_results.json) — this is
               what gets deployed to Vercel
tests/         pytest suite for the backtester (fill timing, no-lookahead,
               Sharpe annualization, aggregate/trade-log reconciliation)
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

## Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests run against a hand-built deterministic synthetic price series, not
live data — no network access needed, same result every run. Covers fill
timing (next-open fills, intrabar/gap-aware stops), no-lookahead checks on
every indicator, Sharpe annualization, and that aggregate metrics
reconcile exactly with the per-trade log.

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
   your machine). The repo must be **public** on the Hobby plan — Vercel
   blocks auto-deploy for commits from an author who isn't a recognized
   project collaborator on a private repo.
2. In Vercel: "Add New Project" → import the GitHub repo. Framework preset
   "Other" is fine — `vercel.json` already points Vercel at the `site/`
   folder as static output, so no build step runs on Vercel at all.
3. `.github/workflows/scan.yml` re-runs the scan **every hour** (free on
   GitHub Actions for public repos; also runnable on-demand via the Actions
   tab's "Run workflow" button), commits the refreshed `site/` folder back
   to `main`, and Vercel auto-redeploys on every push. This workflow needs
   no secrets for the scan itself — only public market data is read.

**Training data exchange:** uses KuCoin, not Binance — Binance returns
HTTP 451 and geo-blocks GitHub Actions' runner IPs. If KuCoin ever has the
same issue, override via the repo's Settings → Secrets and variables →
Actions → Variables, adding `TRAINING_DATA_EXCHANGE` set to another
ccxt-supported exchange — no code change needed. Same mechanism exists for
`TAKER_FEE_PCT`/`SLIPPAGE_PCT` if your Kraken account is on a different fee
tier than the base rate this defaults to.

## Optional: Supabase (prediction history + live track record)

Without Supabase configured, `scan` still works — it just skips history
sync and the dashboard's live track-record/reliability panels show no data.
To enable it: create a Supabase project, and set `SUPABASE_URL` +
`SUPABASE_SERVICE_ROLE_KEY` (repo secrets for the GitHub Actions workflow,
or your local `.env`). The service-role key is server-side only — never
put it in client code or commit it. The dashboard embeds a separate
*publishable* key for read-only browser queries, which is safe to expose.

## Safety notes

- Never commit `.env` — it's gitignored. `.env.example` has no real values.
- The kill switch (`risk/risk_manager.py`) persists to `logs/risk_state.json`
  so a crash-and-restart can't silently bypass it. After it trips, you must
  run `reset-risk` and explicitly confirm before trading resumes.
- `DRY_RUN=false` is the only thing that makes real orders possible. Treat
  flipping it as a deliberate, one-time decision, not a default.
