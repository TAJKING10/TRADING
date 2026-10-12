# Project Handoff — AI Crypto Trading Scanner

**Last updated:** 2026-10-13, end of session. Everything in this document is
verified against the actual repo/GitHub/Supabase state at that time unless
explicitly marked "unconfirmed" or "pending." If you're a future Claude
session picking this up: read section 12 first (exact current state), then
section 13 (roadmap) for what to do next.

---

## 1. What this project is

A rule-based (not ML) crypto trend-following signal scanner. Hourly, a
GitHub Actions job backtests a Donchian-breakout strategy against every
coin Kraken lists that also has deep enough history on KuCoin, validates
each result honestly (tuned on one slice, checked on a holdout slice it
never saw), and publishes to a public static dashboard. **Signal-only — no
real orders are ever placed by this pipeline.** A separate, unconnected
live/paper-trading code path exists in the repo (`ai_trader/core/trader.py`
etc.) but is not wired to the scan, the validated parameters, or any
automation.

**Live site:** https://tradingcryptobytoto.vercel.app/
**Repo:** https://github.com/TAJKING10/TRADING (public)
**Supabase project:** `aitrackercrypto` (ref `iiezjeqbxqvlkjkjlnws`)

---

## 2. Strategy & indicators

One strategy in the automated path: **Donchian channel breakout + EMA trend
filter + ATR chandelier trailing stop.** No ML in this path — a separate
LightGBM classifier exists (`ai_trader/model/train.py`) but **failed its own
honest holdout validation** and is not used by the scan, dashboard, or live
trader.

- **Entry:** close breaks above the N-bar rolling high (shifted 1 bar, no
  lookahead) AND EMA-fast > EMA-slow.
- **Exit:** whichever triggers first — ATR chandelier trailing stop, EMA
  trend flip, or close below the M-bar rolling low.
- **Parameter grid:** entry ∈ {10,20,40,55} × exit ∈ {10,20} × ATR mult ∈
  {2,3,4} = 24 combos, tuned on an 85% "dev" slice only, ranked by Sharpe.
  The winner is checked **once** on the untouched 15% holdout.
- **Validation bar ("edge_validated"):** holdout trades ≥ 5, holdout Sharpe
  ≥ 0.5, AND holdout return beats simple buy-and-hold over the same window.
- **Tiers shown on dashboard:** Strong (validated + Sharpe≥1.0 + win≥50% +
  trades≥10), Moderate (validated, below those), High-risk (not validated
  but a breakout is live right now), No-signal.

Files: `ai_trader/features/trend_signals.py` (indicators),
`ai_trader/backtest/trend_backtester.py` (simulator),
`ai_trader/scan/run_scan.py` (orchestration).

---

## 3. Data pipelines

- **Kraken** (`ai_trader/data/market_data.py::make_exchange`) — live
  execution-side prices. This is where you'd actually trade.
- **KuCoin** (`make_training_data_exchange`) — deep historical OHLCV for
  backtesting. **Not Binance** — Binance returns HTTP 451 and geo-blocks
  GitHub Actions' runner IPs (discovered + fixed this session, see §4).
- **Universe** (`ai_trader/scan/universe.py::resolve_universe`) — dynamically
  resolved every run: every Kraken active `/USD` spot pair that also has an
  active KuCoin `/USDT` pair. Currently ~431 coins. Stablecoins/fiat
  excluded. Not a fixed list — new Kraken listings get picked up
  automatically.

---

## 4. Complete bug-fix history (chronological, this session)

| Commit | What | Confirmed how |
|---|---|---|
| `da377bd` | Added Supabase-backed signal history + accuracy tracking | Schema created, tested with dummy inserts |
| `17ebe1b` | Universe scan expanded from 16 hardcoded coins to full dynamic Kraken∩Binance overlap (262 coins) | — |
| `2c7244d` | Tiered filter UI, real trailing-stop numbers added to dashboard | Browser-verified |
| `20ec0a9` | **Bug:** Binance HTTP 451 geo-blocks GitHub Actions. **Fix:** switched training-data exchange to KuCoin (also expanded universe to ~431 coins, more than Binance covered) | Verified end-to-end locally before push; confirmed in production after a real scan failure (`Run #1`, 38s, `ExchangeNotAvailable: 451`) |
| `7fc8085` | Added per-coin price sparklines (Supabase `coin_price_history` view), hourly cron (was 6h), horizon-aware signal_outcomes scoring (was scoring against the literally-previous scan, meaningless once hourly) | Rendered + verified in browser against real Supabase data |
| `b50f713` | **Bug:** backtest fee hardcoded to Binance's 0.1% taker fee, but Kraken (the real execution exchange) is 0.80% base-tier. **Fix:** made fee/slippage configurable (`TAKER_FEE_PCT`, `SLIPPAGE_PCT`), defaulted to Kraken's real rate | Verified against Kraken's published fee schedule via web fetch; before/after comparison showed lower returns with correct (higher) fees |
| `091ed29` | Per-trade logging (`trade_log`, `rejected` lists with full entry/exit/fee/R-multiple/MFE-MAE detail), new `backtest_trades` Supabase table, real BTC-correlation computation, "Statistical reliability" dashboard panel (explicitly labels "effective independent sample size" as **Not established** rather than fabricating one) | 7 new pytest tests, incl. aggregate-metrics-reconciles-with-trade-log |
| `f45d911` | **Bug (the big one): same-candle execution.** A breakout confirmed at bar i's close was filling at that same close — a price only knowable after the fact. Chandelier stops had the same issue, plus only checked the close (missed intrabar touches). **Fix:** 3-phase fill model — entries/trend-flip/Donchian exits confirm at close, fill at next bar's OPEN; chandelier stop is a standing order checked INTRABAR against each bar's low, gap-aware (fills at open if gapped through). Also renamed dashboard "near"/"far" labels to "4H"/"1D" (a user nearly bought BTC thinking a "NEAR" badge meant the NEAR Protocol coin) | 3 new tests (next-open fill, intrabar stop trigger, gap-aware fill). Controlled before/after on real ACT/USDT 4h data (same params, same fetched df): return -6.62%→-3.8%, same 3 trades. 10/10 tests passing at this point. |
| `d090aef` | **Bug: Sharpe annualization hardcoded to 15-minute bars** (`sqrt(365*24*4)`), applied uniformly to 4h/1d trend-strategy data too. Overstated every Sharpe by 4x (4h) to ~9.8x (1d), directly affecting the `Sharpe≥0.5` validation bar. **Fix:** derive periods-per-year from the equity curve's actual median bar spacing. | 2 new tests. Verified: ACT/4h Sharpe -27.47→-6.26 on identical data. 12/12 tests passing. |
| `b1838c5` | README was describing the pre-session state (Binance, 15 coins, 6h cadence). Synced with reality; added explicit "not done yet" scope list. Added EMA/ATR no-lookahead regression test. | 13/13 tests passing (current total) |
| `4b9247a` | Added `HANDOFF.md` (this file) | — |
| `30e1d5d` | **Bug: scan workflow's `git push` had no resilience to `origin/main` advancing during the 15-20+ min scan run.** Discovered when scan #5's backtest succeeded cleanly (21m32s, zero errors) but the final push was rejected (non-fast-forward) because other commits landed while it ran. **Fix:** `git pull --rebase` + retry up to 5x before giving up, in the "Commit updated dashboard" step. | Confirmed root cause via Actions log (non-fast-forward rejection, not a Python error); re-triggered as scan #6 to verify the fix |

**Additional controlled before/after validation** (not a commit, done as
verification): same fetched historical data run through pre-fix vs
post-fix code for 6 real coins (ACT, AKT, ANIME, AR, ARC, ARPA on 4h).
Every coin showed the Sharpe correction in the predicted direction/magnitude.
No validated-edge status flips in this sample.

---

## 5. Supabase schema

Project `aitrackercrypto`, 4 tables, all RLS-enabled with public
**read-only** policies (writes require the service-role key, used only
server-side in GitHub Actions — never committed, never in client code).

- **`scans`** — one row per scan run (`generated_at`).
- **`coin_signals`** — one row per coin×horizon with an "ok" result.
  Columns: `scan_id`, `coin`, `horizon`, `status`, `edge_validated`,
  `live_price`, `live_status`, `donchian_breakout_level`,
  `buy_and_hold_pct`, `entry_lookback`, `exit_lookback`, `chandelier_mult`,
  `holdout_num_trades`, `holdout_sharpe`, `holdout_total_return_pct`,
  `n_candles`, `atr`, `avg_holding_hours`, `btc_correlation`, `created_at`.
- **`backtest_trades`** — one row per simulated round-trip trade (dev or
  holdout, FK to `coin_signals`). Full entry/exit/fee/PnL/R-multiple/MFE-MAE
  detail. **1,506 rows as of last check** (confirms per-trade logging is
  populating Supabase correctly at full scale).
- **`signal_outcomes`** — one row per signal, scored once its own horizon
  window (4h "near", 24h "far") has actually elapsed. **0 rows as of last
  check** — expected, needs more scan cycles to accumulate (first-ever sync
  only recently happened).

Two Postgres views: `coin_accuracy` (hit-rate by coin/horizon/signal type),
`coin_price_history` (last 48 points per coin/horizon, for sparklines).

---

## 6. Deployment

- **Vercel**: static output only (`site/`), no serverless functions.
  Auto-deploys on push to `main`. **Known gotcha, already fixed:** Hobby
  plan blocks deploys from a commit author who isn't a recognized project
  collaborator on a *private* repo — this repo was made **public** to
  resolve it (confirmed working after that change).
- **GitHub Actions** (`.github/workflows/scan.yml`): `cron: "0 * * * *"`
  (hourly, free on public repos), plus manual `workflow_dispatch`. Installs
  `requirements.txt`, runs `python main.py scan`, commits `site/` if
  changed. Secrets used: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
  (repo secrets, already configured). Variables (optional overrides):
  `EXCHANGE`, `TRAINING_DATA_EXCHANGE`, `TAKER_FEE_PCT`, `SLIPPAGE_PCT`.

---

## 7. Dashboard features (`ai_trader/web/dashboard.py`)

Server-generated static HTML + client-side JS reading Supabase directly
(publishable key, read-only, safe to expose):
- Per-coin/horizon cards: tier badge, live price, holdout metrics, trailing
  stop price (from the strategy's own chandelier formula), BTC correlation,
  avg hold time.
- Filter bar (All/Strong/Moderate/High-risk/No-signal), client-side JS.
- Price sparklines (SVG, client-side, fetched from `coin_price_history`).
- "Live track record" panel (fetched from `coin_accuracy`).
- "Statistical reliability" panel — universe/tier counts, total holdout
  trades, median BTC correlation, count highly correlated (|r|≥0.7),
  explicit "Not established" label for effective independent sample size.
- Horizon labels are "4H"/"1D" (not "near"/"far" — see §4, `f45d911`).

---

## 8. Tests (`tests/`, 13 total, all passing as of `b1838c5`)

Run with: `pip install -r requirements-dev.txt && pytest tests/ -v`

`tests/test_trend_backtester.py` (11 tests) — deterministic synthetic price
series, no network: trade opens/closes correctly, fees charged both sides,
MFE/MAE bounds, R-multiple sign matches PnL, **aggregate metrics reconcile
exactly with trade_log**, Donchian no-lookahead, **EMA/ATR no-lookahead**,
**entry fills at next-bar open not signal-bar close**, **chandelier stop
triggers intrabar even if close recovers**, **gap-through-stop fills at
open not stale level**, rejected-trade logging.

`tests/test_backtester_metrics.py` (2 tests) — Sharpe annualization factor
matches actual bar spacing exactly (2190/yr for 4h, 365/yr for 1d); not
anywhere near the old hardcoded 15-min value.

---

## 9. Key design decisions & why

- **Rule-based strategy chosen over ML for the live path** — the LightGBM
  model failed its own honest holdout validation; shipping it anyway would
  mean fabricating confidence. Kept in repo as a comparison baseline only.
- **"Not established" over a fabricated number** — when effective
  independent sample size can't be defensibly computed (cross-coin
  correlation isn't modeled), the dashboard says so explicitly rather than
  showing a precise-looking but misleading number.
- **Standing-order vs signal-decision fill timing** — chandelier stops are
  checked intrabar (a resting order, realistic), while breakout/trend-flip/
  Donchian-exit signals fill at the next bar's open (can't react until the
  candle that confirmed them has closed). Conflating these would make stops
  unrealistically slow or signals unrealistically fast — see `f45d911`.
- **KuCoin over Binance for training data** — not a preference, a hard
  requirement: Binance returns HTTP 451 for GitHub Actions IPs. KuCoin also
  happens to cover more of the Kraken universe (431 vs 262 coins).
- **Dev/holdout run as fully independent simulations** — no state (cash,
  open position) carries from dev into holdout. Appropriate for the
  holdout's purpose (clean validation), but means a dev-dangling open
  position isn't reflected in holdout. Documented as a design assumption,
  not fixed as a bug (see §10).

---

## 10. Known limitations (confirmed, not yet fixed)

- **EMA200/ATR "warm-up"** — both are `ewm`-based, so numerically defined
  from bar 1, but not functionally stabilized until ~200 bars accumulate.
  No explicit warm-up skip exists. Confirmed via code audit, not yet fixed.
- **Train/holdout boundary** — dev and holdout are independent simulations;
  a dev position still open at the boundary is silently dropped, not
  reflected in holdout. Consistent with the holdout's validation purpose,
  but worth knowing.
- **Survivorship bias** — universe is today's listed coins; delisted/
  collapsed coins aren't represented in any backtest.
- **KuCoin-vs-Kraken price divergence** — not measured. Training data and
  execution price come from different exchanges; no divergence report
  exists yet (this was explicitly requested, see §13 P0).
- **Supabase sync has no per-coin error isolation** — one failed insert
  aborts the rest of that hour's sync (self-heals next hour; the
  dashboard/site write always succeeds regardless, since Supabase sync
  failure is caught at the top level in `main.py::cmd_scan`).
- **Same-bar signal-to-close assumption remains for indicator computation**
  itself (not the fill, which is now next-bar) — i.e., `breakout_long` at
  bar i still uses bar i's own close vs. prior-bar Donchian levels, which
  is standard and not a lookahead bug (tested), just noting it's a design
  choice, not something "left broken."

---

## 11. Attempted but not completed

- **P0-C: KuCoin/Kraken historical divergence report** — explicitly
  requested, not built. Would need to fetch both exchanges' history for
  overlapping coins and compare price levels/returns/timing.
- **Walk-forward backtesting** — still a single 85/15 split, not multiple
  rolling periods.
- **Strategy comparison** (momentum, mean-reversion vs. Donchian) — not
  built.
- **Portfolio-level risk management** — each coin evaluated independently,
  no cross-coin correlation limits or aggregate exposure caps.
- **Forward paper trading** — no simulated live-order journal.
- **ML vs. rule-based comparison** under identical validation standards —
  not done.
- **Live execution** — correctly not connected to anything (by design,
  confirmed in this session's instructions not to do this yet).

---

## 12. Exact current state — VERIFIED, updated after scan #5/#6

**Git:**
- Branch: `main`
- Latest commit (local AND origin, confirmed identical): `30e1d5d` — "Fix scan workflow: git push had no resilience to origin/main moving mid-run"
- Working tree: clean, 0 uncommitted changes, 0 commits ahead/behind origin.

**Tests:** 13/13 passing (`pytest tests/ -v`, confirmed at commit `b1838c5`; no test-affecting changes since).

**GitHub Actions scan runs** (updated):
- Run #1 (commit `20ec0a9`-era, before KuCoin fix): **FAILED** (38s, Binance HTTP 451) — the original bug discovery.
- Run #2: SUCCESS (16m2s) — first successful KuCoin-based run.
- Run #3: SUCCESS (15m49s) — charts + hourly cron + horizon-aware scoring.
- Run #4 (commit `091ed29`): SUCCESS (15m19s) — per-trade logging, BTC correlation, reliability panel. Live site served this data for a while.
- Run #5 (commit `f45d911`): **FAILED, but not a code bug.** The scan step itself succeeded in 21m32s with zero errors — the fill-timing fix ran correctly at full scale. The failure was entirely in the final `git push`: `origin/main` had advanced during the 20+ min run (concurrent pushes: Sharpe fix, README, HANDOFF.md), so the plain non-retrying push was rejected. **Fixed in `30e1d5d`** — the commit step now does `git pull --rebase` + retry (up to 5x) before giving up.
- Run #6 (commit `30e1d5d`, includes fill-timing + Sharpe + 4H/1D label + the git-retry fix): dispatched, result **pending — check `https://github.com/TAJKING10/TRADING/actions` for current status before doing anything else.**

**If Run #6 succeeded:** the fill-timing fix, Sharpe fix, and label fix are
now FULLY verified at production scale, not just via unit tests. Pull
`scan_results.json` and Supabase row counts to get fresh tier counts
(last known pre-fix baseline: 0 Strong, 0 Moderate, 2 High-risk, 860
No-signal, out of 862 slots — the Sharpe fix could shift this).

**If Run #6 also failed:** check whether it's the same git-push race (the
retry fix not working as intended — investigate why) or a new/different
error. Don't just re-trigger blindly; read the actual logs first.

**Supabase (as of before run #6):**
- `scans`: 3 rows, latest `generated_at` = `2026-10-12 23:19:19` (Run #4's data).
- `coin_signals`: 621 rows. `backtest_trades`: 1,506 rows (confirms per-trade logging works at scale). `signal_outcomes`: 0 rows (expected — needs more scan cycles to accumulate scored outcomes).
- These numbers will be higher after Run #6 lands — re-query before trusting any figure above as current.

**Live site** (https://tradingcryptobytoto.vercel.app/): was serving Run
#4's data (pre-fill-timing-fix, pre-Sharpe-fix, "NEAR"/"FAR" labels) as of
this update. **Re-check after confirming Run #6's outcome** — don't assume
the live site reflects the latest code without verifying `scan_results.json`'s
`generated_at` timestamp first.

---

## 13. Development roadmap

### P0 — Finish verifying today's fixes (not yet done, do this first)

1. Confirm Run #5 completed (success or failure — check the Actions tab).
2. Trigger Run #6 at commit `b1838c5` or later (includes fill-timing +
   Sharpe + label fixes together).
3. Once #6 succeeds: re-pull `scan_results.json`, confirm 4H/1D labels
   render, spot-check a few coins' Sharpe values look corrected (smaller
   magnitude than before), report new tier counts (Strong/Moderate/
   High-risk/No-signal) since the Sharpe fix could change which coins
   validate.
4. Confirm Supabase `coin_signals`/`backtest_trades` row counts grew
   consistent with #6's coin count, and spot-check a `backtest_trades` row
   for sane fee/PnL/R-multiple values.

### P1 — Next substantial work, in priority order

1. **KuCoin/Kraken divergence report** — compare overlapping historical
   candles, measure return differences and breakout-signal disagreement,
   flag assets where KuCoin history is an unreliable proxy for Kraken
   execution. Not yet started.
2. **Walk-forward backtesting** — multiple chronological folds instead of
   one 85/15 split. Optimize only on each fold's training portion,
   evaluate on unseen periods, aggregate out-of-sample results, report
   per-fold performance and flag unstable parameter combinations. Preserve
   the original single-holdout report for comparison rather than replacing
   it outright.
3. **Supabase sync reliability** — per-coin/batched error isolation so one
   bad insert doesn't abort the rest of an hour's sync; idempotent
   upserts; safe retries; tests for API failures, missing candles, stale
   prices, duplicate jobs.
4. **Expand regression/failure test coverage** — beyond the current 13,
   add tests for failed API calls, invalid exchange responses, database
   insertion failures.
5. **Survivorship-bias assessment** — not resolved; would need
   point-in-time listing data this project doesn't currently have access
   to.

### P2 — Only after P0/P1 are solid

- Portfolio-level backtesting (correlation-aware position limits, shared
  capital constraints across simultaneous signals).
- Forward paper trading (real hypothetical order lifecycle against live
  market data).
- Additional strategies (momentum, mean-reversion) benchmarked against
  Donchian breakout under identical validation standards.
- ML (LightGBM) vs. rule-based comparison under identical standards —
  deploy only if it shows measurable out-of-sample improvement.
- Live execution — explicitly not yet, and not until a validated edge
  actually exists. The separate `trader.py` path stays disconnected.

**Guiding instruction carried forward from this session, worth repeating
to any future session:** do not alter validation thresholds to manufacture
more "Strong"/"Moderate" results. If no validated edge exists after all of
the above, that is the system working correctly, not a reason to loosen
the bar. The goal is a system that can reliably discover whether a
profitable strategy exists — not one that looks more impressive.

---

## 14. How to resume tomorrow

1. Read this file.
2. Check Actions tab for Run #5/#6 status (§12 had them unresolved at
   session end) — https://github.com/TAJKING10/TRADING/actions
3. If neither succeeded cleanly, trigger a fresh run at the latest commit
   before doing anything else — this is the P0 item.
4. Re-run `pytest tests/ -v` locally to confirm nothing regressed since
   `b1838c5`.
5. Proceed to P1 roadmap item 1 (KuCoin/Kraken divergence report) once P0
   is clean, per the specialist review's recommended order.
