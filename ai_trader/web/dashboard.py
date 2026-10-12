"""Generates a static, honest dashboard — no fabricated confidence scores.
Every coin/horizon shows real holdout-tested numbers, or says plainly that
no validated edge was found. This is a local file you open in your browser;
it is not deployed anywhere and contains no account/balance data.
"""
import html
import json
from pathlib import Path

STATUS_LABELS = {
    "BREAKOUT_ACTIVE": ("🟢 Breakout active", "#1a7f37"),
    "UPTREND_WAITING": ("🟡 Uptrend, waiting for breakout", "#9a6700"),
    "NO_SIGNAL": ("⚪ No signal", "#6e7781"),
}

# Tier is computed purely from objective backtest numbers already produced by
# the scan — not a personal call. "High risk / high reward" means exactly
# that: live momentum exists but the holdout backtest did NOT validate an
# edge, so there is no evidence behind it beyond the current breakout itself.
TIER_LABELS = {
    "strong": ("🟩 STRONG — validated edge", "#1a7f37"),
    "moderate": ("🟨 MODERATE — validated edge", "#9a6700"),
    "high_risk": ("🟧 HIGH RISK / HIGH REWARD — momentum, unproven edge", "#bf3989"),
    "none": ("⬜ NO SIGNAL", "#6e7781"),
}

# Candle hours per horizon — must match HORIZONS in ai_trader/scan/run_scan.py.
HORIZON_HOURS = {"near": 4, "far": 24}


def _classify_tier(data: dict) -> str:
    if data.get("status") != "ok":
        return "none"
    m = data["holdout_metrics"]
    if data["edge_validated"]:
        strong = (
            m.get("sharpe_annualized", 0) >= 1.0
            and m.get("win_rate_pct", 0) >= 50
            and m.get("num_trades", 0) >= 10
        )
        return "strong" if strong else "moderate"
    if data["live_status"] == "BREAKOUT_ACTIVE":
        return "high_risk"
    return "none"


def _format_price(price: float) -> str:
    decimals = 2 if price >= 100 else 4 if price >= 1 else 6
    return f"${price:,.{decimals}f}"


def _format_hours(hours) -> str:
    if hours is None:
        return "—"
    if hours < 48:
        return f"~{hours:.0f}h"
    return f"~{hours / 24:.1f}d"


def _horizon_card(base: str, horizon_name: str, data: dict) -> str:
    status = data.get("status")
    if status != "ok":
        reason = {"insufficient_data": "Not enough history yet", "no_signal": "No tradable setups in test data",
                   "error": "Scan error"}.get(status, status)
        return f"""
        <div class="horizon-card muted" data-tier="none">
          <div class="horizon-title">{horizon_name.upper()}</div>
          <div class="na">— {html.escape(str(reason))} —</div>
        </div>"""

    m = data["holdout_metrics"]
    validated = data["edge_validated"]
    badge_class = "validated" if validated else "unvalidated"
    badge_text = "VALIDATED EDGE" if validated else "NO VALIDATED EDGE"
    live_label, live_color = STATUS_LABELS.get(data["live_status"], (data["live_status"], "#6e7781"))
    price = data["live_price"]
    price_str = _format_price(price)

    tier = _classify_tier(data)
    tier_label, tier_color = TIER_LABELS[tier]

    # Real trailing-stop distance from THIS strategy's own chandelier exit —
    # only meaningful if a breakout is active right now (i.e. you'd be
    # entering at ~live price). No fixed take-profit exists in this
    # strategy — it exits on trend flip / chandelier stop / Donchian break,
    # so we don't fabricate a take-profit number.
    atr = data.get("atr")
    chandelier_mult = data["params"]["chandelier_mult"]
    if atr is not None and data["live_status"] == "BREAKOUT_ACTIVE":
        stop_price = price - chandelier_mult * atr
        stop_pct = (price - stop_price) / price * 100
        stop_row = f"""
        <tr><td>Trailing stop (chandelier)</td><td>{_format_price(stop_price)} (-{stop_pct:.1f}%)</td></tr>"""
    else:
        stop_row = """
        <tr><td>Trailing stop (chandelier)</td><td class="na-cell">— no position —</td></tr>"""

    return f"""
    <div class="horizon-card" data-tier="{tier}">
      <div class="horizon-title">{horizon_name.upper()} <span class="badge {badge_class}">{badge_text}</span></div>
      <div class="tier" style="color:{tier_color}">{tier_label}</div>
      <div class="live-status" style="color:{live_color}">{live_label}</div>
      <div class="price">{price_str}</div>
      <table class="metrics">
        <tr><td>Strategy (holdout)</td><td class="{'pos' if m['total_return_pct']>=0 else 'neg'}">{m['total_return_pct']:+.2f}%</td></tr>
        <tr><td>Buy &amp; hold (same window)</td><td class="{'pos' if data['buy_and_hold_pct']>=0 else 'neg'}">{data['buy_and_hold_pct']:+.2f}%</td></tr>
        <tr><td>Sharpe</td><td>{m['sharpe_annualized']:.2f}</td></tr>
        <tr><td>Max drawdown</td><td>{m['max_drawdown_pct']:.2f}%</td></tr>
        <tr><td>Trades (holdout)</td><td>{m['num_trades']}</td></tr>
        <tr><td>Win rate</td><td>{m['win_rate_pct']:.1f}%</td></tr>
        <tr><td>Avg hold time (holdout)</td><td>{_format_hours(m.get('avg_holding_hours'))}</td></tr>
        {stop_row}
      </table>
    </div>"""


def _coin_section(base: str, horizons: dict) -> str:
    cards = "".join(_horizon_card(base, name, data) for name, data in horizons.items())
    tiers = {_classify_tier(data) for data in horizons.values()}
    best_tier = next((t for t in ("strong", "moderate", "high_risk", "none") if t in tiers), "none")
    return f"""
    <section class="coin" data-best-tier="{best_tier}">
      <h2>{html.escape(base)}</h2>
      <div class="horizon-row">{cards}</div>
    </section>"""


SUPABASE_URL = "https://iiezjeqbxqvlkjkjlnws.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "sb_publishable_7_rDP4gV_dK5zgHdYivntQ_DGyuVFgS"


def generate_dashboard_html(scan_data: dict) -> str:
    coins_html = "".join(_coin_section(base, horizons) for base, horizons in scan_data["coins"].items())
    generated_at = html.escape(scan_data["generated_at"])

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trading signal dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1100px; margin: 0 auto; padding: 24px 16px 80px;
    background: Canvas; color: CanvasText;
  }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  .meta {{ color: #6e7781; font-size: 0.85rem; margin-bottom: 8px; }}
  .disclaimer {{
    border: 1px solid #d1242f; background: rgba(209, 36, 47, 0.08);
    border-radius: 8px; padding: 12px 16px; font-size: 0.9rem; margin-bottom: 28px;
  }}
  .coin {{ margin-bottom: 28px; }}
  .coin h2 {{ font-size: 1.1rem; margin-bottom: 10px; border-bottom: 1px solid rgba(110,119,129,0.3); padding-bottom: 4px; }}
  .horizon-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
  .horizon-card {{
    border: 1px solid rgba(110,119,129,0.3); border-radius: 10px; padding: 14px 16px;
  }}
  .horizon-card.muted {{ opacity: 0.6; }}
  .horizon-title {{ font-weight: 600; font-size: 0.85rem; letter-spacing: 0.03em; margin-bottom: 8px;
    display: flex; align-items: center; gap: 8px; }}
  .badge {{ font-size: 0.65rem; padding: 2px 8px; border-radius: 999px; font-weight: 700; }}
  .badge.validated {{ background: rgba(26,127,55,0.15); color: #1a7f37; }}
  .badge.unvalidated {{ background: rgba(110,119,129,0.15); color: #6e7781; }}
  .live-status {{ font-size: 0.9rem; margin-bottom: 4px; }}
  .price {{ font-size: 1.3rem; font-weight: 600; margin-bottom: 10px; }}
  .metrics {{ width: 100%; font-size: 0.85rem; border-collapse: collapse; }}
  .metrics td {{ padding: 3px 0; }}
  .metrics td:last-child {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .pos {{ color: #1a7f37; }}
  .neg {{ color: #d1242f; }}
  .na {{ color: #6e7781; font-style: italic; }}
  .na-cell {{ color: #6e7781; font-style: italic; font-size: 0.8rem; }}
  .tier {{ font-size: 0.8rem; font-weight: 600; margin-bottom: 6px; }}
  .filter-bar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }}
  .filter-btn {{
    font: inherit; font-size: 0.8rem; font-weight: 600; padding: 6px 14px; border-radius: 999px;
    border: 1px solid rgba(110,119,129,0.4); background: transparent; color: CanvasText; cursor: pointer;
  }}
  .filter-btn.active {{ background: rgba(9,105,218,0.15); border-color: #0969da; color: #0969da; }}
  .coin.hidden, .horizon-card.hidden {{ display: none; }}
  .track-record {{ margin-bottom: 28px; }}
  .track-record h2 {{ font-size: 1.1rem; margin-bottom: 10px; }}
  .track-record table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .track-record th, .track-record td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid rgba(110,119,129,0.2); }}
  .track-record td:not(:first-child), .track-record th:not(:first-child) {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .track-record .loading, .track-record .empty {{ color: #6e7781; font-style: italic; font-size: 0.85rem; }}
</style>
</head>
<body>
  <h1>Trading signal dashboard</h1>
  <div class="meta">Generated {generated_at} · "near" = 4h candles, "far" = 1d candles</div>
  <div class="disclaimer">
    <strong>Read before trading on any of this:</strong> every number here comes from a real backtest where
    parameters were tuned on one slice of history and then checked against a holdout slice the tuning never
    saw. "VALIDATED EDGE" means the strategy beat simple buy-and-hold on that honest holdout — it does not
    mean it will keep working. "NO VALIDATED EDGE" is the common, expected result, not a bug. Nothing here
    is a guarantee, and most active trading loses to buy-and-hold after costs. Size and risk at your own discretion.
  </div>

  <section class="track-record">
    <h2>Live track record (Supabase)</h2>
    <div id="track-record-body" class="loading">Loading signal accuracy history&hellip;</div>
  </section>

  <div class="filter-bar" id="filter-bar">
    <button class="filter-btn active" data-filter="all">All</button>
    <button class="filter-btn" data-filter="strong">🟩 Strong</button>
    <button class="filter-btn" data-filter="moderate">🟨 Moderate</button>
    <button class="filter-btn" data-filter="high_risk">🟧 High risk / high reward</button>
    <button class="filter-btn" data-filter="none">⬜ No signal</button>
  </div>

  {coins_html}

  <script>
    (async function() {{
      const el = document.getElementById("track-record-body");
      try {{
        const resp = await fetch(
          "{SUPABASE_URL}/rest/v1/coin_accuracy?select=*&order=coin.asc",
          {{ headers: {{
              "apikey": "{SUPABASE_PUBLISHABLE_KEY}",
              "Authorization": "Bearer {SUPABASE_PUBLISHABLE_KEY}"
          }} }}
        );
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        const rows = await resp.json();
        if (!rows.length) {{
          el.outerHTML = '<div class="empty">No scored signals yet — check back after a couple of scan cycles.</div>';
          return;
        }}
        const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({{
          "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
        }}[c]));
        const tableRows = rows.map((r) => `
          <tr>
            <td>${{escapeHtml(r.coin)}}</td>
            <td>${{escapeHtml(r.horizon)}}</td>
            <td>${{escapeHtml(r.live_status)}}</td>
            <td>${{r.n_outcomes}}</td>
            <td>${{r.hit_rate_pct != null ? r.hit_rate_pct + "%" : "—"}}</td>
            <td>${{r.avg_pct_change != null ? r.avg_pct_change + "%" : "—"}}</td>
          </tr>`).join("");
        el.outerHTML = `
          <table>
            <thead><tr>
              <th>Coin</th><th>Horizon</th><th>Signal</th><th>N</th><th>Hit rate</th><th>Avg move</th>
            </tr></thead>
            <tbody>${{tableRows}}</tbody>
          </table>`;
      }} catch (e) {{
        el.outerHTML = '<div class="empty">Could not load track record: ' + escapeHtmlFallback(e.message) + '</div>';
      }}
      function escapeHtmlFallback(s) {{
        return String(s).replace(/[&<>"']/g, (c) => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
      }}
    }})();

    (function() {{
      const buttons = document.querySelectorAll("#filter-bar .filter-btn");
      const cards = document.querySelectorAll(".horizon-card");
      const coins = document.querySelectorAll(".coin");

      function applyFilter(tier) {{
        cards.forEach((card) => {{
          card.classList.toggle("hidden", tier !== "all" && card.dataset.tier !== tier);
        }});
        coins.forEach((coin) => {{
          const visible = Array.from(coin.querySelectorAll(".horizon-card")).some(
            (c) => !c.classList.contains("hidden")
          );
          coin.classList.toggle("hidden", !visible);
        }});
      }}

      buttons.forEach((btn) => {{
        btn.addEventListener("click", () => {{
          buttons.forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          applyFilter(btn.dataset.filter);
        }});
      }});
    }})();
  </script>
</body>
</html>"""


def save_dashboard(scan_data: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "index.html"
    html_path.write_text(generate_dashboard_html(scan_data), encoding="utf-8")
    json_path = out_dir / "scan_results.json"
    json_path.write_text(json.dumps(scan_data, indent=2, default=str), encoding="utf-8")
    return html_path
