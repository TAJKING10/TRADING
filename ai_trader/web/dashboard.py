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


def _horizon_card(base: str, horizon_name: str, data: dict) -> str:
    status = data.get("status")
    if status != "ok":
        reason = {"insufficient_data": "Not enough history yet", "no_signal": "No tradable setups in test data",
                   "error": "Scan error"}.get(status, status)
        return f"""
        <div class="horizon-card muted">
          <div class="horizon-title">{horizon_name.upper()}</div>
          <div class="na">— {html.escape(str(reason))} —</div>
        </div>"""

    m = data["holdout_metrics"]
    validated = data["edge_validated"]
    badge_class = "validated" if validated else "unvalidated"
    badge_text = "VALIDATED EDGE" if validated else "NO VALIDATED EDGE"
    live_label, live_color = STATUS_LABELS.get(data["live_status"], (data["live_status"], "#6e7781"))
    price = data["live_price"]
    decimals = 2 if price >= 100 else 4 if price >= 1 else 6
    price_str = f"${price:,.{decimals}f}"

    return f"""
    <div class="horizon-card">
      <div class="horizon-title">{horizon_name.upper()} <span class="badge {badge_class}">{badge_text}</span></div>
      <div class="live-status" style="color:{live_color}">{live_label}</div>
      <div class="price">{price_str}</div>
      <table class="metrics">
        <tr><td>Strategy (holdout)</td><td class="{'pos' if m['total_return_pct']>=0 else 'neg'}">{m['total_return_pct']:+.2f}%</td></tr>
        <tr><td>Buy &amp; hold (same window)</td><td class="{'pos' if data['buy_and_hold_pct']>=0 else 'neg'}">{data['buy_and_hold_pct']:+.2f}%</td></tr>
        <tr><td>Sharpe</td><td>{m['sharpe_annualized']:.2f}</td></tr>
        <tr><td>Max drawdown</td><td>{m['max_drawdown_pct']:.2f}%</td></tr>
        <tr><td>Trades (holdout)</td><td>{m['num_trades']}</td></tr>
        <tr><td>Win rate</td><td>{m['win_rate_pct']:.1f}%</td></tr>
      </table>
    </div>"""


def _coin_section(base: str, horizons: dict) -> str:
    cards = "".join(_horizon_card(base, name, data) for name, data in horizons.items())
    return f"""
    <section class="coin">
      <h2>{html.escape(base)}</h2>
      <div class="horizon-row">{cards}</div>
    </section>"""


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
  {coins_html}
</body>
</html>"""


def save_dashboard(scan_data: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "index.html"
    html_path.write_text(generate_dashboard_html(scan_data), encoding="utf-8")
    json_path = out_dir / "scan_results.json"
    json_path.write_text(json.dumps(scan_data, indent=2, default=str), encoding="utf-8")
    return html_path
