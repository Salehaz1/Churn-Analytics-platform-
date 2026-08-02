from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from .analytics import format_currency, format_percent


STEP_LABELS = [
    "Dataset Upload",
    "Data Exploration & Cleaning",
    "Executive Dashboard",
    "Machine Learning Analytics",
]


def inject_css(theme: str) -> None:
    dark = theme == "Dark"
    colors = {
        "bg": "#07111f" if dark else "#f5f9ff",
        "surface": "rgba(15, 23, 42, 0.68)" if dark else "rgba(255, 255, 255, 0.76)",
        "surface2": "rgba(30, 41, 59, 0.78)" if dark else "rgba(255, 255, 255, 0.92)",
        "text": "#e5edf8" if dark else "#0f172a",
        "muted": "#9fb0c5" if dark else "#64748b",
        "border": "rgba(148, 163, 184, 0.20)" if dark else "rgba(14, 165, 233, 0.14)",
        "shadow": "0 24px 70px rgba(2, 8, 23, 0.34)" if dark else "0 24px 70px rgba(15, 23, 42, 0.10)",
        "blue": "#38bdf8" if dark else "#0ea5e9",
        "teal": "#2dd4bf" if dark else "#14b8a6",
        "green": "#22c55e",
        "amber": "#f59e0b",
        "red": "#ef4444",
    }
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
  --app-bg: {colors["bg"]};
  --card-bg: {colors["surface"]};
  --card-bg-strong: {colors["surface2"]};
  --text: {colors["text"]};
  --muted: {colors["muted"]};
  --border: {colors["border"]};
  --shadow: {colors["shadow"]};
  --blue: {colors["blue"]};
  --teal: {colors["teal"]};
  --green: {colors["green"]};
  --amber: {colors["amber"]};
  --red: {colors["red"]};
}}

html, body, [data-testid="stAppViewContainer"] {{
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at 12% 0%, rgba(14, 165, 233, .16), transparent 34%),
    radial-gradient(circle at 88% 6%, rgba(20, 184, 166, .16), transparent 30%),
    var(--app-bg);
  color: var(--text);
}}

.block-container {{
  max-width: 1480px;
  padding-top: 1.2rem;
  padding-bottom: 4rem;
}}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(15, 23, 42, .94), rgba(8, 47, 73, .92));
  border-right: 1px solid rgba(125, 211, 252, .18);
  color: #e5edf8;
}}

[data-testid="stSidebar"] * {{
  color: #e5edf8;
}}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {{
  color: #dbeafe;
}}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
  gap: .55rem;
}}

[data-testid="stTabs"] button[role="tab"],
[data-testid="stTabs"] button[role="tab"] p,
[data-testid="stTabs"] button[role="tab"] span {{
  color: var(--text) !important;
  font-weight: 800 !important;
}}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span {{
  color: var(--text) !important;
}}

[data-testid="stTabs"] button[role="tab"][aria-selected="false"],
[data-testid="stTabs"] button[role="tab"][aria-selected="false"] p,
[data-testid="stTabs"] button[role="tab"][aria-selected="false"] span {{
  color: var(--text) !important;
  opacity: .72;
}}

[data-testid="stTabs"] [role="tablist"] {{
  gap: 8px;
  border-bottom: 1px solid var(--border);
}}

h1, h2, h3 {{
  letter-spacing: 0;
  color: var(--text);
}}

.hero {{
  position: relative;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 42px 44px;
  min-height: 290px;
  background:
    linear-gradient(135deg, rgba(14, 165, 233, .20), rgba(20, 184, 166, .16)),
    var(--card-bg);
  box-shadow: var(--shadow);
  backdrop-filter: blur(18px);
  animation: cardIn .52s ease both;
}}

.hero:after {{
  content: "";
  position: absolute;
  inset: auto -80px -120px auto;
  width: 390px;
  height: 390px;
  background: conic-gradient(from 180deg, rgba(14,165,233,.26), rgba(20,184,166,.32), rgba(14,165,233,.12));
  filter: blur(10px);
  border-radius: 999px;
  opacity: .7;
}}

.hero h1 {{
  font-size: clamp(2.2rem, 5vw, 4.8rem);
  line-height: 1.02;
  margin: 0 0 18px;
  font-weight: 800;
  max-width: 960px;
}}

.hero p {{
  color: var(--muted);
  font-size: 1.08rem;
  line-height: 1.8;
  max-width: 920px;
  margin: 0;
}}

.glass-card {{
  border: 1px solid var(--border);
  border-radius: 20px;
  background: var(--card-bg);
  box-shadow: var(--shadow);
  backdrop-filter: blur(18px);
  padding: 20px;
  animation: cardIn .46s ease both;
}}

.metric-card {{
  position: relative;
  overflow: hidden;
  min-height: 150px;
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 18px 18px 16px;
  background:
    linear-gradient(145deg, rgba(255,255,255,.12), rgba(255,255,255,.03)),
    var(--card-bg-strong);
  box-shadow: var(--shadow);
  transition: transform .20s ease, border-color .20s ease, box-shadow .20s ease;
}}

.metric-card:hover {{
  transform: translateY(-3px);
  border-color: rgba(14, 165, 233, .36);
  box-shadow: 0 28px 90px rgba(14, 165, 233, .16);
}}

.metric-label {{
  color: var(--muted);
  font-size: .82rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: 14px;
}}

.metric-value {{
  color: var(--text);
  font-size: clamp(1.55rem, 3vw, 2.35rem);
  font-weight: 800;
  line-height: 1;
  animation: countIn .7s cubic-bezier(.18,.89,.32,1.28) both;
}}

.metric-trend {{
  display: inline-flex;
  gap: 6px;
  align-items: center;
  margin-top: 14px;
  color: var(--muted);
  font-size: .88rem;
}}

.metric-dot {{
  width: 8px;
  height: 8px;
  border-radius: 99px;
  background: var(--teal);
  box-shadow: 0 0 0 5px rgba(20, 184, 166, .12);
}}

.insight-card {{
  position: relative;
  overflow: hidden;
  min-height: 190px;
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 22px;
  background:
    linear-gradient(135deg, rgba(14, 165, 233, .16), rgba(20, 184, 166, .10)),
    var(--card-bg-strong);
  box-shadow: var(--shadow);
  transition: transform .20s ease, border-color .20s ease, box-shadow .20s ease;
}}

.insight-card:hover {{
  transform: translateY(-3px);
  border-color: rgba(20, 184, 166, .38);
  box-shadow: 0 28px 90px rgba(20, 184, 166, .16);
}}

.insight-card:before {{
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 4px;
  background: linear-gradient(90deg, var(--blue), var(--teal));
}}

.insight-kicker {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--teal);
  font-size: .78rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: 14px;
}}

.insight-title {{
  color: var(--text);
  font-size: 1.08rem;
  font-weight: 800;
  margin-bottom: 10px;
}}

.insight-body {{
  color: var(--muted);
  font-size: .96rem;
  line-height: 1.65;
}}

.section-title {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin: 30px 0 12px;
}}

.section-title h2 {{
  margin: 0;
  font-size: 1.45rem;
}}

.section-title p {{
  color: var(--muted);
  margin: 4px 0 0;
}}

.progress-wrap {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 10px 0 22px;
}}

.step {{
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 12px;
  background: var(--card-bg);
  min-height: 78px;
}}

.step .index {{
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 99px;
  margin-right: 8px;
  background: rgba(100, 116, 139, .16);
  color: var(--muted);
  font-weight: 800;
}}

.step.complete {{
  border-color: rgba(34, 197, 94, .40);
  background: linear-gradient(135deg, rgba(34, 197, 94, .15), var(--card-bg));
}}

.step.complete .index {{
  background: rgba(34, 197, 94, .18);
  color: var(--green);
}}

.step.current {{
  border-color: rgba(14, 165, 233, .55);
  box-shadow: 0 18px 40px rgba(14, 165, 233, .15);
}}

.step.locked {{
  opacity: .58;
}}

.step-title {{
  font-weight: 800;
  font-size: .9rem;
}}

.step-status {{
  color: var(--muted);
  font-size: .78rem;
  margin-top: 6px;
}}

.alert {{
  border-radius: 18px;
  padding: 16px 18px;
  border: 1px solid var(--border);
  background: var(--card-bg);
  box-shadow: 0 14px 36px rgba(2, 8, 23, .08);
  margin: 10px 0;
}}

.alert.success {{ border-color: rgba(34, 197, 94, .38); background: linear-gradient(135deg, rgba(34,197,94,.16), var(--card-bg)); }}
.alert.warning {{ border-color: rgba(245, 158, 11, .42); background: linear-gradient(135deg, rgba(245,158,11,.16), var(--card-bg)); }}
.alert.error {{ border-color: rgba(239, 68, 68, .42); background: linear-gradient(135deg, rgba(239,68,68,.16), var(--card-bg)); }}
.alert strong {{ display: block; margin-bottom: 4px; }}
.alert span {{ color: var(--muted); }}

.quality-card {{
  border-radius: 18px;
  padding: 18px;
  border: 1px solid rgba(245, 158, 11, .30);
  background: linear-gradient(135deg, rgba(245, 158, 11, .14), var(--card-bg));
  min-height: 210px;
}}

.severity {{
  display: inline-flex;
  align-items: center;
  border-radius: 99px;
  padding: 5px 10px;
  font-size: .78rem;
  font-weight: 800;
  background: rgba(245, 158, 11, .18);
  color: var(--amber);
}}

.chart-card {{
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 18px;
  background: var(--card-bg-strong);
  box-shadow: var(--shadow);
  margin-bottom: 18px;
}}

.chart-card h3 {{
  margin: 0 0 4px;
  font-size: 1.05rem;
}}

.chart-card p {{
  margin: 0 0 12px;
  color: var(--muted);
  font-size: .92rem;
  line-height: 1.55;
}}

.sidebar-logo {{
  border-radius: 20px;
  padding: 16px;
  background: linear-gradient(135deg, rgba(14,165,233,.26), rgba(20,184,166,.24));
  border: 1px solid rgba(125, 211, 252, .22);
  margin-bottom: 10px;
}}

.sidebar-logo h2 {{
  color: #f8fafc;
  margin: 0;
  font-size: 1.02rem;
}}

.sidebar-logo p {{
  color: #bae6fd;
  margin: 5px 0 0;
  font-size: .78rem;
}}

.footer {{
  color: var(--muted);
  text-align: center;
  padding: 22px;
  margin-top: 24px;
}}

.upload-zone {{
  border: 1.5px dashed rgba(14, 165, 233, .42);
  border-radius: 20px;
  padding: 20px;
  background: linear-gradient(135deg, rgba(14,165,233,.10), rgba(20,184,166,.08));
}}

.stButton > button,
[data-testid="stDownloadButton"] > button {{
  border-radius: 14px;
  border: 1px solid rgba(14, 165, 233, .26);
  background: linear-gradient(135deg, var(--blue), var(--teal));
  color: white;
  font-weight: 800;
  box-shadow: 0 14px 32px rgba(14, 165, 233, .20);
  transition: transform .18s ease, box-shadow .18s ease;
}}

.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover {{
  transform: translateY(-2px);
  box-shadow: 0 18px 42px rgba(14, 165, 233, .28);
}}

[data-testid="stFileUploader"] {{
  border-radius: 20px;
}}

.skeleton {{
  height: 120px;
  border-radius: 20px;
  background: linear-gradient(90deg, rgba(148,163,184,.12), rgba(148,163,184,.26), rgba(148,163,184,.12));
  background-size: 240% 100%;
  animation: shimmer 1.45s ease-in-out infinite;
}}

@keyframes shimmer {{
  0% {{ background-position: 100% 0; }}
  100% {{ background-position: -100% 0; }}
}}

@keyframes countIn {{
  from {{ transform: translateY(8px) scale(.96); opacity: 0; }}
  to {{ transform: translateY(0) scale(1); opacity: 1; }}
}}

@keyframes cardIn {{
  from {{ transform: translateY(10px); opacity: 0; }}
  to {{ transform: translateY(0); opacity: 1; }}
}}

@media (max-width: 860px) {{
  .progress-wrap {{ grid-template-columns: 1fr; }}
  .hero {{ padding: 28px 24px; }}
  .metric-card {{ min-height: 130px; }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
<div class="hero">
  <h1>Customer Churn Analytics Platform</h1>
  <p>Analyze customer churn, identify business risks, predict customer behavior, discover hidden customer segments, and generate AI-powered business recommendations.</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_progress(current_step: int, max_completed: int) -> None:
    html_parts = ['<div class="progress-wrap">']
    for idx, label in enumerate(STEP_LABELS, start=1):
        if idx <= max_completed:
            state = "complete"
            status = "Completed"
            index = "✓"
        elif idx == current_step:
            state = "current"
            status = "In progress"
            index = str(idx)
        else:
            state = "locked"
            status = "Locked"
            index = "⌁"
        html_parts.append(
            f"""
<div class="step {state}">
  <div><span class="index">{index}</span><span class="step-title">{html.escape(label)}</span></div>
  <div class="step-status">{status}</div>
</div>
            """
        )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def section(title: str, subtitle: str | None = None) -> None:
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
<div class="section-title">
  <div><h2>{html.escape(title)}</h2>{sub}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def alert(kind: str, title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="alert {html.escape(kind)}">
  <strong>{html.escape(title)}</strong>
  <span>{html.escape(body)}</span>
</div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, definition: str, trend: str = "Live metric", status_color: str = "teal") -> None:
    color_var = {"green": "var(--green)", "amber": "var(--amber)", "red": "var(--red)", "teal": "var(--teal)", "blue": "var(--blue)"}.get(status_color, "var(--teal)")
    st.markdown(
        f"""
<div class="metric-card" title="{html.escape(definition)}">
  <div class="metric-label">{html.escape(label)}</div>
  <div class="metric-value">{html.escape(value)}</div>
  <div class="metric-trend"><span class="metric-dot" style="background:{color_var}; box-shadow:0 0 0 5px color-mix(in srgb, {color_var} 18%, transparent);"></span>{html.escape(trend)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(index: int, insight: str) -> None:
    st.markdown(
        f"""
<div class="insight-card" title="{html.escape(insight)}">
  <div class="insight-kicker">AI Insight {index}</div>
  <div class="insight-title">Executive Recommendation</div>
  <div class="insight-body">{html.escape(insight)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def quality_issue_card(issue: dict[str, str]) -> None:
    st.markdown(
        f"""
<div class="quality-card">
  <span class="severity">{html.escape(issue["Severity"])}</span>
  <h3>{html.escape(issue["Issue"])}</h3>
  <p><strong>Explanation:</strong> {html.escape(issue["Explanation"])}</p>
  <p><strong>Business Impact:</strong> {html.escape(issue["Business Impact"])}</p>
  <p><strong>Recommended Solution:</strong> {html.escape(issue["Recommended Solution"])}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def chart_card(title: str, description: str, fig: Any, key: str) -> None:
    st.markdown(
        f"""
<div class="chart-card" title="{html.escape(description)}">
  <h3>{html.escape(title)}</h3>
  <p>{html.escape(description)}</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
            "toImageButtonOptions": {"format": "png", "filename": key, "scale": 2},
        },
    )
    try:
        png = fig.to_image(format="png", scale=2)
        st.download_button("Download PNG", data=png, file_name=f"{key}.png", mime="image/png", key=f"png_{key}")
    except Exception:
        st.download_button(
            "Download interactive HTML",
            data=fig.to_html(full_html=True, include_plotlyjs="cdn"),
            file_name=f"{key}.html",
            mime="text/html",
            key=f"html_{key}",
        )


def summary_table(title: str, data: dict[str, Any]) -> None:
    st.markdown(f'<div class="glass-card"><h3>{html.escape(title)}</h3></div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(list(data.items()), columns=["Metric", "Value"]), use_container_width=True, hide_index=True)


def format_metric_value(name: str, value: Any, currency: str, exchange_rate: float) -> str:
    money_terms = ["Revenue", "Loss", "Charges", "Value", "CLV", "Spending"]
    if any(term in name for term in money_terms):
        return format_currency(float(value), currency, exchange_rate)
    if "Rate" in name or "Accuracy" in name or "F1" in name:
        return format_percent(float(value))
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}" if isinstance(value, int) else str(value)
