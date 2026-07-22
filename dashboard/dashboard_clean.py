import streamlit as st
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropGuard — Disease Early Warning",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1a2e 100%);
    color: #e8f4e8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071220 0%, #0d1f33 100%);
    border-right: 1px solid rgba(74, 222, 128, 0.15);
}
[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #94d4a4 !important;
}

/* Title */
.dash-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4ade80, #22d3ee, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
    margin-bottom: 0;
    line-height: 1.1;
}
.dash-subtitle {
    font-size: 0.9rem;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(74, 222, 128, 0.18);
    border-radius: 14px;
    padding: 20px 22px 16px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(74,222,128,0.6), transparent);
}
.kpi-card:hover { border-color: rgba(74,222,128,0.4); }
.kpi-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Space Mono', monospace;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    color: #e2fde8;
    line-height: 1;
}
.kpi-delta {
    font-size: 0.78rem;
    margin-top: 6px;
    font-family: 'DM Sans', sans-serif;
}
.kpi-delta.up   { color: #4ade80; }
.kpi-delta.down { color: #f87171; }
.kpi-delta.warn { color: #fbbf24; }

/* Section Headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4ade80;
    border-left: 3px solid #4ade80;
    padding-left: 10px;
    margin: 24px 0 14px;
}

/* Alert Badge */
.badge-high   { background: rgba(248,113,113,0.15); color:#f87171; border:1px solid rgba(248,113,113,0.3); padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-medium { background: rgba(251,191,36,0.15); color:#fbbf24; border:1px solid rgba(251,191,36,0.3);  padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-low    { background: rgba(74,222,128,0.15);  color:#4ade80; border:1px solid rgba(74,222,128,0.3);  padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

/* Chart containers */
.chart-box {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 6px;
}

/* Insight card */
.insight-card {
    background: linear-gradient(135deg, rgba(74,222,128,0.06), rgba(34,211,238,0.04));
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #c8e6d4;
}
.insight-icon { font-size: 1.1rem; margin-right: 8px; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Selectbox / multiselect labels */
.stSelectbox label, .stMultiSelect label, .stDateInput label { color: #94d4a4 !important; }

/* Divider */
hr { border-color: rgba(74,222,128,0.12) !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(0,0,0,0.2);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}
.stTabs [aria-selected="data/true"] {
    background: rgba(74,222,128,0.15) !important;
    color: #4ade80 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ──────────────────────────────────────────────────────────
DATA_FILE = Path(__file__).resolve().parent / "dashboard_data.csv"

if not DATA_FILE.exists():
    st.error(f"Could not find dashboard CSV: {DATA_FILE}")
    st.stop()

@st.cache_data
def load_dashboard_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df["Week"]  = df["Date"].dt.to_period("W").dt.to_timestamp()
    return df

df_all = load_dashboard_data(DATA_FILE)

# ─── Plotly theme helper ──────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94a3b8", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    margin=dict(l=10, r=10, t=30, b=10),
)
GRID_STYLE = dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)")
GREEN_PALETTE = ["#4ade80", "#22d3ee", "#a78bfa", "#fbbf24", "#f87171", "#34d399", "#fb923c", "#e879f9"]


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 CropGuard")
    st.markdown("<div style='font-size:0.75rem;color:#475569;font-family:Space Mono;margin-bottom:20px'>Early Warning System v2.0</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Filters**")

    all_crops   = sorted(df_all["Crop_Type"].unique())
    all_regions = sorted(df_all["Region"].unique())
    all_risks   = ["High", "Medium", "Low"]
    all_diseases= sorted(df_all["Predicted_Disease"].unique())

    sel_crops   = st.multiselect("Crop Type",   all_crops,   default=all_crops,   key="crops")
    sel_regions = st.multiselect("Region",      all_regions, default=all_regions, key="regions")
    sel_risks   = st.multiselect("Risk Level",  all_risks,   default=all_risks,   key="risks")
    sel_disease = st.multiselect("Disease",     all_diseases,default=all_diseases,key="diseases")

    min_date = df_all["Date"].min().date()
    max_date = df_all["Date"].max().date()
    date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    st.markdown("---")
    st.markdown("**Model**")
    show_confidence = st.slider("Min. Disease Probability (%)", 0, 100, 0)

    st.markdown("---")
    st.caption("📊 Data loaded from dashboard_data.csv — update this file to change the dashboard source.")

# ─── Apply Filters ───────────────────────────────────────────────────────────
df = df_all.copy()
if sel_crops:   df = df[df["Crop_Type"].isin(sel_crops)]
if sel_regions: df = df[df["Region"].isin(sel_regions)]
if sel_risks:   df = df[df["Alert_Status"].isin(sel_risks)]
if sel_disease: df = df[df["Predicted_Disease"].isin(sel_disease)]
if len(date_range) == 2:
    df = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])]
df = df[df["Disease_Probability"] >= show_confidence]


# ─── Header ──────────────────────────────────────────────────────────────────
col_title, col_ts = st.columns([3, 1])
with col_title:
    st.markdown('<div class="dash-title">🌾 CropGuard Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-subtitle">AI-Based Crop Disease Early Warning System</div>', unsafe_allow_html=True)
with col_ts:
    st.markdown(f"<div style='text-align:right;color:#475569;font-size:0.78rem;font-family:Space Mono;padding-top:10px'>{datetime.now().strftime('%d %b %Y  %H:%M')}<br><span style='color:#4ade80'>● LIVE</span></div>", unsafe_allow_html=True)

st.markdown("<div style='margin:8px 0 4px'></div>", unsafe_allow_html=True)

# ─── KPI Row ─────────────────────────────────────────────────────────────────
total     = len(df)
correct   = df["Correct_Prediction"].sum()
success   = round(correct / total * 100, 1) if total else 0
failure   = round(100 - success, 1)
high_risk = (df["Alert_Status"] == "High").sum()
avg_yield_loss = round(df["Yield_Loss_Pct"].mean(), 1)
avg_proc_time  = round(df["Processing_Time_ms"].mean(), 0)

kpi_cols = st.columns(6)
kpis = [
    ("Total Analyses", f"{total:,}", "↑ +12% vs last month", "up"),
    ("Success Rate",   f"{success}%",  "↑ Model accuracy", "up"),
    ("Failure Rate",   f"{failure}%",  "↓ Keep monitoring", "down"),
    ("Avg Proc. Time", f"{int(avg_proc_time)}ms", "⟳ Real-time inference", "warn"),
    ("High Risk Alerts", f"{high_risk}", "⚠ Requires action", "down"),
    ("Avg Yield Loss", f"{avg_yield_loss}%", "↓ Financial impact", "down"),
]
for col, (label, value, delta, cls) in zip(kpi_cols, kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta {cls}">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Disease Analysis", "🌦 Weather Trends", "🗺 Regional Map", "📋 Data & Insights"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Disease Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns(2)

    # Bar: Disease count by crop
    with col_a:
        st.markdown('<div class="section-header">Disease Distribution by Crop</div>', unsafe_allow_html=True)
        bar_data = df.groupby(["Crop_Type", "Predicted_Disease"]).size().reset_index(name="Count")
        fig_bar = px.bar(
            bar_data, x="Crop_Type", y="Count", color="Predicted_Disease",
            color_discrete_sequence=GREEN_PALETTE,
            template="plotly_dark",
        )
        fig_bar.update_layout(**PLOT_LAYOUT, barmode="stack", height=320)
        fig_bar.update_xaxes(**GRID_STYLE, title="")
        fig_bar.update_yaxes(**GRID_STYLE, title="Cases")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Pie: Risk distribution
    with col_b:
        st.markdown('<div class="section-header">Alert Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = df["Alert_Status"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        fig_pie = px.pie(
            risk_counts, values="Count", names="Risk",
            color="Risk",
            color_discrete_map={"High": "#f87171", "Medium": "#fbbf24", "Low": "#4ade80"},
            hole=0.55,
        )
        fig_pie.update_layout(**PLOT_LAYOUT, height=320)
        fig_pie.update_traces(textfont_size=13, marker=dict(line=dict(color="rgba(0,0,0,0.4)", width=2)))
        st.plotly_chart(fig_pie, use_container_width=True)

    col_c, col_d = st.columns(2)

    # Line: Disease trend over time
    with col_c:
        st.markdown('<div class="section-header">Disease Occurrence Trend (Monthly)</div>', unsafe_allow_html=True)
        trend = df.groupby(["Month", "Alert_Status"]).size().reset_index(name="Count")
        fig_line = px.line(
            trend, x="Month", y="Count", color="Alert_Status",
            color_discrete_map={"High": "#f87171", "Medium": "#fbbf24", "Low": "#4ade80"},
            markers=True,
        )
        fig_line.update_layout(**PLOT_LAYOUT, height=300)
        fig_line.update_xaxes(**GRID_STYLE, title="")
        fig_line.update_yaxes(**GRID_STYLE, title="Cases")
        st.plotly_chart(fig_line, use_container_width=True)

    # Bar: Avg yield loss per disease
    with col_d:
        st.markdown('<div class="section-header">Avg Yield Loss by Disease (%)</div>', unsafe_allow_html=True)
        yield_data = df.groupby("Predicted_Disease")["Yield_Loss_Pct"].mean().sort_values(ascending=True).reset_index()
        fig_yld = px.bar(
            yield_data, x="Yield_Loss_Pct", y="Predicted_Disease",
            orientation="h", color="Yield_Loss_Pct",
            color_continuous_scale=["#4ade80", "#fbbf24", "#f87171"],
        )
        fig_yld.update_layout(**PLOT_LAYOUT, height=300, coloraxis_showscale=False)
        fig_yld.update_xaxes(**GRID_STYLE, title="Avg Yield Loss (%)")
        fig_yld.update_yaxes(**GRID_STYLE, title="")
        st.plotly_chart(fig_yld, use_container_width=True)

    # Scatter: Temperature vs Humidity coloured by risk
    st.markdown('<div class="section-header">Temperature vs Humidity — Disease Risk</div>', unsafe_allow_html=True)
    fig_sc = px.scatter(
        df.sample(min(300, len(df))), x="Temperature_C", y="Humidity_Pct",
        color="Alert_Status", size="Disease_Probability",
        color_discrete_map={"High": "#f87171", "Medium": "#fbbf24", "Low": "#4ade80"},
        hover_data=["Crop_Type", "Predicted_Disease", "Region"],
        opacity=0.75,
    )
    fig_sc.update_layout(**PLOT_LAYOUT, height=330)
    fig_sc.update_xaxes(**GRID_STYLE, title="Temperature (°C)")
    fig_sc.update_yaxes(**GRID_STYLE, title="Humidity (%)")
    st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Weather Trends
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.markdown('<div class="section-header">Avg Temperature Over Time</div>', unsafe_allow_html=True)
        temp_trend = df.groupby("Month")["Temperature_C"].mean().reset_index()
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=temp_trend["Month"], y=temp_trend["Temperature_C"],
            mode="lines+markers", name="Avg Temp",
            line=dict(color="#f87171", width=2.5),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.1)",
        ))
        fig_temp.update_layout(**PLOT_LAYOUT, height=280)
        fig_temp.update_xaxes(**GRID_STYLE)
        fig_temp.update_yaxes(**GRID_STYLE, title="°C")
        st.plotly_chart(fig_temp, use_container_width=True)

    with col_w2:
        st.markdown('<div class="section-header">Avg Humidity Over Time</div>', unsafe_allow_html=True)
        hum_trend = df.groupby("Month")["Humidity_Pct"].mean().reset_index()
        fig_hum = go.Figure()
        fig_hum.add_trace(go.Scatter(
            x=hum_trend["Month"], y=hum_trend["Humidity_Pct"],
            mode="lines+markers", name="Avg Humidity",
            line=dict(color="#22d3ee", width=2.5),
            fill="tozeroy", fillcolor="rgba(34,211,238,0.1)",
        ))
        fig_hum.update_layout(**PLOT_LAYOUT, height=280)
        fig_hum.update_xaxes(**GRID_STYLE)
        fig_hum.update_yaxes(**GRID_STYLE, title="%")
        st.plotly_chart(fig_hum, use_container_width=True)

    col_w3, col_w4 = st.columns(2)

    with col_w3:
        st.markdown('<div class="section-header">Monthly Rainfall (mm)</div>', unsafe_allow_html=True)
        rain_trend = df.groupby("Month")["Rainfall_mm"].sum().reset_index()
        fig_rain = px.bar(rain_trend, x="Month", y="Rainfall_mm",
                          color_discrete_sequence=["#60a5fa"])
        fig_rain.update_layout(**PLOT_LAYOUT, height=280)
        fig_rain.update_xaxes(**GRID_STYLE, title="")
        fig_rain.update_yaxes(**GRID_STYLE, title="mm")
        st.plotly_chart(fig_rain, use_container_width=True)

    with col_w4:
        st.markdown('<div class="section-header">VPD Distribution by Risk Level</div>', unsafe_allow_html=True)
        fig_vpd = px.box(
            df, x="Alert_Status", y="VPD", color="Alert_Status",
            color_discrete_map={"High": "#f87171", "Medium": "#fbbf24", "Low": "#4ade80"},
        )
        fig_vpd.update_layout(**PLOT_LAYOUT, height=280, showlegend=False)
        fig_vpd.update_xaxes(**GRID_STYLE, title="Risk Level")
        fig_vpd.update_yaxes(**GRID_STYLE, title="VPD (kPa)")
        st.plotly_chart(fig_vpd, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-header">Weather Parameter Correlation</div>', unsafe_allow_html=True)
    corr_cols = ["Temperature_C", "Humidity_Pct", "Rainfall_mm", "VPD", "Wet_Leaf_Hours", "Disease_Probability", "Yield_Loss_Pct"]
    corr = df[corr_cols].corr().round(2)
    fig_hm = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale=["#1e3a5f", "#0d2137", "#4ade80"],
        zmin=-1, zmax=1,
    )
    fig_hm.update_layout(**PLOT_LAYOUT, height=350)
    st.plotly_chart(fig_hm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Regional Map
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    REGION_COORDS = {
        "Punjab":          (30.90, 75.85),
        "Maharashtra":     (19.75, 75.71),
        "Karnataka":       (15.32, 75.71),
        "Uttar Pradesh":   (26.85, 80.91),
        "Tamil Nadu":      (11.13, 78.66),
        "Rajasthan":       (27.02, 74.22),
        "Andhra Pradesh":  (15.91, 79.74),
        "West Bengal":     (22.99, 87.85),
    }

    region_stats = df.groupby("Region").agg(
        Total=("Crop_ID", "count"),
        High_Risk=("Alert_Status", lambda x: (x == "High").sum()),
        Avg_Yield_Loss=("Yield_Loss_Pct", "mean"),
        Avg_Disease_Prob=("Disease_Probability", "mean"),
    ).reset_index()

    region_stats["Lat"] = region_stats["Region"].map(lambda r: REGION_COORDS.get(r, (20, 78))[0])
    region_stats["Lon"] = region_stats["Region"].map(lambda r: REGION_COORDS.get(r, (20, 78))[1])
    region_stats["Avg_Yield_Loss"] = region_stats["Avg_Yield_Loss"].round(1)
    region_stats["Avg_Disease_Prob"] = region_stats["Avg_Disease_Prob"].round(1)

    col_map, col_reg = st.columns([2, 1])

    with col_map:
        st.markdown('<div class="section-header">Region-wise Disease Risk Map</div>', unsafe_allow_html=True)
        fig_map = px.scatter_mapbox(
            region_stats, lat="Lat", lon="Lon", size="High_Risk",
            color="Avg_Yield_Loss",
            color_continuous_scale=["#4ade80", "#fbbf24", "#f87171"],
            hover_name="Region",
            hover_data={"Total": True, "High_Risk": True, "Avg_Yield_Loss": True, "Lat": False, "Lon": False},
            zoom=4, height=440,
            size_max=40,
            mapbox_style="carto-darkmatter",
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), **{k: v for k, v in PLOT_LAYOUT.items() if k != 'margin'})
        st.plotly_chart(fig_map, use_container_width=True)

    with col_reg:
        st.markdown('<div class="section-header">Region Stats</div>', unsafe_allow_html=True)
        for _, row in region_stats.sort_values("High_Risk", ascending=False).iterrows():
            risk_pct = round(row["High_Risk"] / row["Total"] * 100, 0) if row["Total"] else 0
            bar_w = int(risk_pct)
            st.markdown(f"""
            <div style='margin-bottom:10px;padding:10px 14px;background:rgba(255,255,255,0.03);border-radius:10px;border:1px solid rgba(255,255,255,0.07)'>
                <div style='font-size:0.82rem;font-weight:600;color:#c8e6d4'>{row['Region']}</div>
                <div style='font-size:0.72rem;color:#64748b;margin:3px 0'>
                    {int(row['Total'])} analyses &bull; {int(row['High_Risk'])} high-risk &bull; {row['Avg_Yield_Loss']}% yield loss
                </div>
                <div style='background:rgba(255,255,255,0.07);border-radius:4px;height:5px;margin-top:6px'>
                    <div style='background:{"#f87171" if risk_pct>40 else "#fbbf24" if risk_pct>20 else "#4ade80"};width:{bar_w}%;height:5px;border-radius:4px'></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Bar: region comparison
    st.markdown('<div class="section-header">Region-wise Case Comparison</div>', unsafe_allow_html=True)
    fig_reg = px.bar(
        region_stats.sort_values("Total", ascending=False),
        x="Region", y=["Total", "High_Risk"],
        barmode="group",
        color_discrete_sequence=["#22d3ee", "#f87171"],
    )
    fig_reg.update_layout(**PLOT_LAYOUT, height=300)
    fig_reg.update_xaxes(**GRID_STYLE)
    fig_reg.update_yaxes(**GRID_STYLE)
    st.plotly_chart(fig_reg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Data Table & Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)

    # Compute insights dynamically
    top_disease  = df["Predicted_Disease"].value_counts().index[0] if len(df) else "N/A"
    top_region   = df["Region"].value_counts().index[0] if len(df) else "N/A"
    top_crop     = df["Crop_Type"].value_counts().index[0] if len(df) else "N/A"
    highest_loss = df.groupby("Region")["Yield_Loss_Pct"].mean().idxmax() if len(df) else "N/A"
    highest_loss_val = round(df.groupby("Region")["Yield_Loss_Pct"].mean().max(), 1) if len(df) else 0
    high_risk_pct = round((df["Alert_Status"] == "High").mean() * 100, 1) if len(df) else 0

    insights = [
        (f"🦠 Most frequent disease: **{top_disease}** — Monitor closely with targeted fungicide applications."),
        (f"📍 Highest-risk region: **{top_region}** — Deploy field scouts and activate spray advisories."),
        (f"🌾 Most affected crop: **{top_crop}** — Review planting schedules and resistant variety options."),
        (f"📉 **{highest_loss}** has the highest avg yield loss at **{highest_loss_val}%** — prioritize intervention here."),
        (f"⚠ **{high_risk_pct}%** of analyses flagged as High Risk — early intervention can reduce yield losses by up to 40%."),
        ("🌡 High humidity (>75%) combined with temperatures 25–35°C strongly correlates with disease outbreak onset."),
        (f"🎯 Model accuracy is stable at **{round(df['Model_Accuracy_Pct'].mean(),1)}%** — fusion model performing well."),
    ]
    for ins in insights:
        st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Detailed Crop Data Table</div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search_term = st.text_input("data/🔍 Search Crop ID or Disease", "")
    with col_s2:
        n_rows = st.selectbox("Rows to display", [20, 50, 100, 200], index=0)

    display_cols = ["Crop_ID", "Date", "Crop_Type", "Region", "Predicted_Disease",
                    "Disease_Probability", "Alert_Status", "Yield_Loss_Pct",
                    "Temperature_C", "Humidity_Pct", "Rainfall_mm", "Correct_Prediction"]

    df_display = df[display_cols].copy()
    if search_term:
        mask = (
            df_display["Crop_ID"].str.contains(search_term, case=False, na=False) |
            df_display["Predicted_Disease"].str.contains(search_term, case=False, na=False)
        )
        df_display = df_display[mask]

    df_display = df_display.sort_values("Date", ascending=False).head(n_rows)
    df_display["Date"] = df_display["Date"].dt.strftime("%d %b %Y")
    df_display["Disease_Probability"] = df_display["Disease_Probability"].apply(lambda x: f"{x}%")
    df_display["Yield_Loss_Pct"]      = df_display["Yield_Loss_Pct"].apply(lambda x: f"{x}%")
    df_display["Correct_Prediction"]  = df_display["Correct_Prediction"].map({True: "✅", False: "❌"})

    st.dataframe(
        df_display.rename(columns={
            "Crop_ID": "ID", "Crop_Type": "Crop", "Predicted_Disease": "Disease",
            "Disease_Probability": "Prob.", "Alert_Status": "Risk",
            "Yield_Loss_Pct": "Yield Loss", "Temperature_C": "Temp (°C)",
            "Humidity_Pct": "Humidity", "Rainfall_mm": "Rainfall",
            "Correct_Prediction": "Correct",
        }),
        use_container_width=True, height=460,
    )

    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_data = df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Filtered Data (CSV)", csv_data,
                           file_name="cropguard_data.csv", mime="data/text/csv")
    with col_dl2:
        summary = df.describe().round(2)
        sum_csv = summary.to_csv().encode("utf-8")
        st.download_button("⬇ Download Summary Statistics (CSV)", sum_csv,
                           file_name="cropguard_summary.csv", mime="text/csv")

    # Model Performance section
    st.markdown("---")
    st.markdown('<div class="section-header">Model Performance Over Time</div>', unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        perf_trend = df.groupby("Month").agg(
            Accuracy=("Model_Accuracy_Pct", "mean"),
            Proc_Time=("Processing_Time_ms", "mean"),
        ).reset_index()
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=perf_trend["Month"], y=perf_trend["Accuracy"],
            name="Accuracy (%)", line=dict(color="#4ade80", width=2.5), mode="lines+markers",
        ))
        fig_acc.update_layout(**PLOT_LAYOUT, height=260, title="Model Accuracy")
        fig_acc.update_xaxes(**GRID_STYLE)
        fig_acc.update_yaxes(**GRID_STYLE, range=[85, 100])
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_p2:
        fig_proc = go.Figure()
        fig_proc.add_trace(go.Scatter(
            x=perf_trend["Month"], y=perf_trend["Proc_Time"],
            name="Proc. Time (ms)", line=dict(color="#a78bfa", width=2.5), mode="lines+markers",
            fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
        ))
        fig_proc.update_layout(**PLOT_LAYOUT, height=260, title="Avg Processing Time")
        fig_proc.update_xaxes(**GRID_STYLE)
        fig_proc.update_yaxes(**GRID_STYLE, title="ms")
        st.plotly_chart(fig_proc, use_container_width=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#334155;font-family:Space Mono;font-size:0.72rem;padding:8px'>"
    "CropGuard v2.0 &nbsp;·&nbsp; AI-Based Crop Disease Early Warning System &nbsp;·&nbsp; "
    "EfficientNet-B4 + LSTM Fusion &nbsp;·&nbsp; Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)