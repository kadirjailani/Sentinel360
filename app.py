"""
app.py

Sentinel360 Healthcare — Executive Early-Warning Dashboard
Reads pre-computed CSV outputs from outputs/.
Does not recalculate analytical results.

Run from the project root:
    streamlit run app.py
"""

import base64
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import altair as alt

# streamlit run prepends this dir to sys.path, but other runners
# (AppTest, pytest) do not — make the analytics import work everywhere
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from analytics.simulate_scenario_v1 import simulate_scenario

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sentinel360 Healthcare",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent.resolve()
ASSETS_DIR = BASE_DIR / "assets"

OUTPUT_FILES = {
    "warning":     BASE_DIR / "outputs" / "kpi_warning_output_v1.csv",
    "cross":       BASE_DIR / "outputs" / "cross_domain_signal_output_v1.csv",
    "forecast":    BASE_DIR / "outputs" / "kpi_forecast_output_v1.csv",
    "ranking":     BASE_DIR / "outputs" / "executive_risk_ranking_v1.csv",
}

# Live inputs for the Scenario Lab — simulation recomputes from these via
# the analytics engine; it never reads the pre-computed output CSVs.
INPUT_FILES = {
    "kpi_data":    BASE_DIR / "data" / "processed" / "hospital_kpi_data_v2.csv",
    "kpi_config":  BASE_DIR / "config" / "kpi_config_v2.csv",
    "scenarios":   BASE_DIR / "config" / "scenario_baselines_v1.csv",
}

DATA_DIR = BASE_DIR / "data" / "processed"
GROUND_TRUTH_PATH = BASE_DIR / "validation" / "ground_truth_v2.xlsx"

# Full expected inventory for the Data & Validation checklist
EXPECTED_PROJECT_FILES = {
    "Data (processed)": [DATA_DIR / f for f in [
        "hospital_kpi_data_v2.csv", "bed_occupancy_records_v1.csv",
        "patient_complaints_v1.csv", "patient_encounters_v1.csv",
        "patient_queue_records_v1.csv", "patient_survey_v1.csv",
        "staff_attendance_v1.csv", "staff_master_v1.csv", "staff_roster_v1.csv",
    ]],
    "Config": [BASE_DIR / "config" / "kpi_config_v2.csv",
               BASE_DIR / "config" / "scenario_baselines_v1.csv"],
    "Outputs": list(OUTPUT_FILES.values()),
    "Validation": [GROUND_TRUTH_PATH],
}

# Duplicate-check keys; files not listed use their first (ID) column
DUPLICATE_KEYS = {
    "hospital_kpi_data_v2.csv": ["reporting_month", "kpi_code"],
    "kpi_warning_output_v1.csv": ["reporting_month", "kpi_code"],
    "cross_domain_signal_output_v1.csv": ["reporting_month"],
    "kpi_forecast_output_v1.csv": ["forecast_month", "kpi_code"],
    "executive_risk_ranking_v1.csv": ["kpi_code"],
    "kpi_config_v2.csv": ["kpi_code"],
    "scenario_baselines_v1.csv": ["scenario_code", "parameter_code"],
}

# Monthly files and their month column, for coverage reporting
MONTH_COLUMNS = {
    "hospital_kpi_data_v2.csv": "reporting_month",
    "kpi_warning_output_v1.csv": "reporting_month",
    "cross_domain_signal_output_v1.csv": "reporting_month",
    "kpi_forecast_output_v1.csv": "forecast_month",
    "executive_risk_ranking_v1.csv": "ranking_as_of_month",
}


# ──────────────────────────────────────────────────────────────────────────────
# THEME  (light-blue squircle look; config.toml sets colours + base radius)
# ──────────────────────────────────────────────────────────────────────────────

# Dark headline-chart background so the main graph stands out against the page.
CHART_BG = "#0E1526"


@st.cache_data(show_spinner=False)
def img_data_uri(name: str) -> str:
    """Base64 data URI for a small image in assets/ (for CSS backgrounds)."""
    data = (ASSETS_DIR / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def render_grid(df: pd.DataFrame) -> None:
    """Render a DataFrame as a borderless zebra table inside a squircle card."""
    html_table = df.fillna("").to_html(index=False, border=0,
                                       classes="rank-table", escape=False)
    st.markdown(f"<div class='rank-card'>{html_table}</div>", unsafe_allow_html=True)


def inject_theme() -> None:
    """White floating squircle cards, plus rounded clipping for charts."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
        /* Fonts: Space Mono for titles, Plus Jakarta Sans for body */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Space Mono', monospace !important;
        }
        /* Bordered containers -> white floating squircle cards */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF;
            border-radius: 1.75rem;
            border: 1px solid #E3ECF7 !important;
            box-shadow: 0 6px 20px rgba(30, 64, 120, 0.07);
        }
        /* Metrics -> soft light-blue squircle tiles */
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E9F0F9;
            border-radius: 1.25rem;
            padding: 0.9rem 1.1rem;
        }
        /* Right-aligned reporting-month pill */
        .month-pill {
            background: transparent;
            border-radius: 999px;
            padding: 0.4rem 1rem;
            box-shadow: 0 2px 8px rgba(30, 64, 120, 0.08);
            font-size: 0.85rem;
            font-weight: 600;
            color: #0E1526;
            white-space: nowrap;
        }
        /* Type scale (main area only — leaves the sidebar untouched) */
        [data-testid="stMain"] h1 { font-size: 54px !important; font-weight: 700; line-height: 1.1; }
        [data-testid="stMain"] h2 { font-size: 36px !important; }
        [data-testid="stMain"] h3 { font-size: 28px !important; }
        [data-testid="stMain"] h4 { font-size: 22px !important; }
        [data-testid="stMain"] h5 { font-size: 18px !important; }
        [data-testid="stMain"] p,
        [data-testid="stMain"] li { font-size: 16px; }
        /* 60px spacing above each section heading (page title excluded) */
        [data-testid="stMain"] [data-testid="stHeading"] { margin-top: 60px; }
        [data-testid="stMain"] [data-testid="stHeading"]:has(h1) { margin-top: 0; }
        /* Current KPI Status cards */
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E3ECF7;
            border-radius: 1.75rem;
            box-shadow: 0 6px 20px rgba(30, 64, 120, 0.07);
            padding: 1.6rem 1.8rem;
            min-height: 260px;
            margin-bottom: 15px;
        }
        .kpi-card .kpi-title { font-size: 18px; font-weight: 700; color: #0E1526; }
        .kpi-card .kpi-domain { font-size: 14px; color: #5A6474; margin-top: 0.25rem; }
        .kpi-card .kpi-value {
            font-family: 'Space Mono', monospace;
            font-size: 60px; font-weight: 700; color: #0E1526;
            line-height: 1; margin: 1.3rem 0 1.7rem;
        }
        .kpi-card .kpi-line { font-size: 15px; color: #0E1526; margin-top: 0.25rem; }
        /* Priority Ranking table -> white squircle card, no lines, zebra rows */
        .rank-card {
            background: #FFFFFF;
            border: 1px solid #E3ECF7;
            border-radius: 1.75rem;
            box-shadow: 0 6px 20px rgba(30, 64, 120, 0.07);
            padding: 0.75rem 1.5rem 1.25rem;
            overflow-x: auto;
        }
        .rank-table { width: 100%; border-collapse: collapse; font-size: 15px; }
        .rank-table th {
            text-align: left; font-weight: 700; color: #5A6474;
            padding: 16px; border: none !important;
        }
        .rank-table td {
            padding: 18px 16px; border: none !important; color: #0E1526;
        }
        .rank-table tbody tr:nth-child(even) td { background: #F4F8FF; }
        .rank-table, .rank-table thead, .rank-table tbody, .rank-table tr,
        .rank-table thead th { border: none !important; }
        /* Equal-height native cards in a columns row (scoped by container key) */
        .st-key-alert_cards [data-testid="stHorizontalBlock"],
        .st-key-risk_row [data-testid="stHorizontalBlock"] { align-items: stretch; }
        .st-key-alert_cards [data-testid="stColumn"],
        .st-key-risk_row [data-testid="stColumn"] { display: flex; flex-direction: column; }
        .st-key-alert_cards [data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] {
            height: 100%; padding: 2rem 2.25rem;
        }
        .st-key-risk_row [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
            height: 100%; display: flex; flex-direction: column;
        }
        .st-key-risk_row [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > * { flex: 1 1 auto; }
        .st-key-risk_row [data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] { height: 100%; }
        /* Recommendation card -> dark bg, light-blue title, white body */
        .st-key-rec_card,
        .st-key-rec_card [data-testid="stVerticalBlockBorderWrapper"] {
            background: #0E1526 !important;
            border-color: #0E1526 !important;
        }
        .st-key-rec_card h3 { color: #4DA3FF !important; }
        .st-key-rec_card [data-testid="stMarkdownContainer"],
        .st-key-rec_card [data-testid="stMarkdownContainer"] p { color: #FFFFFF !important; }
        /* Top navigation -> horizontal pills, left-aligned (white default) */
        .st-key-topnav { margin-bottom: 40px; }
        .st-key-topnav [data-testid="stButtonGroup"] {
            gap: 0.5rem; flex-wrap: wrap; justify-content: flex-start;
        }
        .st-key-topnav [data-testid="stButtonGroup"] button {
            border-radius: 999px !important;
            border: 1px solid #E3ECF7 !important;
            background: #FFFFFF !important;
            color: #0E1526 !important;
            font-weight: 600 !important;
            padding: 1.55rem 1.2rem !important;
        }
        /* Hover + current page -> black pill, white text */
        .st-key-topnav [data-testid="stButtonGroup"] button:hover,
        .st-key-topnav [data-testid="stButtonGroup"] button[aria-pressed="true"],
        .st-key-topnav [data-testid="stButtonGroup"] button[aria-checked="true"],
        .st-key-topnav [data-testid="stButtonGroup"] button[aria-selected="true"],
        .st-key-topnav [data-testid="stButtonGroup"] button[kind="pillsActive"],
        .st-key-topnav [data-testid="stButtonGroup"] [data-testid*="Active" i] {
            background: #0E1526 !important;
            border-color: #0E1526 !important;
            color: #FFFFFF !important;
        }
        .st-key-topnav [data-testid="stButtonGroup"] button:hover p,
        .st-key-topnav [data-testid="stButtonGroup"] button[aria-pressed="true"] p,
        .st-key-topnav [data-testid="stButtonGroup"] button[aria-checked="true"] p,
        .st-key-topnav [data-testid="stButtonGroup"] button[aria-selected="true"] p,
        .st-key-topnav [data-testid="stButtonGroup"] button[kind="pillsActive"] p,
        .st-key-topnav [data-testid="stButtonGroup"] [data-testid*="Active" i] p {
            color: #FFFFFF !important;
        }
        /* Clip charts to a squircle (keeps the dark headline chart rounded) */
        [data-testid="stVegaLiteChart"],
        [data-testid="stAltairChart"] {
            border-radius: 1.75rem;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_risk_trend_chart(cross: pd.DataFrame) -> None:
    """Headline dark area chart — cross-domain risk score across reporting months."""
    df = pd.DataFrame({
        "Month": pd.to_datetime(cross["reporting_month"]),
        "Risk Score": pd.to_numeric(cross["cross_domain_risk_score"], errors="coerce"),
    }).dropna().sort_values("Month")

    area = alt.Chart(df).mark_area(
        interpolate="monotone",
        line={"color": "#4DA3FF", "strokeWidth": 3},
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color=CHART_BG, offset=0.0),
                   alt.GradientStop(color="#2F6BFF", offset=1.0)],
            x1=1, x2=1, y1=1, y2=0,
        ),
    ).encode(
        x=alt.X("Month:T", title=None,
                axis=alt.Axis(format="%b %y", labelColor="#9AA7BD",
                              tickColor="#3A4356", domainColor="#3A4356", grid=False)),
        y=alt.Y("Risk Score:Q", title=None, scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(labelColor="#9AA7BD", tickColor="#3A4356",
                              domain=False, gridColor="#1E2740")),
        tooltip=[alt.Tooltip("Month:T", format="%b %Y", title="Month"),
                 alt.Tooltip("Risk Score:Q", title="Risk Score")],
    ).properties(height=280)

    chart = (area
             .configure(background=CHART_BG)
             .configure_view(strokeWidth=0, fill=CHART_BG)
             .configure_axis(labelFontSize=12))
    st.altair_chart(chart, use_container_width=True)


def render_warning_chart(df: pd.DataFrame, unit: str) -> None:
    """Dark line chart — KPI value against the flat Warning / Critical boundaries."""
    colors = {"KPI Value": "#4DA3FF",
              "Warning Boundary": "#F0A202",
              "Critical Boundary": "#FF5C5C"}
    order = list(colors)
    long = df.melt("Month", var_name="Series", value_name="Value")

    line = alt.Chart(long).mark_line(interpolate="monotone", strokeWidth=3).encode(
        x=alt.X("Month:T", title=None,
                axis=alt.Axis(format="%b %y", labelColor="#9AA7BD",
                              tickColor="#3A4356", domainColor="#3A4356", grid=False)),
        y=alt.Y("Value:Q", title=unit,
                axis=alt.Axis(labelColor="#9AA7BD", tickColor="#3A4356",
                              domain=False, gridColor="#1E2740", titleColor="#9AA7BD")),
        color=alt.Color("Series:N", sort=order,
                        scale=alt.Scale(domain=order,
                                        range=[colors[s] for s in order]),
                        legend=alt.Legend(title=None, orient="top",
                                          labelColor="#9AA7BD")),
        strokeDash=alt.StrokeDash("Series:N", sort=order, legend=None,
                                  scale=alt.Scale(domain=order,
                                                  range=[[1, 0], [6, 4], [6, 4]])),
        tooltip=[alt.Tooltip("Month:T", format="%b %Y", title="Month"),
                 alt.Tooltip("Series:N", title="Series"),
                 alt.Tooltip("Value:Q", title="Value")],
    ).properties(height=280).interactive()

    chart = (line
             .configure(background=CHART_BG)
             .configure_view(strokeWidth=0, fill=CHART_BG)
             .configure_axis(labelFontSize=12)
             .configure_legend(labelFontSize=12))
    st.altair_chart(chart, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_csv(key: str) -> pd.DataFrame | None:
    """
    Load a named output CSV.
    Returns a DataFrame on success, or None if the file is missing.
    Displays a styled error message in the UI when the file is not found.
    """
    path = OUTPUT_FILES.get(key)
    if path is None:
        st.error(f"Unknown data key: '{key}'")
        return None
    if not path.exists():
        st.error(
            f"**Required file not found:** `{path.relative_to(BASE_DIR)}`  \n"
            f"Run the corresponding analytics script to generate this file before "
            f"launching the dashboard."
        )
        return None
    return pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")


def load_all() -> dict[str, pd.DataFrame | None]:
    """Load all four output files and return as a dict."""
    return {key: load_csv(key) for key in OUTPUT_FILES}


@st.cache_data(show_spinner=False)
def load_input_csv(key: str) -> pd.DataFrame | None:
    """Load a named live input CSV for the Scenario Lab."""
    path = INPUT_FILES[key]
    if not path.exists():
        st.error(f"**Required input file not found:** `{path.relative_to(BASE_DIR)}`")
        return None
    return pd.read_csv(path, encoding="utf-8-sig",
                       keep_default_na=False, na_values=[""])


def get_latest_month(data: dict) -> str:
    """Return the latest reporting month across available files."""
    candidates = []
    if data.get("warning") is not None:
        candidates.append(data["warning"]["reporting_month"].max())
    if data.get("cross") is not None:
        candidates.append(data["cross"]["reporting_month"].max())
    if not candidates:
        return "Unavailable"
    return max(candidates)


def get_forecast_origin(data: dict) -> str:
    """Return the forecast origin month from the forecast file."""
    df = data.get("forecast")
    if df is None or df.empty:
        return "Unavailable"
    return df["forecast_origin_month"].iloc[0]


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar(data: dict) -> None:
    with st.sidebar:
        st.title("Sentinel360")
        st.caption("Healthcare Executive Early-Warning")
        st.divider()

        latest = get_latest_month(data)
        forecast_origin = get_forecast_origin(data)

        st.markdown("**Data Status**")
        st.metric("Latest Reporting Month", latest)
        if forecast_origin != "Unavailable":
            st.metric("Forecast Origin Month", forecast_origin)

        st.divider()
        st.markdown("**Output Files**")
        for key, path in OUTPUT_FILES.items():
            icon = "✅" if path.exists() else "❌"
            st.markdown(f"{icon} `{path.name}`")

        st.divider()
        st.caption(
            "Data reflects pre-computed analytical outputs. "
            "Refresh the page to reload the latest files."
        )


# ──────────────────────────────────────────────────────────────────────────────
# PAGE DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

PAGES = {
    "Executive Overview": {
        "icon": "📊",
        "description": (
            "A single-screen summary of the current executive risk ranking across all six "
            "headline KPIs, the latest cross-domain signal pattern, and the highest-priority "
            "management attention items as of the latest reporting month."
        ),
    },
    "Warning Evidence": {
        "icon": "⚠️",
        "description": (
            "Detailed KPI-level warning and anomaly evidence including status streaks, "
            "deterioration trends, statistical anomaly flags, and alert priorities across "
            "all reporting months."
        ),
    },
    "Forecast": {
        "icon": "📈",
        "description": (
            "Short-term 1–3-month KPI forecasts with prediction intervals, threshold-crossing "
            "flags, forecast risk levels, and confidence ratings based on recent performance trends."
        ),
    },
    "Scenario Lab": {
        "icon": "🔬",
        "description": (
            "Structured scenario exploration for the two approved planning scenarios: "
            "Clinical Staffing Pressure and Facility Capacity Disruption. "
            "Review recommended management actions and cross-domain context."
        ),
    },
    "Data & Validation": {
        "icon": "🗂️",
        "description": (
            "Inspect the underlying output files, review row and column counts, check "
            "data quality statuses, and confirm that all required files are present and "
            "structurally complete."
        ),
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# PAGE RENDERERS
# ──────────────────────────────────────────────────────────────────────────────

def page_executive_overview(data: dict) -> None:
    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.title("Executive Overview")
        st.caption(
            "⚠️ Prototype based on synthetic monthly data for management decision support."
        )
    with head_r:
        st.markdown(
            f"<div style='text-align:right;padding-top:1.4rem'>"
            f"<span class='month-pill'>Data Status: {get_latest_month(data)}</span></div>",
            unsafe_allow_html=True,
        )

    # ── Guard: require all four files ─────────────────────────────────────────
    required = ["ranking", "cross", "warning", "forecast"]
    if any(data.get(k) is None for k in required):
        st.error(
            "One or more required output files are missing. "
            "Run all four analytics scripts before loading this page."
        )
        return

    rank  = data["ranking"]
    cross = data["cross"]
    warn  = data["warning"]

    # ── Derive key values ─────────────────────────────────────────────────────
    latest_month = warn["reporting_month"].max()
    cross_row    = cross[cross["reporting_month"] == latest_month].iloc[0]
    latest_warn  = warn[warn["reporting_month"] == latest_month]

    risk_score   = cross_row["cross_domain_risk_score"]
    risk_level   = cross_row["cross_domain_risk_level"]
    pattern_name = cross_row["signal_pattern_name"]
    mgmt_interp  = cross_row["management_interpretation"]

    rank1        = rank[rank["executive_rank"] == 1].iloc[0]

    # ── Colour helpers ────────────────────────────────────────────────────────
    STATUS_COLOUR = {
        "Critical": "🔴", "Warning": "🟡", "Normal": "🟢",
        "Not Available": "⚪",
    }
    PRIORITY_COLOUR = {
        "Critical": "🔴", "High": "🟠", "Medium": "🟡",
        "Low": "🔵", "None": "⚪",
    }
    RISK_COLOUR = {
        "Critical": "🔴", "High": "🟠", "Moderate": "🟡",
        "Low": "🟢", "Not Available": "⚪",
    }
    LEVEL_COLOUR = {
        "Critical": "🔴", "High": "🟠", "Watch": "🟡",
        "Routine": "🟢", "Not Available": "⚪",
    }

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Reporting month
    # ══════════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Cross-domain risk
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Cross-Domain Risk Signal")
    st.caption("Composite risk score (0–100) across reporting months.")
    render_risk_trend_chart(cross)

    warn_uri = img_data_uri("warning.png")
    note_uri = img_data_uri("note.png")
    st.markdown(
        "<style>"
        f".st-key-risk_card,.st-key-risk_card [data-testid='stVerticalBlockBorderWrapper']{{"
        f"background-image:url('{warn_uri}') !important;background-repeat:no-repeat !important;"
        "background-position:right -20px top -20px !important;background-size:240px !important;"
        "min-height:240px;background-color:#fff !important;}"
        ".st-key-risk_card [data-testid='stMetric']"
        "{background:transparent !important;border:none !important;padding:0.35rem 0;}"
        ".st-key-risk_card [data-testid='stMetricValue']"
        "{white-space:normal !important;overflow-wrap:anywhere;text-wrap:auto;line-height:1.2;}"
        f".st-key-interp_card,.st-key-interp_card [data-testid='stVerticalBlockBorderWrapper']{{"
        f"background-image:url('{note_uri}') !important;background-repeat:no-repeat !important;"
        "background-position:right -20px bottom -20px !important;background-size:240px !important;"
        "background-color:transparent !important;padding-right:30% !important;}"
        "</style>",
        unsafe_allow_html=True,
    )

    with st.container(key="risk_row"):
        risk_col, interp_col = st.columns([1, 2])
        with risk_col:
            with st.container(border=True, key="risk_card"):
                st.caption("Risk Score (0–100)")
                st.markdown(f"## {risk_score}")
                st.metric("Risk Level", f"{RISK_COLOUR.get(risk_level, '')} {risk_level}")
                st.metric("Signal Pattern", pattern_name)
        with interp_col:
            with st.container(border=True, key="interp_card"):
                st.markdown("### Management Interpretation")
                st.write(mgmt_interp)
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Six KPI status cards
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Current KPI Status")

    # Sort by domain order then kpi_code for consistent layout
    domain_order = {"Workforce": 0, "Operations": 1, "Patient Experience": 2}
    kpi_cards = latest_warn.copy()
    kpi_cards["_dom_sort"] = kpi_cards["domain"].map(domain_order).fillna(9)
    kpi_cards = kpi_cards.sort_values(["_dom_sort", "kpi_code"]).reset_index(drop=True)

    cols = st.columns(3)
    for i, (_, row) in enumerate(kpi_cards.iterrows()):
        status_icon   = STATUS_COLOUR.get(row["status"], "")
        priority_icon = PRIORITY_COLOUR.get(row["alert_priority"], "")
        with cols[i % 3]:
            st.markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-title'>{row['kpi_name']}</div>"
                f"<div class='kpi-domain'>Domain: <em>{row['domain']}</em></div>"
                f"<div class='kpi-value'>{row['kpi_value']}</div>"
                f"<div class='kpi-line'>Status: {status_icon} <b>{row['status']}</b></div>"
                f"<div class='kpi-line'>Alert Priority: {priority_icon} "
                f"<b>{row['alert_priority']}</b></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Executive priority ranking table
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Priority Ranking")

    table_df = rank[[
        "kpi_name", "latest_actual_status",
        "executive_priority_score", "executive_priority_level",
        "ranking_driver", "recommended_scenario",
    ]].copy()

    table_df.columns = [
        "KPI", "Current Status", "Priority Score",
        "Priority Level", "Ranking Driver", "Recommended Scenario",
    ]

    # Add colour indicators to status and level
    table_df["Current Status"] = table_df["Current Status"].apply(
        lambda s: f"{STATUS_COLOUR.get(s, '')} {s}"
    )
    table_df["Priority Level"] = table_df["Priority Level"].apply(
        lambda l: f"{LEVEL_COLOUR.get(l, '')} {l}"
    )
    table_df["Priority Score"] = table_df["Priority Score"].map(
        lambda v: f"{float(v):.1f}"
    )

    table_html = table_df.to_html(index=False, border=0,
                                  classes="rank-table", escape=False)
    st.markdown(f"<div class='rank-card'>{table_html}</div>",
                unsafe_allow_html=True)
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Top management alert (Rank 1)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Top Management Alert")
    st.caption(f"Rank 1 of 6 — {rank1['kpi_name']}")

    level_icon = LEVEL_COLOUR.get(rank1["executive_priority_level"], "")
    st.markdown(
        f"{level_icon} **Priority Level: {rank1['executive_priority_level']}** &nbsp;·&nbsp; "
        f"**Priority Score:** {rank1['executive_priority_score']}"
    )

    with st.container(key="alert_cards"):
        sum_col, rec_col = st.columns([2, 1])
        with sum_col:
            with st.container(border=True):
                st.markdown("### Executive Summary")
                st.write(rank1["executive_summary"])
        with rec_col:
            with st.container(border=True, key="rec_card"):
                st.markdown("### Recommendation")
                st.write(rank1["recommended_management_action"])
    st.divider()


def page_warning_evidence(data: dict) -> None:
    st.title("Warning Evidence")
    st.markdown(PAGES["Warning Evidence"]["description"])
    st.divider()

    df_warn = data.get("warning")
    df_kpi = load_input_csv("kpi_data")
    df_config = load_input_csv("kpi_config")
    if df_warn is None or df_kpi is None or df_config is None:
        return

    # KPI selector — the six approved KPIs from config
    labels = {f"{r['kpi_name']} ({r['kpi_code']})": r["kpi_code"]
              for _, r in df_config.iterrows()}
    selected_label = st.selectbox("Select KPI", list(labels))
    kpi_code = labels[selected_label]
    unit = df_config.loc[df_config["kpi_code"] == kpi_code, "unit"].iloc[0]

    warn_kpi = (df_warn[df_warn["kpi_code"] == kpi_code]
                .sort_values("reporting_month").reset_index(drop=True))
    if warn_kpi.empty:
        st.error(f"No warning output rows found for {kpi_code}.")
        return
    latest = warn_kpi.iloc[-1]

    # Monthly trend with the approved Warning / Critical boundaries
    hist = (df_kpi[df_kpi["kpi_code"] == kpi_code]
            .sort_values("reporting_month"))
    chart_df = pd.DataFrame({
        "Month": pd.to_datetime(hist["reporting_month"]),
        "KPI Value": pd.to_numeric(hist["kpi_value"], errors="coerce"),
        "Warning Boundary": float(latest["warning_boundary"]),
        "Critical Boundary": float(latest["critical_boundary"]),
    }).dropna(subset=["KPI Value"])
    render_warning_chart(chart_df, unit)

    # Latest evidence — values taken directly from the warning output file
    st.subheader(f"Latest Evidence — {latest['reporting_month']}")
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            m = st.columns(2)
            m[0].metric(f"KPI Value ({unit})", latest["kpi_value"])
            m[1].metric("Status",
                        f"{STATUS_ICONS.get(latest['status'], '⚪')} {latest['status']}")
            m = st.columns(2)
            m[0].metric("Alert Priority", latest["alert_priority"])
            m[1].metric("Anomaly Flag", latest["anomaly_flag"])
            m = st.columns(2)
            m[0].metric("Consecutive Warning Months",
                        latest["consecutive_warning_months"])
            m[1].metric("Deterioration Streak (months)",
                        latest["deterioration_streak_months"])

    with right:
        with st.container(border=True):
            st.markdown(f"#### {latest['alert_title']}")
            st.write(latest["alert_explanation"])
        with st.container(border=True):
            st.markdown("#### Recommended Management Attention")
            st.write(latest["recommended_management_attention"])

    # Monthly evidence table
    st.subheader("Monthly Evidence")
    evidence = warn_kpi[["reporting_month", "kpi_value", "status",
                         "trend_direction", "anomaly_flag", "alert_priority"]].copy()
    evidence["status"] = evidence["status"].map(
        lambda s: f"{STATUS_ICONS.get(s, '⚪')} {s}")
    evidence.columns = ["Reporting Month", f"KPI Value ({unit})", "Status",
                        "Trend Direction", "Anomaly Flag", "Alert Priority"]
    render_grid(evidence.sort_values("Reporting Month", ascending=False))

    st.caption(latest["analytical_limitation"])


def page_forecast(data: dict) -> None:
    st.title("Forecast")
    st.markdown(PAGES["Forecast"]["description"])
    st.divider()

    df_fc = data.get("forecast")
    df_kpi = load_input_csv("kpi_data")
    if df_fc is None or df_kpi is None:
        return

    # KPI selector
    kpi_pairs = df_fc[["kpi_code", "kpi_name"]].drop_duplicates()
    labels = {f"{r['kpi_name']} ({r['kpi_code']})": r["kpi_code"]
              for _, r in kpi_pairs.iterrows()}
    selected_label = st.selectbox("Select KPI", list(labels))
    kpi_code = labels[selected_label]

    fc = df_fc[df_fc["kpi_code"] == kpi_code].copy()
    fc["forecast_horizon_months"] = pd.to_numeric(fc["forecast_horizon_months"])
    fc = fc.sort_values("forecast_horizon_months").reset_index(drop=True)
    if fc.empty:
        st.error(f"No forecast rows found for {kpi_code}.")
        return
    first = fc.iloc[0]
    unit = first["unit"]

    st.warning("Approximate prototype forecast — not a guaranteed outcome.")

    # Latest actual and origin
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Latest Actual ({unit})", first["latest_actual_value"])
    c2.metric("Latest Actual Status",
              f"{STATUS_ICONS.get(first['latest_actual_status'], '⚪')} "
              f"{first['latest_actual_status']}")
    c3.metric("Forecast Origin Month", first["forecast_origin_month"])

    # Actual history plus central forecast and prediction bounds.
    # The origin-month point is repeated in the forecast series so the
    # forecast lines connect visually to the last actual value.
    hist = df_kpi[df_kpi["kpi_code"] == kpi_code].sort_values("reporting_month")
    hist_df = pd.DataFrame({
        "Month": pd.to_datetime(hist["reporting_month"]),
        "Actual": pd.to_numeric(hist["kpi_value"], errors="coerce"),
    })
    origin_val = float(first["latest_actual_value"])
    fc_df = pd.DataFrame({
        "Month": [pd.to_datetime(first["forecast_origin_month"])]
                 + list(pd.to_datetime(fc["forecast_month"])),
        "Forecast": [origin_val] + list(pd.to_numeric(fc["forecast_value"])),
        "Lower Bound": [origin_val] + list(pd.to_numeric(fc["lower_prediction_bound"])),
        "Upper Bound": [origin_val] + list(pd.to_numeric(fc["upper_prediction_bound"])),
    })
    chart_df = hist_df.merge(fc_df, on="Month", how="outer").sort_values("Month")
    st.line_chart(chart_df, x="Month",
                  y=["Actual", "Forecast", "Lower Bound", "Upper Bound"],
                  color=["#1f77b4", "#d62728", "#bbbbbb", "#888888"])
    st.caption(f"Forecast method: {first['forecast_method']}")

    # Per-month forecast detail — straight from the approved output file
    st.subheader("Forecast Months")
    detail = fc[["forecast_month", "forecast_value", "forecast_status",
                 "projected_direction", "forecast_risk_level",
                 "forecast_confidence", "threshold_crossing_flag"]].copy()
    detail["forecast_status"] = detail["forecast_status"].map(
        lambda s: f"{STATUS_ICONS.get(s, '⚪')} {s}")
    detail.columns = ["Forecast Month", f"Forecast Value ({unit})", "Status",
                      "Projected Direction", "Risk Level", "Confidence",
                      "Threshold Crossing"]
    render_grid(detail)

    # Narrative for the nearest forecast month
    st.markdown(f"**Forecast Message — {first['forecast_month']}**")
    st.info(first["forecast_message"])
    st.markdown("**Recommended Management Attention**")
    st.warning(first["recommended_management_attention"])

    st.caption(first["analytical_limitation"])


SCENARIO_PRESETS = {
    "Baseline (no disruption)": "baseline_value",
    "No intervention":          "default_no_intervention_value",
    "With mitigation":          "default_mitigation_value",
}

STATUS_ICONS = {"Critical": "🔴", "Warning": "🟡", "Normal": "🟢",
                "Not Available": "⚪"}


def page_scenario_lab(data: dict) -> None:
    st.title("Scenario Lab")
    st.markdown(PAGES["Scenario Lab"]["description"])
    st.divider()

    df_kpi = load_input_csv("kpi_data")
    df_config = load_input_csv("kpi_config")
    df_scenarios = load_input_csv("scenarios")
    if df_kpi is None or df_config is None or df_scenarios is None:
        return

    # Scenario + preset selection
    scenario_names = df_scenarios[["scenario_code", "scenario_name"]].drop_duplicates()
    name_to_code = dict(zip(scenario_names["scenario_name"], scenario_names["scenario_code"]))

    col_a, col_b = st.columns(2)
    with col_a:
        selected_name = st.radio("Scenario", list(name_to_code), horizontal=True)
    with col_b:
        preset = st.radio("Preset", list(SCENARIO_PRESETS), horizontal=True)

    scenario_code = name_to_code[selected_name]
    preset_col = SCENARIO_PRESETS[preset]
    df_params = df_scenarios[df_scenarios["scenario_code"] == scenario_code]

    # Parameter sliders, grouped by category. Widget keys include the
    # scenario and preset so switching either re-seeds the defaults.
    st.subheader("Scenario Parameters")
    params: dict[str, float] = {}
    for category in df_params["parameter_category"].unique():
        group = df_params[df_params["parameter_category"] == category]
        st.markdown(f"**{category}**")
        cols = st.columns(len(group))
        for col, (_, p) in zip(cols, group.iterrows()):
            with col:
                params[p["parameter_code"]] = st.slider(
                    label=f"{p['parameter_name']} ({p['unit']})",
                    min_value=float(p["minimum_value"]),
                    max_value=float(p["maximum_value"]),
                    value=float(p[preset_col]),
                    step=float(p["step_value"]),
                    key=f"{scenario_code}|{preset}|{p['parameter_code']}",
                    help=p["business_description"],
                )

    # Live simulation via the analytics engine (no output CSVs involved)
    result = simulate_scenario(df_kpi, df_config, scenario_code, params)

    st.divider()
    st.subheader("Projected KPI Impact")
    st.caption(
        f"Baseline is a live one-month-ahead forecast from origin month "
        f"{result['forecast_origin_month']}, computed by the analytics engine. "
        f"Scenario effects assume a {result['duration_days']:.0f}-day event."
    )

    impact_df = pd.DataFrame([{
        "KPI": r["kpi_name"],
        "Domain": r["domain"],
        "Baseline Forecast": r["baseline_forecast"],
        "Baseline Status": f"{STATUS_ICONS.get(r['baseline_status'], '⚪')} {r['baseline_status']}",
        "Scenario Projection": r["scenario_value"],
        "Scenario Status": f"{STATUS_ICONS.get(r['scenario_status'], '⚪')} {r['scenario_status']}",
        "Impact": r["impact_delta"],
        "Unit": r["unit"],
    } for r in result["rows"]])
    render_grid(impact_df)

    worsened = sum(1 for r in result["rows"]
                   if r["scenario_status"] != r["baseline_status"])
    m1, m2 = st.columns(2)
    m1.metric("KPIs Changing Status", worsened)
    m2.metric("Estimated Mitigation Cost",
              f"MYR {result['estimated_mitigation_cost_myr']:,.0f}")

    st.caption(result["analytical_limitation"])


@st.cache_data(show_spinner="Scanning project files...", ttl=300)
def scan_project_files() -> dict:
    """Read-only structural scan of every expected project file."""
    checklist, coverage, quality = [], [], []
    for group, paths in EXPECTED_PROJECT_FILES.items():
        for path in paths:
            row = {"Group": group, "File": path.name,
                   "Available": "✅" if path.exists() else "❌",
                   "Rows": None, "Columns": None,
                   "Duplicate Key Rows": None, "Duplicate Key": ""}
            if path.exists() and path.suffix == ".csv":
                df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False,
                                 na_values=[""], low_memory=False)
                key = DUPLICATE_KEYS.get(path.name, [df.columns[0]])
                row.update({
                    "Rows": len(df), "Columns": len(df.columns),
                    "Duplicate Key Rows": int(df.duplicated(subset=key).sum()),
                    "Duplicate Key": ", ".join(key),
                })
                month_col = MONTH_COLUMNS.get(path.name)
                if month_col and month_col in df.columns:
                    months = df[month_col].dropna()
                    coverage.append({
                        "File": path.name, "Month Column": month_col,
                        "First Month": months.min(), "Last Month": months.max(),
                        "Distinct Months": months.nunique(),
                    })
                if "data_quality_status" in df.columns:
                    counts = df["data_quality_status"].value_counts()
                    quality.append({
                        "File": path.name,
                        "Pass": int(counts.get("Pass", 0)),
                        "Warning": int(counts.get("Warning", 0)),
                        "Fail": int(counts.get("Fail", 0)),
                    })
            checklist.append(row)

    outputs_status = []
    for path in OUTPUT_FILES.values():
        if path.exists():
            stat = path.stat()
            outputs_status.append({
                "Output File": path.name,
                "Last Generated": datetime.fromtimestamp(stat.st_mtime)
                                          .strftime("%Y-%m-%d %H:%M"),
                "Size (KB)": round(stat.st_size / 1024, 1),
            })
        else:
            outputs_status.append({"Output File": path.name,
                                   "Last Generated": "Missing", "Size (KB)": None})

    ground_truth = {"available": GROUND_TRUTH_PATH.exists(), "sheets": None}
    if ground_truth["available"]:
        try:
            sheets = pd.read_excel(GROUND_TRUTH_PATH, sheet_name=None)
            ground_truth["sheets"] = {name: len(s) for name, s in sheets.items()}
        except Exception as exc:  # openpyxl missing or unreadable workbook
            ground_truth["error"] = str(exc)

    return {
        "checklist": pd.DataFrame(checklist),
        "coverage": pd.DataFrame(coverage),
        "quality": pd.DataFrame(quality),
        "outputs_status": pd.DataFrame(outputs_status),
        "ground_truth": ground_truth,
    }


def page_data_validation(data: dict) -> None:
    st.title("Data & Validation")
    st.markdown(PAGES["Data & Validation"]["description"])
    st.divider()
    st.caption("This page is read-only: it inspects files and previews uploads. "
               "It never modifies project data or analytical outputs.")

    scan = scan_project_files()

    st.subheader("File Availability & Structure")
    render_grid(scan["checklist"])

    st.subheader("Reporting-Month Coverage")
    render_grid(scan["coverage"])

    st.subheader("Data-Quality Status Counts")
    render_grid(scan["quality"])

    st.subheader("Latest Output Generation")
    render_grid(scan["outputs_status"])
    st.caption("Regenerate outputs by running the four analytics scripts; "
               "this page refreshes its scan every five minutes.")

    st.subheader("Ground Truth Workbook")
    gt = scan["ground_truth"]
    if gt["available"]:
        if gt["sheets"]:
            sheet_list = ", ".join(f"{name} ({rows} rows)"
                                   for name, rows in gt["sheets"].items())
            st.markdown(f"`ground_truth_v2.xlsx` — sheets: {sheet_list}")
        elif gt.get("error"):
            st.markdown(f"`ground_truth_v2.xlsx` present (not readable here: {gt['error']})")
    else:
        st.error("`validation/ground_truth_v2.xlsx` not found.")
    st.info("ground_truth_v2.xlsx exists for functional validation only: it holds "
            "the expected analytical results used to verify that the pipeline "
            "scripts compute correctly. It is not an input to any calculation "
            "and is never shown on the analytical pages.")

    # ── Upload preview (in-memory only) ──
    st.divider()
    st.subheader("Upload Preview")
    st.caption("Preview a CSV or Excel file and check its columns against a "
               "project file. Uploads stay in memory — approved project files "
               "are never overwritten from this page.")
    uploaded = st.file_uploader("Upload a CSV or Excel file",
                                type=["csv", "xlsx"])
    if uploaded is None:
        return
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_up = pd.read_csv(uploaded, encoding="utf-8-sig",
                                keep_default_na=False, na_values=[""])
        else:
            df_up = pd.read_excel(uploaded)
    except Exception as exc:
        st.error(f"Could not read `{uploaded.name}`: {exc}")
        return

    st.markdown(f"**`{uploaded.name}`** — {len(df_up):,} rows × "
                f"{len(df_up.columns)} columns")
    st.markdown("**Preview (first 10 rows)**")
    render_grid(df_up.head(10))
    with st.expander(f"Detected columns ({len(df_up.columns)})"):
        st.write(list(df_up.columns))

    reference_files = {p.name: p
                       for paths in EXPECTED_PROJECT_FILES.values()
                       for p in paths if p.suffix == ".csv" and p.exists()}
    choice = st.selectbox("Check required columns against project file",
                          ["(preview only)"] + sorted(reference_files))
    if choice != "(preview only)":
        required = list(pd.read_csv(reference_files[choice], nrows=0,
                                    encoding="utf-8-sig").columns)
        missing = [c for c in required if c not in df_up.columns]
        extra = [c for c in df_up.columns if c not in required]
        if missing:
            st.error(f"Missing required columns ({len(missing)}): "
                     f"{', '.join(missing)}")
        else:
            st.success(f"All {len(required)} required columns of `{choice}` "
                       f"are present.")
        if extra:
            st.caption(f"Extra columns not in `{choice}`: {', '.join(extra)}")


PAGE_RENDERERS = {
    "Executive Overview": page_executive_overview,
    "Warning Evidence":   page_warning_evidence,
    "Forecast":           page_forecast,
    "Scenario Lab":       page_scenario_lab,
    "Data & Validation":  page_data_validation,
}


# ──────────────────────────────────────────────────────────────────────────────
# NAVIGATION
# ───────────────────────────────────────────────────────���──────────────────────

def render_navigation() -> str:
    """Render top-of-page pill navigation and return the selected page name."""
    pages = list(PAGES.keys())
    with st.container(key="topnav"):
        choice = st.pills(
            label="Pages",
            options=pages,
            selection_mode="single",
            default=pages[0],
            label_visibility="collapsed",
        )
    return choice or pages[0]


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_theme()

    # Load data (cached)
    data = load_all()

    # Sidebar: data status + navigation
    render_sidebar(data)
    selected_page = render_navigation()

    # Render selected page
    renderer = PAGE_RENDERERS.get(selected_page)
    if renderer:
        renderer(data)
    else:
        st.error(f"Page '{selected_page}' not found.")


if __name__ == "__main__":
    main()
