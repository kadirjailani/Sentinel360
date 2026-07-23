"""
app.py

Sentinel360 Healthcare — Executive Early-Warning Dashboard
Reads pre-computed CSV outputs from outputs/.
Does not recalculate analytical results.

Run from the project root:
    streamlit run app.py
"""

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

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
    st.header("📊 Executive Overview")

    # ── Prototype notice ──────────────────────────────────────────────────────
    st.caption(
        "⚠️ Prototype based on synthetic monthly data for management decision support."
    )
    st.divider()

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
    fore  = data["forecast"]

    # ── Derive key values ─────────────────────────────────────────────────────
    latest_month = warn["reporting_month"].max()
    cross_row    = cross[cross["reporting_month"] == latest_month].iloc[0]
    latest_warn  = warn[warn["reporting_month"] == latest_month]

    risk_score   = cross_row["cross_domain_risk_score"]
    risk_level   = cross_row["cross_domain_risk_level"]
    pattern_name = cross_row["signal_pattern_name"]
    mgmt_interp  = cross_row["management_interpretation"]

    rank1        = rank[rank["executive_rank"] == 1].iloc[0]

    warn_forecasts  = int((fore["forecast_status"] == "Warning").sum())
    crit_forecasts  = int((fore["forecast_status"] == "Critical").sum())
    high_risk       = int((fore["forecast_risk_level"] == "High").sum())
    crit_risk       = int((fore["forecast_risk_level"] == "Critical").sum())

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
    st.subheader("Reporting Period")
    st.metric("Latest Reporting Month", latest_month)
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Cross-domain risk
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Cross-Domain Risk Signal")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            "Risk Score (0–100)",
            f"{risk_score}",
            help="Composite score across Workforce, Operations and Patient Experience domains.",
        )
    with col_b:
        st.metric(
            "Risk Level",
            f"{RISK_COLOUR.get(risk_level, '')} {risk_level}",
        )
    with col_c:
        st.metric("Signal Pattern", pattern_name)

    with st.container(border=True):
        st.markdown("**Management Interpretation**")
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
            with st.container(border=True):
                st.markdown(f"**{row['kpi_name']}**")
                st.markdown(f"Domain: *{row['domain']}*")
                val = row["kpi_value"]
                st.metric("Latest Value", val)
                st.markdown(
                    f"Status: {status_icon} **{row['status']}**  \n"
                    f"Alert Priority: {priority_icon} **{row['alert_priority']}**"
                )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Executive priority ranking table
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Executive Priority Ranking")
    st.caption("Ranked by composite priority score as of the latest reporting month.")

    table_df = rank[[
        "executive_rank", "kpi_name", "latest_actual_status",
        "executive_priority_score", "executive_priority_level",
        "ranking_driver", "recommended_scenario",
    ]].copy()

    table_df.columns = [
        "Rank", "KPI", "Current Status",
        "Priority Score", "Priority Level",
        "Ranking Driver", "Recommended Scenario",
    ]

    # Add colour indicators to status and level
    table_df["Current Status"] = table_df["Current Status"].apply(
        lambda s: f"{STATUS_COLOUR.get(s, '')} {s}"
    )
    table_df["Priority Level"] = table_df["Priority Level"].apply(
        lambda l: f"{LEVEL_COLOUR.get(l, '')} {l}"
    )

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank":               st.column_config.NumberColumn(width="small"),
            "KPI":                st.column_config.TextColumn(width="medium"),
            "Current Status":     st.column_config.TextColumn(width="medium"),
            "Priority Score":     st.column_config.NumberColumn(width="small", format="%.1f"),
            "Priority Level":     st.column_config.TextColumn(width="medium"),
            "Ranking Driver":     st.column_config.TextColumn(width="large"),
            "Recommended Scenario": st.column_config.TextColumn(width="medium"),
        },
    )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Top management alert (Rank 1)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Top Management Alert")
    st.caption(f"Rank 1 of 6 — {rank1['kpi_name']}")

    level_icon = LEVEL_COLOUR.get(rank1["executive_priority_level"], "")
    with st.container(border=True):
        st.markdown(
            f"{level_icon} **Priority Level: {rank1['executive_priority_level']}**  \n"
            f"**Priority Score:** {rank1['executive_priority_score']}"
        )
        st.markdown("**Executive Summary**")
        st.write(rank1["executive_summary"])
        st.markdown("**Recommended Management Action**")
        st.write(rank1["recommended_management_action"])
        st.markdown("**Recommended Scenario**")
        st.info(rank1["recommended_scenario"])
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Forecast summary
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("3-Month Forecast Summary")
    st.caption(
        "Counts across all six KPIs over the three forecast months "
        f"({fore['forecast_month'].min()} to {fore['forecast_month'].max()})."
    )

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.metric("Warning Forecasts", warn_forecasts,
                  help="Forecast rows where the central forecast status is Warning.")
    with fc2:
        st.metric("Critical Forecasts", crit_forecasts,
                  help="Forecast rows where the central forecast status is Critical.")
    with fc3:
        st.metric("High Risk Forecasts", high_risk,
                  help="Forecast rows assigned a High forecast risk level.")
    with fc4:
        st.metric("Critical Risk Forecasts", crit_risk,
                  help="Forecast rows assigned a Critical forecast risk level.")


def page_warning_evidence(data: dict) -> None:
    st.header("⚠️ Warning Evidence")
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
        f"KPI Value ({unit})": pd.to_numeric(hist["kpi_value"], errors="coerce"),
        "Warning Boundary": float(latest["warning_boundary"]),
        "Critical Boundary": float(latest["critical_boundary"]),
    })
    st.line_chart(
        chart_df, x="Month",
        y=[f"KPI Value ({unit})", "Warning Boundary", "Critical Boundary"],
        color=["#d62728", "#1f77b4", "#f0a202"],
    )

    # Latest evidence — values taken directly from the warning output file
    st.subheader(f"Latest Evidence — {latest['reporting_month']}")
    row1 = st.columns(3)
    row1[0].metric(f"KPI Value ({unit})", latest["kpi_value"])
    row1[1].metric("Status",
                   f"{STATUS_ICONS.get(latest['status'], '⚪')} {latest['status']}")
    row1[2].metric("Alert Priority", latest["alert_priority"])
    row2 = st.columns(3)
    row2[0].metric("Anomaly Flag", latest["anomaly_flag"])
    row2[1].metric("Consecutive Warning Months", latest["consecutive_warning_months"])
    row2[2].metric("Deterioration Streak (months)", latest["deterioration_streak_months"])

    st.markdown(f"**{latest['alert_title']}**")
    st.info(latest["alert_explanation"])
    st.markdown("**Recommended Management Attention**")
    st.warning(latest["recommended_management_attention"])

    # Monthly evidence table
    st.subheader("Monthly Evidence")
    evidence = warn_kpi[["reporting_month", "kpi_value", "status",
                         "trend_direction", "anomaly_flag", "alert_priority"]].copy()
    evidence["status"] = evidence["status"].map(
        lambda s: f"{STATUS_ICONS.get(s, '⚪')} {s}")
    evidence.columns = ["Reporting Month", f"KPI Value ({unit})", "Status",
                        "Trend Direction", "Anomaly Flag", "Alert Priority"]
    st.dataframe(evidence.sort_values("Reporting Month", ascending=False),
                 width="stretch", hide_index=True)

    st.caption(latest["analytical_limitation"])


def page_forecast(data: dict) -> None:
    st.header("📈 Forecast")
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
    st.dataframe(detail, width="stretch", hide_index=True)

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
    st.header("🔬 Scenario Lab")
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
    st.dataframe(impact_df, width="stretch", hide_index=True)

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
    st.header("🗂️ Data & Validation")
    st.markdown(PAGES["Data & Validation"]["description"])
    st.divider()
    st.caption("This page is read-only: it inspects files and previews uploads. "
               "It never modifies project data or analytical outputs.")

    scan = scan_project_files()

    st.subheader("File Availability & Structure")
    st.dataframe(scan["checklist"], width="stretch", hide_index=True)

    st.subheader("Reporting-Month Coverage")
    st.dataframe(scan["coverage"], width="stretch", hide_index=True)

    st.subheader("Data-Quality Status Counts")
    st.dataframe(scan["quality"], width="stretch", hide_index=True)

    st.subheader("Latest Output Generation")
    st.dataframe(scan["outputs_status"], width="stretch", hide_index=True)
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
    st.dataframe(df_up.head(10), width="stretch", hide_index=True)
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
    """Render top navigation and return the selected page name."""
    page_labels = [f"{v['icon']} {k}" for k, v in PAGES.items()]
    page_names  = list(PAGES.keys())

    with st.sidebar:
        st.markdown("**Navigation**")
        selected_label = st.radio(
            label="Pages",
            options=page_labels,
            label_visibility="collapsed",
        )

    # Strip icon prefix to get clean page name
    selected = next(
        (name for name, meta in PAGES.items()
         if f"{meta['icon']} {name}" == selected_label),
        page_names[0],
    )
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
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
