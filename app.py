"""
app.py

Sentinel360 Healthcare — Executive Early-Warning Dashboard
Reads pre-computed CSV outputs from outputs/.
Does not recalculate analytical results.

Run from the project root:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path

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
    st.info("Full page content coming soon. KPI warning detail will appear here.")


def page_forecast(data: dict) -> None:
    st.header("📈 Forecast")
    st.markdown(PAGES["Forecast"]["description"])
    st.divider()
    st.info("Full page content coming soon. KPI forecast detail will appear here.")


def page_scenario_lab(data: dict) -> None:
    st.header("🔬 Scenario Lab")
    st.markdown(PAGES["Scenario Lab"]["description"])
    st.divider()
    st.info("Full page content coming soon. Scenario exploration tools will appear here.")


def page_data_validation(data: dict) -> None:
    st.header("🗂️ Data & Validation")
    st.markdown(PAGES["Data & Validation"]["description"])
    st.divider()
    st.info("Full page content coming soon. File inspection tools will appear here.")


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
