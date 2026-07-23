"""
generate_executive_risk_ranking_v1.py

Sentinel360 Healthcare — Executive Risk Ranking Engine
Reads hospital_kpi_data_v2.csv, kpi_warning_output_v1.csv,
cross_domain_signal_output_v1.csv, kpi_forecast_output_v1.csv, and kpi_config_v2.csv,
produces a transparent 6-KPI executive early-warning ranking.

Author: Sentinel360 prototype team
"""

import sys

import pandas as pd
import numpy as np
from pathlib import Path

# Windows consoles default to cp1252, which cannot print the → / ✓ progress glyphs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ──────────────────────────────────────────────────────────
# 1. PATHS
# ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.resolve()

KPI_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "hospital_kpi_data_v2.csv"
    if (BASE_DIR / "data" / "processed" / "hospital_kpi_data_v2.csv").exists()
    else BASE_DIR / "hospital_kpi_data_v2.csv"
)
WARNING_PATH = (
    BASE_DIR / "outputs" / "kpi_warning_output_v1.csv"
    if (BASE_DIR / "outputs" / "kpi_warning_output_v1.csv").exists()
    else BASE_DIR / "kpi_warning_output_v1.csv"
)
CROSS_DOMAIN_PATH = (
    BASE_DIR / "outputs" / "cross_domain_signal_output_v1.csv"
    if (BASE_DIR / "outputs" / "cross_domain_signal_output_v1.csv").exists()
    else BASE_DIR / "cross_domain_signal_output_v1.csv"
)
FORECAST_PATH = (
    BASE_DIR / "outputs" / "kpi_forecast_output_v1.csv"
    if (BASE_DIR / "outputs" / "kpi_forecast_output_v1.csv").exists()
    else BASE_DIR / "kpi_forecast_output_v1.csv"
)
CONFIG_PATH = (
    BASE_DIR / "config" / "kpi_config_v2.csv"
    if (BASE_DIR / "config" / "kpi_config_v2.csv").exists()
    else BASE_DIR / "kpi_config_v2.csv"
)
OUTPUT_PATH = (
    BASE_DIR / "outputs" / "executive_risk_ranking_v1.csv"
    if (BASE_DIR / "outputs").exists()
    else BASE_DIR / "executive_risk_ranking_v1.csv"
)


# ──────────────────────────────────────────────────────────
# 2. CONSTANTS
# ──────────────────────────────────────────────────────────

EXPECTED_KPIS = ["WF_STAFF_LEVEL", "WF_ABSENCE_RATE", "OP_BED_OCCUPANCY",
                 "OP_WAIT_TIME", "PX_COMPLAINT_RATE", "PX_SATISFACTION"]

STATUS_SEVERITY = {"Critical": 3, "Warning": 2, "Normal": 1, "Not Available": 0}
RISK_SEVERITY = {"Critical": 4, "High": 3, "Moderate": 2, "Low": 1, "Not Available": 0}


# ──────────────────────────────────────────────────────────
# 3. VALIDATION
# ──────────────────────────────────────────────────────────

def validate_inputs(df_kpi: pd.DataFrame, df_warn: pd.DataFrame,
                    df_cross: pd.DataFrame, df_forecast: pd.DataFrame,
                    df_cfg: pd.DataFrame) -> list[str]:
    errors = []

    # Column checks
    kpi_cols = {"reporting_month", "kpi_code", "kpi_value", "status", "data_quality_status"}
    missing = kpi_cols - set(df_kpi.columns)
    if missing:
        errors.append(f"Missing columns in hospital_kpi_data_v2.csv: {missing}")

    warn_cols = {"reporting_month", "kpi_code", "kpi_value", "status", "alert_priority",
                 "consecutive_warning_months", "consecutive_critical_months",
                 "deterioration_streak_months", "anomaly_flag", "trend_direction"}
    missing = warn_cols - set(df_warn.columns)
    if missing:
        errors.append(f"Missing columns in kpi_warning_output_v1.csv: {missing}")

    cross_cols = {"reporting_month", "signal_pattern_code", "cross_domain_risk_level", "cross_domain_risk_score"}
    missing = cross_cols - set(df_cross.columns)
    if missing:
        errors.append(f"Missing columns in cross_domain_signal_output_v1.csv: {missing}")

    forecast_cols = {"kpi_code", "forecast_month", "forecast_horizon_months",
                     "forecast_status", "forecast_risk_level", "forecast_confidence",
                     "threshold_crossing_flag", "data_quality_status"}
    missing = forecast_cols - set(df_forecast.columns)
    if missing:
        errors.append(f"Missing columns in kpi_forecast_output_v1.csv: {missing}")

    cfg_cols = {"kpi_code", "kpi_name", "domain", "unit"}
    missing = cfg_cols - set(df_cfg.columns)
    if missing:
        errors.append(f"Missing columns in kpi_config_v2.csv: {missing}")

    # KPI codes
    for df, name in [(df_kpi, "kpi data"), (df_warn, "warning output"), (df_forecast, "forecast output")]:
        actual = set(df["kpi_code"].unique())
        expected = set(EXPECTED_KPIS)
        if actual != expected:
            errors.append(f"KPI mismatch in {name}: expected {expected}, got {actual}")

    # Duplicates
    for df, name in [(df_kpi, "kpi data"), (df_warn, "warning output")]:
        dups = df.duplicated(subset=["reporting_month", "kpi_code"], keep=False).sum()
        if dups:
            errors.append(f"Duplicates found in {name}: {dups} rows")

    # Latest month consistency
    latest_kpi = df_kpi["reporting_month"].max()
    latest_warn = df_warn["reporting_month"].max()
    latest_cross = df_cross["reporting_month"].max()
    if not (latest_kpi == latest_warn == latest_cross):
        errors.append(f"Latest month mismatch: kpi={latest_kpi}, warn={latest_warn}, cross={latest_cross}")

    # Forecast origin check
    forecast_origin = df_forecast["forecast_origin_month"].iloc[0] if len(df_forecast) > 0 else None
    if forecast_origin != latest_kpi:
        errors.append(f"Forecast origin mismatch: {forecast_origin} vs {latest_kpi}")

    # Check 3 forecast rows per KPI
    for kpi in EXPECTED_KPIS:
        count = len(df_forecast[df_forecast["kpi_code"] == kpi])
        if count != 3:
            errors.append(f"KPI {kpi} has {count} forecast rows, expected 3")

    # Check forecast horizons
    horizons = sorted(df_forecast["forecast_horizon_months"].unique())
    if horizons != [1, 2, 3]:
        errors.append(f"Forecast horizons != [1,2,3]: got {horizons}")

    return errors


# ──────────────────────────────────────────────────────────
# 4. SEVERITY HELPERS
# ──────────────────────────────────────────────────────────

def status_severity(status: str) -> int:
    return STATUS_SEVERITY.get(status, 0)


def risk_severity(risk: str) -> int:
    return RISK_SEVERITY.get(risk, 0)


# ──────────────────────────────────────────────────────────
# 5. SCORE CALCULATIONS
# ──────────────────────────────────────────────────────────

def calculate_current_severity(latest_status: str) -> float:
    mapping = {"Normal": 0.0, "Warning": 10.0, "Critical": 20.0, "Not Available": np.nan}
    return mapping.get(latest_status, np.nan)


def calculate_persistence_score(consec_warn: int, consec_crit: int) -> float:
    warn_map = {0: 0, 1: 3, 2: 6, 3: 9}
    warn_score = warn_map.get(min(consec_warn, 4), 12)
    crit_map = {0: 0, 1: 1, 2: 2}
    crit_score = crit_map.get(min(consec_crit, 3), 3)
    return min(float(warn_score + crit_score), 15.0)


def calculate_deterioration_score(deterioration_streak: int) -> float:
    det_map = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}
    return float(det_map.get(min(deterioration_streak, 5), 10))


def calculate_anomaly_component(anomaly_flag: str, trend_direction: str) -> float:
    mapping = {"Not Available": 0.0, "Normal": 0.0, "Moderate Anomaly": 2.0, "Strong Anomaly": 5.0}
    score = mapping.get(anomaly_flag, 0.0)
    if score > 0 and trend_direction == "Improving":
        score = 1.0
    return min(score, 5.0)


def calculate_forecast_exposure(forecast_rows: pd.DataFrame) -> float:
    status_pts = {"Normal": 0, "Warning": 3, "Critical": 5, "Not Available": 0}
    risk_pts = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3, "Not Available": 0}

    status_score = sum(status_pts.get(r, 0) for r in forecast_rows["forecast_status"])
    risk_score = sum(risk_pts.get(r, 0) for r in forecast_rows["forecast_risk_level"])

    harmful_crossings = {"Normal to Warning", "Normal to Critical", "Warning to Critical", "Critical Persists"}
    crossing_bonus = 1.0 if any(c in harmful_crossings for c in forecast_rows["threshold_crossing_flag"]) else 0.0

    return min(float(status_score + risk_score + crossing_bonus), 25.0)


def calculate_cross_domain_score(cross_risk_level: str) -> float:
    mapping = {"Low": 0.0, "Moderate": 5.0, "High": 10.0, "Critical": 15.0, "Not Available": np.nan}
    return mapping.get(cross_risk_level, np.nan)


def derive_data_quality_status(latest_dq: str, forecast_dq_values: list[str]) -> str:
    all_dq = [latest_dq] + forecast_dq_values
    if any(d == "Fail" for d in all_dq):
        return "Fail"
    if any(d == "Warning" for d in all_dq):
        return "Warning"
    return "Pass"


def data_quality_penalty(dq_status: str) -> float:
    mapping = {"Pass": 0.0, "Warning": -5.0, "Fail": -15.0}
    return mapping.get(dq_status, 0.0)


def calculate_priority_score(severity: float, persistence: float, deterioration: float,
                             anomaly: float, forecast_exposure: float,
                             cross_domain: float, dq_penalty: float) -> float:
    if any(pd.isna(v) for v in [severity, persistence, deterioration, anomaly, forecast_exposure, cross_domain]):
        return np.nan
    total = severity + persistence + deterioration + anomaly + forecast_exposure + cross_domain + dq_penalty
    return round(total, 2)


def assign_priority_level(score: float, dq_status: str) -> str:
    if pd.isna(score):
        return "Not Available"
    if dq_status == "Fail":
        # Cap at Watch
        if score < 20:
            return "Routine"
        elif score < 40:
            return "Watch"
        else:
            return "Watch"  # capped
    if score < 20:
        return "Routine"
    if score < 40:
        return "Watch"
    if score < 65:
        return "High"
    return "Critical"


# ──────────────────────────────────────────────────────────
# 6. FORECAST SUMMARY HELPERS
# ──────────────────────────────────────────────────────────

def summarize_forecast_confidence(confidences: list[str]) -> str:
    unique = set(confidences)
    if len(unique) == 1:
        val = list(unique)[0]
        if val == "Not Available" and all(c == "Not Available" for c in confidences):
            return "Not Available"
        return val
    if any(c != "Not Available" for c in confidences):
        return "Mixed"
    return "Not Available"


def determine_earliest_crossing(forecast_rows: pd.DataFrame) -> str:
    new_crossings = {"Normal to Warning", "Normal to Critical", "Warning to Critical"}
    for _, r in forecast_rows.iterrows():
        if r["threshold_crossing_flag"] in new_crossings:
            return r["forecast_month"]
    return "None"


def worst_forecast_status(forecast_rows: pd.DataFrame) -> str:
    statuses = list(forecast_rows["forecast_status"])
    for s in ["Critical", "Warning", "Normal", "Not Available"]:
        if s in statuses:
            return s
    return "Not Available"


def worst_forecast_risk(forecast_rows: pd.DataFrame) -> str:
    risks = list(forecast_rows["forecast_risk_level"])
    for r in ["Critical", "High", "Moderate", "Low", "Not Available"]:
        if r in risks:
            return r
    return "Not Available"


# ──────────────────────────────────────────────────────────
# 7. RANKING DRIVER
# ──────────────────────────────────────────────────────────

def build_ranking_driver(severity: float, persistence: float, deterioration: float,
                         anomaly: float, forecast_exposure: float, cross_domain: float) -> str:
    components = [
        ("Critical current status", severity),
        ("Warning current status", severity),
        ("persistent breach", persistence),
        ("sustained deterioration", deterioration),
        ("strong anomaly", anomaly),
        ("forecast threshold crossing", forecast_exposure),
        ("connected-domain risk", cross_domain),
    ]
    # Filter meaningful components
    drivers = []
    if severity >= 20:
        drivers.append("Critical current status")
    elif severity >= 10:
        drivers.append("Warning current status")
    if persistence >= 10:
        drivers.append("persistent breach")
    elif persistence >= 6:
        drivers.append("sustained warning persistence")
    if deterioration >= 8:
        drivers.append("sustained deterioration")
    elif deterioration >= 4:
        drivers.append("worsening trend")
    if anomaly >= 2:
        drivers.append("anomaly detected")
    if forecast_exposure >= 20:
        drivers.append("Critical forecast exposure")
    elif forecast_exposure >= 12:
        drivers.append("forecast threshold crossing")
    if cross_domain >= 10:
        drivers.append("connected-domain risk")
    elif cross_domain >= 5:
        drivers.append("cross-domain context")

    if not drivers:
        return "Stable conditions; routine monitoring appropriate"

    return "; ".join(drivers[:3])


# ──────────────────────────────────────────────────────────
# 8. EXECUTIVE SUMMARY
# ──────────────────────────────────────────────────────────

def build_executive_summary(rank: int, kpi_name: str, unit: str, latest_value: float,
                            latest_status: str, persistence_months: int,
                            worst_forecast_status_str: str, worst_forecast_risk_str: str,
                            cross_domain_risk: str, cross_domain_pattern: str,
                            priority_level: str, dq_status: str) -> str:
    if dq_status == "Fail":
        return f"Ranked {rank} of 6, {kpi_name} ranking is provisional due to data-quality issues. Validate data completeness before action."

    latest_rounded = round(latest_value, 2) if not pd.isna(latest_value) else "N/A"

    parts = [
        f"Ranked {rank} of 6, {kpi_name} is currently "
        f"{latest_status} at {latest_rounded} {unit}"
    ]

    if persistence_months > 0:
        parts.append(
            f" and has remained outside the acceptable range "
            f"for {persistence_months} consecutive months"
        )

    parts.append(f". The worst three-month forecast status is {worst_forecast_status_str}, with {worst_forecast_risk_str} forecast risk.")

    if cross_domain_risk in ("High", "Critical"):
        parts.append(f" The latest cross-domain signal is {cross_domain_risk} ({cross_domain_pattern}), which should be reviewed in the context of this KPI.")

    if priority_level == "Critical":
        parts.append(" Immediate cross-functional management review and scenario testing is recommended.")
    elif priority_level == "High":
        parts.append(" Management review and scenario testing is recommended.")
    elif priority_level == "Watch":
        parts.append(" Targeted review is recommended.")
    else:
        parts.append(" Routine monitoring remains appropriate.")

    return "".join(parts)


# ──────────────────────────────────────────────────────────
# 9. MANAGEMENT ACTION & SCENARIO
# ──────────────────────────────────────────────────────────

def build_management_action(kpi_code: str, priority_level: str, dq_status: str) -> str:
    if dq_status == "Fail":
        return "Validate data completeness before action."

    base_actions = {
        "WF_STAFF_LEVEL": [
            "Review workforce availability and roster coverage",
            "Examine sustained staffing gaps",
            "Run Clinical Staffing Pressure scenario",
        ],
        "WF_ABSENCE_RATE": [
            "Review absence patterns and workforce availability",
            "Escalate workforce review if Critical persistence continues",
            "Run Clinical Staffing Pressure scenario",
        ],
        "OP_BED_OCCUPANCY": [
            "Review usable-bed pressure and service capacity",
            "Examine blocked-bed and discharge constraints",
            "Run Facility Capacity Disruption scenario",
        ],
        "OP_WAIT_TIME": [
            "Examine queue pressure and service-flow constraints",
            "Review service-area bottlenecks",
            "Run Facility Capacity Disruption scenario",
        ],
        "PX_COMPLAINT_RATE": [
            "Review complaint themes and recurring service issues",
            "Examine links with operational pressure",
            "Run the relevant operational scenario",
        ],
        "PX_SATISFACTION": [
            "Review patient-experience indicators and complaint themes",
            "Conduct cross-functional service review",
            "Run both approved scenarios when connected deterioration persists",
        ],
    }

    actions = base_actions.get(kpi_code, ["Continue routine monitoring"])

    priority_guidance = {
        "Routine": "Continue monitoring.",
        "Watch": "Conduct targeted review.",
        "High": "Schedule management review and scenario testing.",
        "Critical": "Escalate for immediate management review and scenario testing.",
    }

    parts = [priority_guidance.get(priority_level, "Continue monitoring.")]
    parts.extend(actions)
    return " ".join(parts)


def determine_recommended_scenario(kpi_code: str, priority_level: str,
                                   cross_domain_risk: str, cross_domain_pattern: str,
                                   dq_status: str) -> str:
    if dq_status == "Fail":
        return "Data Validation Required"
    if priority_level == "Routine":
        return "No Scenario Required"

    if kpi_code in ("WF_STAFF_LEVEL", "WF_ABSENCE_RATE"):
        return "Clinical Staffing Pressure"
    if kpi_code in ("OP_BED_OCCUPANCY", "OP_WAIT_TIME"):
        return "Facility Capacity Disruption"
    if kpi_code == "PX_COMPLAINT_RATE":
        if cross_domain_pattern == "PATTERN_ALL_DOMAINS":
            return "Both Scenarios"
        return "Facility Capacity Disruption"
    if kpi_code == "PX_SATISFACTION":
        if cross_domain_risk in ("High", "Critical"):
            return "Both Scenarios"
        return "Facility Capacity Disruption"

    return "No Scenario Required"


# ──────────────────────────────────────────────────────────
# 10. MAIN PIPELINE
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Sentinel360 — Executive Risk Ranking Engine")
    print("=" * 60)

    # Load inputs
    print(f"\nReading: {KPI_DATA_PATH}")
    df_kpi = pd.read_csv(KPI_DATA_PATH, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    print(f"  Rows: {len(df_kpi)}")

    print(f"\nReading: {WARNING_PATH}")
    df_warn = pd.read_csv(WARNING_PATH, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    print(f"  Rows: {len(df_warn)}")

    print(f"\nReading: {CROSS_DOMAIN_PATH}")
    df_cross = pd.read_csv(CROSS_DOMAIN_PATH, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    print(f"  Rows: {len(df_cross)}")

    print(f"\nReading: {FORECAST_PATH}")
    df_forecast = pd.read_csv(FORECAST_PATH, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    print(f"  Rows: {len(df_forecast)}")

    print(f"\nReading: {CONFIG_PATH}")
    df_cfg = pd.read_csv(CONFIG_PATH, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    print(f"  Rows: {len(df_cfg)}")

    # Validate
    print("\nValidating inputs...")
    errors = validate_inputs(df_kpi, df_warn, df_cross, df_forecast, df_cfg)
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise ValueError("Input validation failed.")
    print("  Validation passed.")

    # Latest month
    latest_month = df_warn["reporting_month"].max()
    print(f"\nRanking as-of month: {latest_month}")

    # Latest cross-domain context
    cross_latest = df_cross[df_cross["reporting_month"] == latest_month].iloc[0]
    cross_pattern = cross_latest["signal_pattern_code"]
    cross_risk_level = cross_latest["cross_domain_risk_level"]
    cross_risk_score = float(cross_latest["cross_domain_risk_score"])

    # Config lookup
    cfg_lookup = {}
    for _, r in df_cfg.iterrows():
        cfg_lookup[r["kpi_code"]] = {
            "kpi_name": r["kpi_name"],
            "domain": r["domain"],
            "unit": r["unit"],
        }

    # Latest actual lookup
    latest_kpi = df_kpi[df_kpi["reporting_month"] == latest_month]
    latest_kpi_lookup = {}
    for _, r in latest_kpi.iterrows():
        latest_kpi_lookup[r["kpi_code"]] = {
            "value": float(r["kpi_value"]) if pd.notna(r["kpi_value"]) and r["kpi_value"] != "" else np.nan,
            "status": r["status"],
            "dq": r["data_quality_status"],
        }

    # Latest warning lookup
    latest_warn = df_warn[df_warn["reporting_month"] == latest_month]
    latest_warn_lookup = {}
    for _, r in latest_warn.iterrows():
        latest_warn_lookup[r["kpi_code"]] = {
            "alert_priority": r["alert_priority"],
            "consecutive_warning_months": int(r["consecutive_warning_months"]) if pd.notna(r["consecutive_warning_months"]) and r["consecutive_warning_months"] != "" else 0,
            "consecutive_critical_months": int(r["consecutive_critical_months"]) if pd.notna(r["consecutive_critical_months"]) and r["consecutive_critical_months"] != "" else 0,
            "deterioration_streak_months": int(r["deterioration_streak_months"]) if pd.notna(r["deterioration_streak_months"]) and r["deterioration_streak_months"] != "" else 0,
            "anomaly_flag": r["anomaly_flag"],
            "trend_direction": r["trend_direction"],
        }

    # Build records
    records = []
    for kpi_code in EXPECTED_KPIS:
        cfg = cfg_lookup[kpi_code]
        warn = latest_warn_lookup[kpi_code]
        kpi = latest_kpi_lookup[kpi_code]

        # Forecast rows for this KPI
        forecast_rows = df_forecast[df_forecast["kpi_code"] == kpi_code].sort_values("forecast_horizon_months").reset_index(drop=True)

        # Calculate scores
        severity = calculate_current_severity(kpi["status"])
        persistence = calculate_persistence_score(
            warn["consecutive_warning_months"], warn["consecutive_critical_months"]
        )
        deterioration = calculate_deterioration_score(warn["deterioration_streak_months"])
        anomaly = calculate_anomaly_component(warn["anomaly_flag"], warn["trend_direction"])
        forecast_exposure = calculate_forecast_exposure(forecast_rows)
        cross_domain = calculate_cross_domain_score(cross_risk_level)

        # Data quality
        forecast_dq = list(forecast_rows["data_quality_status"])
        dq_status = derive_data_quality_status(kpi["dq"], forecast_dq)
        dq_penalty = data_quality_penalty(dq_status)

        # Priority score
        priority_score = calculate_priority_score(
            severity, persistence, deterioration, anomaly, forecast_exposure, cross_domain, dq_penalty
        )
        priority_level = assign_priority_level(priority_score, dq_status)

        # Forecast summary fields
        forecast_warning_months = int((forecast_rows["forecast_status"] == "Warning").sum())
        forecast_critical_months = int((forecast_rows["forecast_status"] == "Critical").sum())
        forecast_high_risk_months = int((forecast_rows["forecast_risk_level"] == "High").sum())
        forecast_critical_risk_months = int((forecast_rows["forecast_risk_level"] == "Critical").sum())
        earliest_crossing = determine_earliest_crossing(forecast_rows)
        worst_f_status = worst_forecast_status(forecast_rows)
        worst_f_risk = worst_forecast_risk(forecast_rows)
        conf_summary = summarize_forecast_confidence(list(forecast_rows["forecast_confidence"]))

    # Persistence months for summary (use consecutive_warning_months only)
        persistence_months = warn["consecutive_warning_months"]

        record = {
            "ranking_as_of_month": latest_month,
            "executive_rank": 0,  # placeholder
            "kpi_code": kpi_code,
            "kpi_name": cfg["kpi_name"],
            "domain": cfg["domain"],
            "latest_actual_value": round(kpi["value"], 2) if not pd.isna(kpi["value"]) else "",
            "latest_actual_status": kpi["status"],
            "latest_alert_priority": warn["alert_priority"],
            "consecutive_warning_months": int(warn["consecutive_warning_months"]),
            "consecutive_critical_months": int(warn["consecutive_critical_months"]),
            "deterioration_streak_months": int(warn["deterioration_streak_months"]),
            "latest_anomaly_flag": warn["anomaly_flag"],
            "latest_cross_domain_pattern": cross_pattern,
            "latest_cross_domain_risk_level": cross_risk_level,
            "latest_cross_domain_risk_score": round(cross_risk_score, 2),
            "forecast_horizon_count": 3,
            "forecast_warning_months": forecast_warning_months,
            "forecast_critical_months": forecast_critical_months,
            "forecast_high_risk_months": forecast_high_risk_months,
            "forecast_critical_risk_months": forecast_critical_risk_months,
            "earliest_forecast_threshold_crossing": earliest_crossing,
            "worst_forecast_status": worst_f_status,
            "worst_forecast_risk_level": worst_f_risk,
            "forecast_confidence_summary": conf_summary,
            "current_severity_score": round(severity, 2) if not pd.isna(severity) else "",
            "persistence_score": round(persistence, 2),
            "deterioration_score": round(deterioration, 2),
            "anomaly_score_component": round(anomaly, 2),
            "forecast_exposure_score": round(forecast_exposure, 2),
            "cross_domain_score": round(cross_domain, 2) if not pd.isna(cross_domain) else "",
            "data_quality_penalty": round(dq_penalty, 2),
            "executive_priority_score": round(priority_score, 2) if not pd.isna(priority_score) else "",
            "executive_priority_level": priority_level,
            "ranking_driver": build_ranking_driver(severity, persistence, deterioration, anomaly, forecast_exposure, cross_domain),
            "executive_summary": build_executive_summary(
                0, cfg["kpi_name"], cfg["unit"], kpi["value"], kpi["status"],
                persistence_months, worst_f_status, worst_f_risk,
                cross_risk_level, cross_pattern, priority_level, dq_status
            ),
            "recommended_management_action": build_management_action(kpi_code, priority_level, dq_status),
            "recommended_scenario": determine_recommended_scenario(
                kpi_code, priority_level, cross_risk_level, cross_pattern, dq_status
            ),
            "evidence_source": "hospital_kpi_data_v2.csv; kpi_warning_output_v1.csv; cross_domain_signal_output_v1.csv; kpi_forecast_output_v1.csv; kpi_config_v2.csv",
            "data_quality_status": dq_status,
            "analytical_limitation": (
                "Rankings use synthetic monthly data. Score weights are transparent prototype assumptions. "
                "Rankings prioritise management attention, not clinical urgency. "
                "Cross-domain movements do not prove causation. Forecasts use a short six-month recent window. "
                "Thresholds are illustrative prototype assumptions calibrated to the synthetic dataset. "
                "This ranking should support, not replace, human judgement."
            ),
        }
        records.append(record)

    # Build DataFrame
    df_out = pd.DataFrame(records)

    # Assign ranks
    # Sort by: priority_score desc, status_severity desc, worst_forecast_risk_severity desc, kpi_code asc
    df_out["_score"] = df_out["executive_priority_score"].apply(lambda x: float(x) if x != "" and pd.notna(x) else -9999)
    df_out["_status_sev"] = df_out["latest_actual_status"].apply(status_severity)
    df_out["_risk_sev"] = df_out["worst_forecast_risk_level"].apply(risk_severity)
    df_out = df_out.sort_values(["_score", "_status_sev", "_risk_sev", "kpi_code"],
                                 ascending=[False, False, False, True]).reset_index(drop=True)
    df_out["executive_rank"] = np.arange(1, len(df_out) + 1)
    df_out = df_out.drop(columns=["_score", "_status_sev", "_risk_sev"])

    # Update executive summaries with actual rank
    for idx in df_out.index:
        rank = df_out.at[idx, "executive_rank"]
        old_summary = df_out.at[idx, "executive_summary"]
        df_out.at[idx, "executive_summary"] = old_summary.replace("Ranked 0 of 6", f"Ranked {rank} of 6")

    # Sort by executive_rank ascending
    df_out = df_out.sort_values(["executive_rank", "kpi_code"]).reset_index(drop=True)

    # Ensure exact column order (40 columns)
    expected_cols = [
        "ranking_as_of_month", "executive_rank", "kpi_code", "kpi_name", "domain",
        "latest_actual_value", "latest_actual_status", "latest_alert_priority",
        "consecutive_warning_months", "consecutive_critical_months", "deterioration_streak_months",
        "latest_anomaly_flag", "latest_cross_domain_pattern", "latest_cross_domain_risk_level",
        "latest_cross_domain_risk_score", "forecast_horizon_count", "forecast_warning_months",
        "forecast_critical_months", "forecast_high_risk_months", "forecast_critical_risk_months",
        "earliest_forecast_threshold_crossing", "worst_forecast_status", "worst_forecast_risk_level",
        "forecast_confidence_summary", "current_severity_score", "persistence_score",
        "deterioration_score", "anomaly_score_component", "forecast_exposure_score",
        "cross_domain_score", "data_quality_penalty", "executive_priority_score",
        "executive_priority_level", "ranking_driver", "executive_summary",
        "recommended_management_action", "recommended_scenario", "evidence_source",
        "data_quality_status", "analytical_limitation",
    ]
    df_out = df_out[expected_cols]

    # ── Final validation ──
    print("\nFinal validation...")
    validations = []

    if len(df_out) != 6:
        validations.append(f"Row count != 6: got {len(df_out)}")
    else:
        print("  Rows: 6 ✓")

    if len(df_out.columns) != 40:
        validations.append(f"Column count != 40: got {len(df_out.columns)}")
    else:
        print("  Columns: 40 ✓")

    for kpi in EXPECTED_KPIS:
        if kpi not in df_out["kpi_code"].values:
            validations.append(f"KPI {kpi} missing")
    print("  All 6 KPIs present ✓")

    ranks = sorted(df_out["executive_rank"].tolist())
    if ranks != [1, 2, 3, 4, 5, 6]:
        validations.append(f"Ranks != [1,2,3,4,5,6]: got {ranks}")
    else:
        print("  Ranks 1-6 ✓")

    if df_out["executive_rank"].duplicated().any():
        validations.append("Duplicate ranks found")
    else:
        print("  No duplicate ranks ✓")

    # Score component ranges
    for col, min_v, max_v in [
        ("current_severity_score", 0, 20),
        ("persistence_score", 0, 15),
        ("deterioration_score", 0, 10),
        ("anomaly_score_component", 0, 5),
        ("forecast_exposure_score", 0, 25),
        ("cross_domain_score", 0, 15),
    ]:
        vals = df_out[col].apply(lambda x: float(x) if x != "" and pd.notna(x) else np.nan)
        if vals.min() < min_v - 0.01 or vals.max() > max_v + 0.01:
            validations.append(f"{col} out of range [{min_v},{max_v}]")
    print("  Score components in range ✓")

    dq_vals = df_out["data_quality_penalty"].unique()
    for v in dq_vals:
        if v not in (0, -5, -15):
            validations.append(f"Invalid data_quality_penalty: {v}")
    print("  Data quality penalties valid ✓")

    scores = df_out["executive_priority_score"].apply(lambda x: float(x) if x != "" else np.nan)
    if scores.min() < -15 or scores.max() > 90:
        validations.append(f"Priority score out of range: {scores.min()} to {scores.max()}")
    print("  Priority scores in range ✓")

    # Check None preservation
    none_count = (df_out["earliest_forecast_threshold_crossing"] == "None").sum()
    print(f"  None preserved: {none_count} rows ✓")

    if validations:
        print("\nVALIDATION FAILURES:")
        for v in validations:
            print(f"  {v}")
        raise ValueError("Final validation failed.")

    # ── Export ──
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nExported: {OUTPUT_PATH}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Ranking as-of:        {latest_month}")
    print(f"Output rows:          {len(df_out)}")
    print()
    print("Priority level counts:")
    print(df_out["executive_priority_level"].value_counts().to_string())
    print()
    print("Ranking:")
    for _, r in df_out.iterrows():
        print(f"  Rank {r['executive_rank']}: {r['kpi_code']} — score={r['executive_priority_score']}, level={r['executive_priority_level']}")
    print()
    print("Score components (average):")
    for col in ["current_severity_score", "persistence_score", "deterioration_score",
                "anomaly_score_component", "forecast_exposure_score", "cross_domain_score", "data_quality_penalty"]:
        vals = df_out[col].apply(lambda x: float(x) if x != "" else np.nan)
        print(f"  {col}: {vals.mean():.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
