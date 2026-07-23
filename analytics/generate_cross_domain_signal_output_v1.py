"""
generate_cross_domain_signal_output_v1.py

Sentinel360 Healthcare — Cross-Domain Signal Detection Engine
Reads hospital_kpi_data_v2.csv, kpi_warning_output_v1.csv, and kpi_config_v2.csv,
computes domain scores, risk scores, signal patterns, and management guidance,
then writes cross_domain_signal_output_v1.csv.

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

# Preferred project structure with fallback to same folder
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
CONFIG_PATH = (
    BASE_DIR / "config" / "kpi_config_v2.csv"
    if (BASE_DIR / "config" / "kpi_config_v2.csv").exists()
    else BASE_DIR / "kpi_config_v2.csv"
)
OUTPUT_PATH = (
    BASE_DIR / "outputs" / "cross_domain_signal_output_v1.csv"
    if (BASE_DIR / "outputs").exists()
    else BASE_DIR / "cross_domain_signal_output_v1.csv"
)


# ──────────────────────────────────────────────────────────
# 2. CONSTANTS
# ──────────────────────────────────────────────────────────

DOMAIN_KPIS = {
    "Workforce": ["WF_STAFF_LEVEL", "WF_ABSENCE_RATE"],
    "Operations": ["OP_BED_OCCUPANCY", "OP_WAIT_TIME"],
    "Patient Experience": ["PX_COMPLAINT_RATE", "PX_SATISFACTION"],
}

STATUS_POINTS = {
    "Normal": 0,
    "Warning": 1,
    "Critical": 2,
    "Not Available": np.nan,
}

SIGNAL_PATTERNS = {
    "PATTERN_STABLE": "Stable or Isolated Normal Variation",
    "PATTERN_WORKFORCE_ONLY": "Workforce Pressure",
    "PATTERN_OPERATIONS_ONLY": "Operational Pressure",
    "PATTERN_PATIENT_ONLY": "Patient Experience Pressure",
    "PATTERN_WORKFORCE_OPERATIONS": "Workforce-to-Operations Pressure",
    "PATTERN_OPERATIONS_PATIENT": "Operations-to-Patient Experience Pressure",
    "PATTERN_WORKFORCE_PATIENT": "Workforce and Patient Experience Pressure",
    "PATTERN_ALL_DOMAINS": "Connected Workforce-to-Patient Deterioration",
    "PATTERN_INSUFFICIENT_DATA": "Insufficient Data",
}


# ──────────────────────────────────────────────────────────
# 3. VALIDATION
# ──────────────────────────────────────────────────────────

def validate_inputs(df_kpi: pd.DataFrame, df_warn: pd.DataFrame, df_cfg: pd.DataFrame) -> list[str]:
    errors = []

    # Required columns
    kpi_cols = {"reporting_month", "kpi_code", "kpi_value", "status",
                "trend_direction", "data_quality_status"}
    missing = kpi_cols - set(df_kpi.columns)
    if missing:
        errors.append(f"Missing columns in hospital_kpi_data_v2.csv: {missing}")

    warn_cols = {"reporting_month", "kpi_code", "anomaly_flag", "alert_priority"}
    missing = warn_cols - set(df_warn.columns)
    if missing:
        errors.append(f"Missing columns in kpi_warning_output_v1.csv: {missing}")

    cfg_cols = {"kpi_code", "kpi_name", "domain"}
    missing = cfg_cols - set(df_cfg.columns)
    if missing:
        errors.append(f"Missing columns in kpi_config_v2.csv: {missing}")

    # KPI codes
    expected = {"WF_STAFF_LEVEL", "WF_ABSENCE_RATE", "OP_BED_OCCUPANCY",
                "OP_WAIT_TIME", "PX_COMPLAINT_RATE", "PX_SATISFACTION"}
    for df, name in [(df_kpi, "kpi data"), (df_warn, "warning output")]:
        actual = set(df["kpi_code"].unique())
        if actual != expected:
            errors.append(f"KPI mismatch in {name}: expected {expected}, got {actual}")

    # Duplicates
    for df, name in [(df_kpi, "kpi data"), (df_warn, "warning output")]:
        dups = df.duplicated(subset=["reporting_month", "kpi_code"], keep=False).sum()
        if dups:
            errors.append(f"Duplicates found in {name}: {dups} rows")

    # Month coverage match
    kpi_months = set(df_kpi["reporting_month"].unique())
    warn_months = set(df_warn["reporting_month"].unique())
    if kpi_months != warn_months:
        errors.append(f"Month coverage mismatch: kpi={len(kpi_months)}, warn={len(warn_months)}")

    return errors


# ──────────────────────────────────────────────────────────
# 4. CALCULATION HELPERS
# ──────────────────────────────────────────────────────────

def status_to_points(status: str) -> float:
    return STATUS_POINTS.get(status, np.nan)


def score_to_domain_level(score) -> str:
    if pd.isna(score):
        return "Not Available"
    if score == 0:
        return "Normal"
    if score == 1:
        return "Watch"
    if score == 2:
        return "High"
    return "Critical"


def calc_domain_score(df_month: pd.DataFrame, kpi_codes: list[str]) -> float:
    sub = df_month[df_month["kpi_code"].isin(kpi_codes)]
    if len(sub) < 2:
        return np.nan
    points = sub["status"].apply(status_to_points)
    if points.isna().any():
        return np.nan
    return points.sum()


def assign_signal_pattern(levels: dict[str, str]) -> tuple[str, str]:
    """Return (pattern_code, pattern_name) based on domain levels."""
    # Check for insufficient data first
    if any(l == "Not Available" for l in levels.values()):
        return "PATTERN_INSUFFICIENT_DATA", SIGNAL_PATTERNS["PATTERN_INSUFFICIENT_DATA"]

    w = levels["Workforce"]
    o = levels["Operations"]
    p = levels["Patient Experience"]

    w_active = w in ("Watch", "High", "Critical")
    o_active = o in ("Watch", "High", "Critical")
    p_active = p in ("Watch", "High", "Critical")

    if not w_active and not o_active and not p_active:
        return "PATTERN_STABLE", SIGNAL_PATTERNS["PATTERN_STABLE"]
    if w_active and not o_active and not p_active:
        return "PATTERN_WORKFORCE_ONLY", SIGNAL_PATTERNS["PATTERN_WORKFORCE_ONLY"]
    if o_active and not w_active and not p_active:
        return "PATTERN_OPERATIONS_ONLY", SIGNAL_PATTERNS["PATTERN_OPERATIONS_ONLY"]
    if p_active and not w_active and not o_active:
        return "PATTERN_PATIENT_ONLY", SIGNAL_PATTERNS["PATTERN_PATIENT_ONLY"]
    if w_active and o_active and not p_active:
        return "PATTERN_WORKFORCE_OPERATIONS", SIGNAL_PATTERNS["PATTERN_WORKFORCE_OPERATIONS"]
    if o_active and p_active and not w_active:
        return "PATTERN_OPERATIONS_PATIENT", SIGNAL_PATTERNS["PATTERN_OPERATIONS_PATIENT"]
    if w_active and p_active and not o_active:
        return "PATTERN_WORKFORCE_PATIENT", SIGNAL_PATTERNS["PATTERN_WORKFORCE_PATIENT"]
    if w_active and o_active and p_active:
        return "PATTERN_ALL_DOMAINS", SIGNAL_PATTERNS["PATTERN_ALL_DOMAINS"]

    return "PATTERN_STABLE", SIGNAL_PATTERNS["PATTERN_STABLE"]


def calculate_risk_score(domain_scores: dict[str, float],
                         worsening_count: int,
                         anomaly_count: int,
                         connected_count: int) -> float:
    """Transparent cross-domain risk score 0-100."""
    # Domain severity component (max 60)
    sev = 0.0
    for score in domain_scores.values():
        if not pd.isna(score):
            sev += (score / 4.0) * 20.0

    # Worsening component (max 15)
    worsening_comp = (worsening_count / 6.0) * 15.0

    # Strong anomaly component (max 10)
    anomaly_comp = (anomaly_count / 6.0) * 10.0

    # Connected-domain component (max 15)
    connected_map = {0: 0, 1: 3, 2: 8, 3: 15}
    connected_comp = connected_map.get(connected_count, 15)

    total = sev + worsening_comp + anomaly_comp + connected_comp
    return round(min(total, 100.0), 2)


def assign_risk_level(score) -> str:
    if pd.isna(score):
        return "Not Available"
    if score < 20:
        return "Low"
    if score < 40:
        return "Moderate"
    if score < 65:
        return "High"
    return "Critical"


def build_evidence_kpis(df_month: pd.DataFrame) -> str:
    """List KPIs with status above Normal."""
    elevated = df_month[df_month["status"].isin(["Warning", "Critical"])]
    if len(elevated) == 0:
        return "None"
    parts = []
    for _, r in elevated.iterrows():
        parts.append(f"{r['kpi_code']}={r['status']}")
    return "; ".join(parts)


def build_signal_chain(pattern_code: str, levels: dict[str, str]) -> str:
    if pattern_code == "PATTERN_INSUFFICIENT_DATA":
        return "Insufficient data to determine a cross-domain signal pattern."
    if pattern_code == "PATTERN_STABLE":
        return "No cross-domain warning pattern is detected. KPI movements remain within normal or isolated variation."

    active = [d for d, l in levels.items() if l in ("Watch", "High", "Critical")]
    if len(active) == 1:
        return f"{active[0]} indicators show pressure, while other domains remain within normal ranges."
    if len(active) == 2:
        return f"{active[0]} and {active[1]} pressure are present in the same month. This is a plausible connected pattern requiring closer review."
    return "Lower staffing coverage and higher absence coincide with operational pressure and worsening patient-experience indicators. This is a plausible connected pattern, not proof of causation."


def build_management_interpretation(df_month: pd.DataFrame,
                                     levels: dict[str, str],
                                     pattern_code: str,
                                     risk_level: str) -> str:
    if pattern_code == "PATTERN_INSUFFICIENT_DATA":
        return "Data completeness issues prevent reliable cross-domain interpretation."

    active_domains = [d for d, l in levels.items() if l in ("Watch", "High", "Critical")]
    elevated = df_month[df_month["status"].isin(["Warning", "Critical"])]

    parts = []
    if len(active_domains) > 0:
        parts.append(f"{'; '.join(active_domains)} require attention.")

    status_parts = []
    for _, r in elevated.iterrows():
        status_parts.append(f"{r['kpi_name']} is {r['status']}")
    if status_parts:
        parts.append(f"{'; '.join(status_parts)}.")

    if len(active_domains) >= 2:
        parts.append("The simultaneous pattern supports management escalation and scenario testing.")
    elif len(active_domains) == 1:
        parts.append("This is an isolated domain signal; monitor for cross-domain spread.")
    else:
        parts.append("All domains remain within normal parameters.")

    return " ".join(parts)


def build_management_attention(pattern_code: str) -> str:
    mapping = {
        "PATTERN_STABLE": "Continue routine monitoring.",
        "PATTERN_WORKFORCE_ONLY": "Review workforce availability and absence patterns. Consider running Clinical Staffing Pressure scenario.",
        "PATTERN_OPERATIONS_ONLY": "Examine operational flow and capacity pressure. Consider running Facility Capacity Disruption scenario.",
        "PATTERN_PATIENT_ONLY": "Review complaint themes and patient-experience signals.",
        "PATTERN_WORKFORCE_OPERATIONS": "Conduct a joint Workforce–Operations review. Run Clinical Staffing Pressure scenario.",
        "PATTERN_OPERATIONS_PATIENT": "Conduct an Operations–Patient Experience review. Run Facility Capacity Disruption scenario.",
        "PATTERN_WORKFORCE_PATIENT": "Conduct a joint Workforce–Patient Experience review. Examine whether an operational signal is emerging.",
        "PATTERN_ALL_DOMAINS": "Escalate for cross-functional management review. Run both approved scenarios.",
        "PATTERN_INSUFFICIENT_DATA": "Validate data completeness before interpretation.",
    }
    return mapping.get(pattern_code, "Continue routine monitoring.")


def build_data_quality_status(df_month: pd.DataFrame) -> str:
    dq = df_month["data_quality_status"].unique()
    if "Fail" in dq:
        return "Fail"
    if "Warning" in dq:
        return "Warning"
    return "Pass"


# ──────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Sentinel360 — Cross-Domain Signal Detection")
    print("=" * 60)

    # Load inputs
    print(f"\nReading: {KPI_DATA_PATH}")
    df_kpi = pd.read_csv(KPI_DATA_PATH, encoding="utf-8-sig")
    print(f"  Rows: {len(df_kpi)}")

    print(f"\nReading: {WARNING_PATH}")
    df_warn = pd.read_csv(WARNING_PATH, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    print(f"  Rows: {len(df_warn)}")

    print(f"\nReading: {CONFIG_PATH}")
    df_cfg = pd.read_csv(CONFIG_PATH, encoding="utf-8-sig")
    print(f"  Rows: {len(df_cfg)}")

    # Validate
    print("\nValidating inputs...")
    errors = validate_inputs(df_kpi, df_warn, df_cfg)
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise ValueError("Input validation failed.")
    print("  Validation passed.")

    # Parse dates
    df_kpi["reporting_month_dt"] = pd.to_datetime(df_kpi["reporting_month"])
    df_warn["reporting_month_dt"] = pd.to_datetime(df_warn["reporting_month"])

    # Merge anomaly flag from warning output
    df_kpi = df_kpi.merge(
        df_warn[["reporting_month", "kpi_code", "anomaly_flag"]],
        on=["reporting_month", "kpi_code"],
        how="left"
    )

    # Sort
    df_kpi = df_kpi.sort_values(["reporting_month_dt", "domain", "kpi_code"]).reset_index(drop=True)

    # ── Build monthly records ──
    print("\nComputing cross-domain signals...")
    months = sorted(df_kpi["reporting_month_dt"].unique())
    records = []

    for rm_dt in months:
        rm_str = pd.Timestamp(rm_dt).strftime("%Y-%m-%d")
        df_month = df_kpi[df_kpi["reporting_month_dt"] == rm_dt]

        # KPI values and statuses
        kpi_lookup = {}
        for _, r in df_month.iterrows():
            kpi_lookup[r["kpi_code"]] = {
                "value": r["kpi_value"],
                "status": r["status"],
            }

        # Domain scores
        domain_scores = {}
        domain_levels = {}
        for domain_name, kpis in DOMAIN_KPIS.items():
            score = calc_domain_score(df_month, kpis)
            domain_scores[domain_name] = score
            domain_levels[domain_name] = score_to_domain_level(score)

        # Counts
        warning_count = (df_month["status"] == "Warning").sum()
        critical_count = (df_month["status"] == "Critical").sum()
        worsening_count = (df_month["trend_direction"] == "Worsening").sum()
        strong_anomaly_count = (df_month["anomaly_flag"] == "Strong Anomaly").sum()
        connected_count = sum(1 for l in domain_levels.values() if l in ("Watch", "High", "Critical"))

        # Signal pattern
        pattern_code, pattern_name = assign_signal_pattern(domain_levels)

        # Risk score and level
        if pattern_code == "PATTERN_INSUFFICIENT_DATA":
            risk_score = np.nan
            risk_level = "Not Available"
        else:
            risk_score = calculate_risk_score(
                domain_scores, worsening_count, strong_anomaly_count, connected_count
            )
            risk_level = assign_risk_level(risk_score)

        # Evidence KPIs
        evidence = build_evidence_kpis(df_month)

        # Data quality
        dq_status = build_data_quality_status(df_month)
        if dq_status == "Fail":
            pattern_code = "PATTERN_INSUFFICIENT_DATA"
            pattern_name = SIGNAL_PATTERNS["PATTERN_INSUFFICIENT_DATA"]
            risk_score = np.nan
            risk_level = "Not Available"

        # Build text fields
        signal_chain = build_signal_chain(pattern_code, domain_levels)
        mgmt_interp = build_management_interpretation(df_month, domain_levels, pattern_code, risk_level)
        mgmt_attention = build_management_attention(pattern_code)

        # Row
        record = {
            "reporting_month": rm_str,
            "workforce_staffing_value": round(kpi_lookup.get("WF_STAFF_LEVEL", {}).get("value", np.nan), 2) if not pd.isna(kpi_lookup.get("WF_STAFF_LEVEL", {}).get("value", np.nan)) else "",
            "workforce_staffing_status": kpi_lookup.get("WF_STAFF_LEVEL", {}).get("status", "Not Available"),
            "workforce_absence_value": round(kpi_lookup.get("WF_ABSENCE_RATE", {}).get("value", np.nan), 2) if not pd.isna(kpi_lookup.get("WF_ABSENCE_RATE", {}).get("value", np.nan)) else "",
            "workforce_absence_status": kpi_lookup.get("WF_ABSENCE_RATE", {}).get("status", "Not Available"),
            "operations_bed_occupancy_value": round(kpi_lookup.get("OP_BED_OCCUPANCY", {}).get("value", np.nan), 2) if not pd.isna(kpi_lookup.get("OP_BED_OCCUPANCY", {}).get("value", np.nan)) else "",
            "operations_bed_occupancy_status": kpi_lookup.get("OP_BED_OCCUPANCY", {}).get("status", "Not Available"),
            "operations_wait_time_value": round(kpi_lookup.get("OP_WAIT_TIME", {}).get("value", np.nan), 2) if not pd.isna(kpi_lookup.get("OP_WAIT_TIME", {}).get("value", np.nan)) else "",
            "operations_wait_time_status": kpi_lookup.get("OP_WAIT_TIME", {}).get("status", "Not Available"),
            "patient_complaint_rate_value": round(kpi_lookup.get("PX_COMPLAINT_RATE", {}).get("value", np.nan), 2) if not pd.isna(kpi_lookup.get("PX_COMPLAINT_RATE", {}).get("value", np.nan)) else "",
            "patient_complaint_rate_status": kpi_lookup.get("PX_COMPLAINT_RATE", {}).get("status", "Not Available"),
            "patient_satisfaction_value": round(kpi_lookup.get("PX_SATISFACTION", {}).get("value", np.nan), 2) if not pd.isna(kpi_lookup.get("PX_SATISFACTION", {}).get("value", np.nan)) else "",
            "patient_satisfaction_status": kpi_lookup.get("PX_SATISFACTION", {}).get("status", "Not Available"),
            "workforce_domain_score": round(domain_scores["Workforce"], 2) if not pd.isna(domain_scores["Workforce"]) else "",
            "operations_domain_score": round(domain_scores["Operations"], 2) if not pd.isna(domain_scores["Operations"]) else "",
            "patient_experience_domain_score": round(domain_scores["Patient Experience"], 2) if not pd.isna(domain_scores["Patient Experience"]) else "",
            "workforce_domain_level": domain_levels["Workforce"],
            "operations_domain_level": domain_levels["Operations"],
            "patient_experience_domain_level": domain_levels["Patient Experience"],
            "active_warning_kpi_count": int(warning_count),
            "active_critical_kpi_count": int(critical_count),
            "worsening_kpi_count": int(worsening_count),
            "strong_anomaly_kpi_count": int(strong_anomaly_count),
            "connected_domain_count": int(connected_count),
            "signal_pattern_code": pattern_code,
            "signal_pattern_name": pattern_name,
            "cross_domain_risk_score": round(risk_score, 2) if not pd.isna(risk_score) else "",
            "cross_domain_risk_level": risk_level,
            "expected_signal_chain": signal_chain,
            "management_interpretation": mgmt_interp,
            "recommended_management_attention": mgmt_attention,
            "evidence_kpis": evidence,
            "evidence_source": "hospital_kpi_data_v2.csv; kpi_warning_output_v1.csv; kpi_config_v2.csv",
            "data_quality_status": dq_status,
            "analytical_limitation": (
                "The analysis uses synthetic monthly data. Concurrent movements across domains do not prove causation. "
                "Monthly aggregation may hide department-level differences. The risk score uses transparent prototype weights. "
                "Thresholds are illustrative assumptions calibrated to the synthetic dataset, not universal hospital standards."
            ),
        }
        records.append(record)

    # Build DataFrame
    df_out = pd.DataFrame(records)

    # ── Export ──
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nExported: {OUTPUT_PATH}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Reporting months:     {len(df_out)}")
    print(f"Date range:           {df_out['reporting_month'].min()} to {df_out['reporting_month'].max()}")
    print()
    print("Pattern counts:")
    print(df_out["signal_pattern_code"].value_counts().to_string())
    print()
    print("Risk-level counts:")
    print(df_out["cross_domain_risk_level"].value_counts().to_string())
    print()
    high_crit = df_out[df_out["cross_domain_risk_level"].isin(["High", "Critical"])]
    print(f"Months with High/Critical risk: {len(high_crit)}")
    if len(high_crit) > 0:
        for _, r in high_crit.iterrows():
            print(f"  {r['reporting_month']}: {r['cross_domain_risk_level']} (score={r['cross_domain_risk_score']}, pattern={r['signal_pattern_code']})")

    # Ground-truth spot checks
    print(f"\n{'=' * 60}")
    print("GROUND-TRUTH SPOT CHECKS")
    print(f"{'=' * 60}")
    july = df_out[df_out["reporting_month"] == "2024-07-01"].iloc[0]
    print(f"  July 2024: pattern={july['signal_pattern_code']}, risk={july['cross_domain_risk_level']}, score={july['cross_domain_risk_score']}")

    june = df_out[df_out["reporting_month"] == "2026-06-01"].iloc[0]
    print(f"  June 2026: pattern={june['signal_pattern_code']}, risk={june['cross_domain_risk_level']}, score={june['cross_domain_risk_score']}")
    print(f"    W={june['workforce_domain_level']}, O={june['operations_domain_level']}, P={june['patient_experience_domain_level']}")
    print(f"    Evidence: {june['evidence_kpis']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
