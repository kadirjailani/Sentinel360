"""
generate_kpi_warning_output_v1.py

Sentinel360 Healthcare — KPI Warning and Anomaly Detection Engine
Reads hospital_kpi_data_v2.csv and kpi_config_v2.csv,
computes warning streaks, deterioration streaks, rolling z-score anomalies,
and alert priorities, then writes kpi_warning_output_v1.csv.

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

# Try preferred project structure first, fall back to same folder
KPI_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "hospital_kpi_data_v2.csv"
    if (BASE_DIR / "data" / "processed" / "hospital_kpi_data_v2.csv").exists()
    else BASE_DIR / "hospital_kpi_data_v2.csv"
)
KPI_CONFIG_PATH = (
    BASE_DIR / "config" / "kpi_config_v2.csv"
    if (BASE_DIR / "config" / "kpi_config_v2.csv").exists()
    else BASE_DIR / "kpi_config_v2.csv"
)
OUTPUT_PATH = (
    BASE_DIR / "outputs" / "kpi_warning_output_v1.csv"
    if (BASE_DIR / "outputs").exists()
    else BASE_DIR / "kpi_warning_output_v1.csv"
)


# ──────────────────────────────────────────────────────────
# 2. VALIDATION
# ──────────────────────────────────────────────────────────
REQUIRED_KPI_COLS = [
    "reporting_month", "kpi_code", "kpi_name", "domain", "kpi_value",
    "status", "previous_month_value", "month_on_month_change",
    "trend_direction", "threshold_gap", "record_count",
    "numerator_value", "denominator_value",
    "data_quality_status", "data_quality_note", "source_files",
    "calculation_note",
]

REQUIRED_CONFIG_COLS = [
    "kpi_code", "kpi_name", "domain", "unit", "direction",
    "target_value", "warning_threshold", "critical_threshold",
    "minimum_valid", "maximum_valid",
]

EXPECTED_KPIS = {
    "WF_STAFF_LEVEL", "WF_ABSENCE_RATE", "OP_BED_OCCUPANCY",
    "OP_WAIT_TIME", "PX_COMPLAINT_RATE", "PX_SATISFACTION",
}


def validate_inputs(df_kpi: pd.DataFrame, df_config: pd.DataFrame) -> list[str]:
    """Return a list of validation error messages; empty list means pass."""
    errors = []

    # Column checks
    missing_kpi = [c for c in REQUIRED_KPI_COLS if c not in df_kpi.columns]
    if missing_kpi:
        errors.append(f"Missing columns in hospital_kpi_data_v2.csv: {missing_kpi}")

    missing_cfg = [c for c in REQUIRED_CONFIG_COLS if c not in df_config.columns]
    if missing_cfg:
        errors.append(f"Missing columns in kpi_config_v2.csv: {missing_cfg}")

    if errors:
        return errors

    # KPI code checks
    actual_kpis = set(df_kpi["kpi_code"].unique())
    missing_from_data = EXPECTED_KPIS - actual_kpis
    if missing_from_data:
        errors.append(f"Missing KPI codes in data: {missing_from_data}")

    extra_kpis = actual_kpis - EXPECTED_KPIS
    if extra_kpis:
        errors.append(f"Unexpected KPI codes in data: {extra_kpis}")

    # Duplicate check
    dups = df_kpi.duplicated(subset=["reporting_month", "kpi_code"], keep=False).sum()
    if dups:
        errors.append(f"Found {dups} duplicate rows on (reporting_month, kpi_code).")

    # Numeric check
    non_numeric = df_kpi["kpi_value"].apply(lambda x: not (pd.isna(x) or isinstance(x, (int, float))))
    if non_numeric.any():
        errors.append(f"Found {non_numeric.sum()} non-numeric kpi_value entries.")

    # Reporting month format check
    try:
        df_kpi["reporting_month_dt"] = pd.to_datetime(df_kpi["reporting_month"])
    except Exception as e:
        errors.append(f"reporting_month parsing failed: {e}")

    return errors


# ──────────────────────────────────────────────────────────
# 3. CONSTANTS AND CALCULATION HELPERS
# ──────────────────────────────────────────────────────────

# KPI-specific stability tolerances for deterioration streak
STABILITY_TOLERANCE = {
    "WF_STAFF_LEVEL": 0.10,
    "WF_ABSENCE_RATE": 0.10,
    "OP_BED_OCCUPANCY": 0.10,
    "OP_WAIT_TIME": 0.10,
    "PX_COMPLAINT_RATE": 0.02,
    "PX_SATISFACTION": 0.10,
}


def ordinal_word(n: int) -> str:
    """Return ordinal word for small integers."""
    mapping = {1: "first", 2: "second", 3: "third"}
    return mapping.get(n, f"{n}th")


def calc_critical_explanation_gap(row) -> float:
    """
    Compute the gap between KPI value and critical boundary.
    For lower-is-better: value - crit_boundary (positive = beyond critical).
    For higher-is-better: crit_boundary - value (positive = beyond critical).
    Returns absolute positive distance.
    """
    status = row["status"]
    if status != "Critical":
        return np.nan
    val = row["kpi_value"]
    crit = row["critical_boundary"]
    direction = row["direction"]
    if pd.isna(val) or pd.isna(crit):
        return np.nan
    if direction == "higher_is_better":
        return abs(crit - val)
    else:
        return abs(val - crit)

def calc_threshold_breach(status: str) -> str:
    """Map status to threshold_breach label."""
    mapping = {
        "Normal": "None",
        "Warning": "Warning",
        "Critical": "Critical",
        "Not Available": "Not Available",
    }
    return mapping.get(status, "Not Available")


def calc_consecutive_warning(group: pd.DataFrame) -> pd.Series:
    """
    Count consecutive months where status is Warning or Critical.
    Reset to 0 on Normal. Returns a Series aligned to group index.
    """
    statuses = group["status"].values
    counts = np.zeros(len(statuses), dtype=int)
    streak = 0
    for i, s in enumerate(statuses):
        if s in ("Warning", "Critical"):
            streak += 1
            counts[i] = streak
        else:
            streak = 0
            counts[i] = 0
    return pd.Series(counts, index=group.index)


def calc_consecutive_critical(group: pd.DataFrame) -> pd.Series:
    """
    Count consecutive months where status is Critical only.
    Reset on Warning or Normal.
    """
    statuses = group["status"].values
    counts = np.zeros(len(statuses), dtype=int)
    streak = 0
    for i, s in enumerate(statuses):
        if s == "Critical":
            streak += 1
            counts[i] = streak
        else:
            streak = 0
            counts[i] = 0
    return pd.Series(counts, index=group.index)


def calc_deterioration_streak(group: pd.DataFrame) -> pd.Series:
    """
    Count consecutive months where performance is deteriorating.
    Uses KPI-specific stability tolerance from STABILITY_TOLERANCE.
    Higher-is-better: decrease beyond tolerance = deterioration
    Lower-is-better: increase beyond tolerance = deterioration
    Reset on improvement, Stable, or missing previous value.
    """
    code = group["kpi_code"].iloc[0]
    tolerance = STABILITY_TOLERANCE.get(code, 0.10)

    direction = group["direction"].iloc[0]
    values = group["kpi_value"].values
    changes = group["month_on_month_change"].values
    streaks = np.zeros(len(values), dtype=int)

    current_streak = 0
    for i in range(len(values)):
        if i == 0 or pd.isna(changes[i]) or pd.isna(values[i]) or pd.isna(values[i - 1]):
            current_streak = 0
            streaks[i] = 0
            continue

        change = changes[i]

        if direction == "higher_is_better":
            is_deteriorating = change < -tolerance
        elif direction == "lower_is_better":
            is_deteriorating = change > tolerance
        else:  # range_is_better — treat as lower-is-better for deterioration
            is_deteriorating = change > tolerance

        if is_deteriorating:
            current_streak += 1
            streaks[i] = current_streak
        elif abs(change) < tolerance:
            # Stable resets streak
            current_streak = 0
            streaks[i] = 0
        else:
            # Improving resets streak
            current_streak = 0
            streaks[i] = 0

    return pd.Series(streaks, index=group.index)


def calc_anomaly_score(group: pd.DataFrame) -> pd.Series:
    """
    Rolling six-month z-score using ONLY prior observations.
    First 6 months → blank / Not Available.
    """
    values = group["kpi_value"].values
    scores = np.empty(len(values), dtype=float)
    scores[:] = np.nan

    for i in range(len(values)):
        if i < 6:
            scores[i] = np.nan
            continue
        prior = values[i - 6:i]
        prior = prior[~np.isnan(prior)]
        if len(prior) < 6:
            scores[i] = np.nan
            continue
        mean = np.mean(prior)
        std = np.std(prior, ddof=0)
        if std == 0:
            scores[i] = np.nan
        else:
            scores[i] = round((values[i] - mean) / std, 2)

    return pd.Series(scores, index=group.index)


def flag_anomaly(score) -> str:
    if pd.isna(score):
        return "Not Available"
    abs_score = abs(score)
    if abs_score < 1.5:
        return "Normal"
    elif abs_score < 2.0:
        return "Moderate Anomaly"
    else:
        return "Strong Anomaly"


def calc_alert_priority_vec(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized hierarchical alert priority rules.
    Returns a Series of priority strings.
    """
    s = df["status"]
    cw = df["consecutive_warning_months"]
    cc = df["consecutive_critical_months"]
    ds = df["deterioration_streak_months"]
    anomaly = df["anomaly_flag"]
    dq = df["data_quality_status"]
    gap = df["threshold_gap"]
    trend = df["trend_direction"]

    # Start with default "None"
    priority = pd.Series("None", index=df.index)

    # Critical: status=Critical AND (cw>=2 OR cc>=2)
    mask_crit = (s == "Critical") & ((cw >= 2) | (cc >= 2))
    priority[mask_crit] = "Critical"

    # High: status=Critical (not already Critical priority)
    mask_high_crit = (s == "Critical") & ~mask_crit
    priority[mask_high_crit] = "High"

    # High: status=Warning AND ds>=2
    mask_high_warn_ds = (s == "Warning") & (ds >= 2)
    priority[mask_high_warn_ds] = "High"

    # High: status=Warning AND anomaly=Strong Anomaly
    mask_high_warn_anom = (s == "Warning") & (anomaly == "Strong Anomaly")
    priority[mask_high_warn_anom] = "High"

    # Medium: status=Warning (not already High)
    mask_med_warn = (s == "Warning") & (priority == "None")
    priority[mask_med_warn] = "Medium"

    # Medium: status=Normal AND ds>=3 AND gap<2.0
    mask_med_norm = (s == "Normal") & (ds >= 3) & gap.notna() & (gap < 2.0) & (priority == "None")
    priority[mask_med_norm] = "Medium"

    # Low: anomaly is Moderate or Strong, trend=Worsening
    mask_low = anomaly.isin(["Moderate Anomaly", "Strong Anomaly"]) & (trend == "Worsening") & (priority == "None")
    priority[mask_low] = "Low"

    # Low: anomaly is Moderate or Strong, trend is improving/stable (informational)
    mask_low_info = anomaly.isin(["Moderate Anomaly", "Strong Anomaly"]) & (priority == "None")
    priority[mask_low_info] = "Low"

    # Data quality fail caps Warning/Critical at Medium
    mask_dq = (dq == "Fail") & s.isin(["Warning", "Critical"])
    priority[mask_dq] = "Medium"

    return priority


# ──────────────────────────────────────────────────────────
# 4. ALERT TEXT GENERATION
# ──────────────────────────────────────────────────────────

def build_alert_title(row) -> str:
    status = row["status"]
    kpi_name = row["kpi_name"]
    ds = row["deterioration_streak_months"]
    anomaly = row["anomaly_flag"]
    trend = row["trend_direction"]

    if status == "Critical":
        return f"{kpi_name} Reached Critical Level"
    if status == "Warning":
        if ds >= 3:
            return f"{kpi_name} Deteriorating for {ds} Consecutive Months"
        if ds >= 2:
            return f"{kpi_name} Deteriorating for {ds} Consecutive Months"
        return f"{kpi_name} Entered Warning Range"
    if anomaly == "Strong Anomaly" and trend == "Worsening":
        return f"{kpi_name} Shows Strong Negative Anomaly"
    if anomaly in ("Moderate Anomaly", "Strong Anomaly"):
        return f"{kpi_name} Shows Statistical Anomaly"
    return f"{kpi_name} Within Normal Range"


def build_alert_explanation(row) -> str:
    status = row["status"]
    kpi_name = row["kpi_name"]
    val = row["kpi_value"]
    unit = row["unit"]
    change = row["month_on_month_change"]
    warn_bound = row["warning_boundary"]
    crit_bound = row["critical_boundary"]
    cw = row["consecutive_warning_months"]
    ds = row["deterioration_streak_months"]
    anomaly = row["anomaly_flag"]
    anomaly_score = row["anomaly_score"]
    trend = row["trend_direction"]
    dq = row["data_quality_status"]

    parts = []

    if dq == "Fail":
        parts.append(f"{kpi_name} result requires validation due to data-quality concerns.")
        return " ".join(parts)

    if pd.notna(val):
        parts.append(f"{kpi_name} is {val}{' ' + unit if unit != 'Percent (%)' and unit != 'Per 1,000 Encounters' else ''}.")

    if status == "Normal":
        parts.append("Status is Normal.")
    elif status == "Warning":
        parts.append(f"Status is Warning for the {ordinal_word(cw)} consecutive month.")
    elif status == "Critical":
        parts.append(f"Status is Critical for the {ordinal_word(cw)} consecutive month.")

    if pd.notna(change):
        direction_word = "increased" if change > 0 else "decreased"
        parts.append(f"Month-on-month {direction_word} by {abs(change):.2f}.")

    if status == "Warning" and pd.notna(warn_bound):
        gap = abs(row["threshold_gap"])
        parts.append(f"The value is {gap:.2f} beyond the {warn_bound} warning boundary.")
    if status == "Critical" and pd.notna(crit_bound):
        crit_gap = calc_critical_explanation_gap(row)
        if pd.notna(crit_gap):
            parts.append(f"The value is {crit_gap:.2f} beyond the {crit_bound} critical boundary.")

    if anomaly == "Strong Anomaly" and pd.notna(anomaly_score):
        parts.append(f"A strong statistical anomaly is detected (z-score = {anomaly_score}).")
    elif anomaly == "Moderate Anomaly" and pd.notna(anomaly_score):
        parts.append(f"A moderate statistical anomaly is detected (z-score = {anomaly_score}).")

    if ds >= 2 and trend == "Worsening":
        parts.append(f"Deterioration has continued for {ds} consecutive months.")

    return " ".join(parts) if parts else f"{kpi_name} result available."


def build_management_attention(row) -> str:
    priority = row["alert_priority"]
    status = row["status"]
    kpi_code = row["kpi_code"]
    anomaly = row["anomaly_flag"]
    trend = row["trend_direction"]
    dq = row["data_quality_status"]

    if dq == "Fail":
        return "Validate data quality before taking action."

    if priority == "Critical":
        return "Escalate for immediate management review."
    if priority == "High":
        if kpi_code == "WF_STAFF_LEVEL":
            return "Review workforce availability and absence patterns."
        if kpi_code == "WF_ABSENCE_RATE":
            return "Review workforce availability and absence patterns."
        if kpi_code == "OP_BED_OCCUPANCY":
            return "Review bed-capacity constraints and discharge planning."
        if kpi_code == "OP_WAIT_TIME":
            return "Examine queue pressure by service area."
        if kpi_code == "PX_COMPLAINT_RATE":
            return "Review complaint themes and service bottlenecks."
        if kpi_code == "PX_SATISFACTION":
            return "Review complaint themes and service bottlenecks."
        return "Review operational performance urgently."

    if priority == "Medium":
        if kpi_code in ("OP_WAIT_TIME", "OP_BED_OCCUPANCY"):
            return "Examine queue pressure by service area."
        if kpi_code in ("PX_COMPLAINT_RATE", "PX_SATISFACTION"):
            return "Review complaint themes and service bottlenecks."
        if kpi_code in ("WF_STAFF_LEVEL", "WF_ABSENCE_RATE"):
            return "Review workforce availability and absence patterns."
        return "Monitor closely and prepare contingency planning."

    if anomaly in ("Moderate Anomaly", "Strong Anomaly") and trend != "Worsening":
        return "Validate unusual movement before intervention."

    return "Continue routine monitoring."


def build_limitation(row) -> str:
    kpi_name = row["kpi_name"]
    return (
        f"Anomaly detection for {kpi_name} uses a rolling six-month z-score on synthetic monthly data. "
        "The baseline is short and may not capture seasonal patterns. Statistical association does not establish causation. "
        "Thresholds are illustrative prototype assumptions calibrated to the synthetic dataset, not universal hospital standards."
    )


# ──────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Sentinel360 — KPI Warning and Anomaly Detection")
    print("=" * 60)

    # Load inputs
    print(f"\nReading: {KPI_DATA_PATH}")
    df_kpi = pd.read_csv(KPI_DATA_PATH, encoding="utf-8-sig")
    print(f"  Rows: {len(df_kpi)}")

    print(f"\nReading: {KPI_CONFIG_PATH}")
    df_config = pd.read_csv(KPI_CONFIG_PATH, encoding="utf-8-sig")
    print(f"  Rows: {len(df_config)}")

    # Validate
    print("\nValidating inputs...")
    errors = validate_inputs(df_kpi, df_config)
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise ValueError("Input validation failed. See errors above.")
    print("  Validation passed.")

    # Parse dates
    df_kpi["reporting_month_dt"] = pd.to_datetime(df_kpi["reporting_month"])

    # Merge config thresholds
    df = df_kpi.merge(
        df_config[["kpi_code", "warning_threshold", "critical_threshold", "direction"]],
        on="kpi_code",
        how="left",
        suffixes=("", "_cfg")
    )

    # Rename for clarity
    df = df.rename(columns={
        "warning_threshold_cfg": "warning_boundary",
        "critical_threshold_cfg": "critical_boundary",
    })

    # Sort for chronological processing per KPI
    df = df.sort_values(["kpi_code", "reporting_month_dt"]).reset_index(drop=True)

    # ── Compute streaks and anomalies ──
    # Use loop-based assignment to guarantee index alignment
    print("\nComputing warning/critical streaks...")
    df["consecutive_warning_months"] = 0
    df["consecutive_critical_months"] = 0
    df["deterioration_streak_months"] = 0
    df["anomaly_score"] = np.nan

    for code, group in df.groupby("kpi_code", sort=False):
        idx = group.index
        df.loc[idx, "consecutive_warning_months"] = calc_consecutive_warning(group).values
        df.loc[idx, "consecutive_critical_months"] = calc_consecutive_critical(group).values
        df.loc[idx, "deterioration_streak_months"] = calc_deterioration_streak(group).values
        df.loc[idx, "anomaly_score"] = calc_anomaly_score(group).values

    df["anomaly_method"] = "Rolling six-month z-score"
    df["anomaly_flag"] = df["anomaly_score"].apply(flag_anomaly)

    # Override anomaly for data-quality failures
    df.loc[df["data_quality_status"] == "Fail", "anomaly_flag"] = "Not Available"
    df.loc[df["data_quality_status"] == "Fail", "anomaly_score"] = np.nan

    # ── Threshold breach ──
    df["threshold_breach"] = df["status"].apply(calc_threshold_breach)

    # ── Alert priority ──
    print("Computing alert priorities...")
    df["alert_priority"] = calc_alert_priority_vec(df)

    # ── Alert text ──
    print("Generating alert titles and explanations...")
    df["alert_title"] = df.apply(build_alert_title, axis=1)
    df["alert_explanation"] = df.apply(build_alert_explanation, axis=1)
    df["recommended_management_attention"] = df.apply(build_management_attention, axis=1)

    # ── Metadata ──
    df["evidence_source"] = "hospital_kpi_data_v2.csv; kpi_config_v2.csv"
    df["analytical_limitation"] = df.apply(build_limitation, axis=1)

    # ── Final sort ──
    df = df.sort_values(["reporting_month_dt", "domain", "kpi_code"]).reset_index(drop=True)

    # ── Select and reorder output columns ──
    output_cols = [
        "reporting_month", "kpi_code", "kpi_name", "domain", "kpi_value",
        "status", "previous_month_value", "month_on_month_change",
        "trend_direction", "warning_boundary", "critical_boundary",
        "threshold_gap", "threshold_breach", "consecutive_warning_months",
        "consecutive_critical_months", "deterioration_streak_months",
        "anomaly_method", "anomaly_score", "anomaly_flag",
        "alert_priority", "alert_title", "alert_explanation",
        "recommended_management_attention", "evidence_source",
        "data_quality_status", "analytical_limitation",
    ]
    df_out = df[output_cols].copy()

    # ── Export ──
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nExported: {OUTPUT_PATH}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Input rows:           {len(df_kpi)}")
    print(f"Output rows:          {len(df_out)}")
    print(f"Normal rows:          {(df_out['status'] == 'Normal').sum()}")
    print(f"Warning rows:         {(df_out['status'] == 'Warning').sum()}")
    print(f"Critical rows:        {(df_out['status'] == 'Critical').sum()}")
    print(f"High alerts:          {(df_out['alert_priority'] == 'High').sum()}")
    print(f"Critical alerts:      {(df_out['alert_priority'] == 'Critical').sum()}")

    # ── Ground-truth spot checks ──
    print(f"\n{'=' * 60}")
    print("GROUND-TRUTH SPOT CHECKS")
    print(f"{'=' * 60}")
    checks = [
        ("2026-06-01", "WF_STAFF_LEVEL", "Warning"),
        ("2026-06-01", "WF_ABSENCE_RATE", "Critical"),
        ("2026-06-01", "OP_BED_OCCUPANCY", "Normal"),
        ("2026-06-01", "OP_WAIT_TIME", "Warning"),
        ("2026-06-01", "PX_COMPLAINT_RATE", "Warning"),
        ("2026-06-01", "PX_SATISFACTION", "Critical"),
    ]
    for rm, code, expected_status in checks:
        row = df_out[(df_out["reporting_month"] == rm) & (df_out["kpi_code"] == code)]
        if len(row) == 1:
            actual = row["status"].values[0]
            ok = "OK" if actual == expected_status else "MISMATCH"
            print(f"  {ok}: {code} {rm} → status={actual} (expected {expected_status})")
        else:
            print(f"  MISSING: {code} {rm}")

    # Anomaly sanity check
    na_anomalies = (df_out["anomaly_flag"] == "Not Available").sum()
    print(f"\nAnomaly Not Available: {na_anomalies} rows (expected ~36 = 6 KPIs × 6 first months)")

    print("\nDone.")


if __name__ == "__main__":
    main()
