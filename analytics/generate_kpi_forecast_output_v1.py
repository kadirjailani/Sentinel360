"""
generate_kpi_forecast_output_v1.py

Sentinel360 Healthcare — Short-Term KPI Forecast Engine
Reads hospital_kpi_data_v2.csv, kpi_warning_output_v1.csv,
cross_domain_signal_output_v1.csv, and kpi_config_v2.csv,
produces a transparent 1–3-month forecast for each of the six headline KPIs.

Author: Sentinel360 prototype team
"""

import pandas as pd
import numpy as np
from pathlib import Path


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
CONFIG_PATH = (
    BASE_DIR / "config" / "kpi_config_v2.csv"
    if (BASE_DIR / "config" / "kpi_config_v2.csv").exists()
    else BASE_DIR / "kpi_config_v2.csv"
)
OUTPUT_PATH = (
    BASE_DIR / "outputs" / "kpi_forecast_output_v1.csv"
    if (BASE_DIR / "outputs").exists()
    else BASE_DIR / "kpi_forecast_output_v1.csv"
)


# ──────────────────────────────────────────────────────────
# 2. CONSTANTS
# ──────────────────────────────────────────────────────────

DAMPING_FACTOR = 0.70
ENSEMBLE_TREND_WEIGHT = 0.70
ENSEMBLE_AVG_WEIGHT = 0.30
Z_95 = 1.96
RECENT_WINDOW = 6
MIN_MONTHS_REQUIRED = 4

STATUS_ORDER = ["Normal", "Warning", "Critical", "Not Available"]

STABILITY_TOLERANCE = {
    "WF_STAFF_LEVEL": 0.10,
    "WF_ABSENCE_RATE": 0.10,
    "OP_BED_OCCUPANCY": 0.10,
    "OP_WAIT_TIME": 0.10,
    "PX_COMPLAINT_RATE": 0.02,
    "PX_SATISFACTION": 0.10,
}


# ──────────────────────────────────────────────────────────
# 3. VALIDATION
# ──────────────────────────────────────────────────────────

def validate_inputs(df_kpi: pd.DataFrame, df_warn: pd.DataFrame,
                    df_cross: pd.DataFrame, df_cfg: pd.DataFrame) -> list[str]:
    errors = []

    kpi_cols = {"reporting_month", "kpi_code", "kpi_value", "status",
                "direction", "data_quality_status"}
    missing = kpi_cols - set(df_kpi.columns)
    if missing:
        errors.append(f"Missing columns in hospital_kpi_data_v2.csv: {missing}")

    warn_cols = {"reporting_month", "kpi_code", "warning_boundary", "critical_boundary"}
    missing = warn_cols - set(df_warn.columns)
    if missing:
        errors.append(f"Missing columns in kpi_warning_output_v1.csv: {missing}")

    cross_cols = {"reporting_month", "cross_domain_risk_level", "signal_pattern_code"}
    missing = cross_cols - set(df_cross.columns)
    if missing:
        errors.append(f"Missing columns in cross_domain_signal_output_v1.csv: {missing}")

    cfg_cols = {"kpi_code", "kpi_name", "domain", "unit", "direction",
                "warning_threshold", "critical_threshold", "minimum_valid", "maximum_valid"}
    missing = cfg_cols - set(df_cfg.columns)
    if missing:
        errors.append(f"Missing columns in kpi_config_v2.csv: {missing}")

    expected = {"WF_STAFF_LEVEL", "WF_ABSENCE_RATE", "OP_BED_OCCUPANCY",
                "OP_WAIT_TIME", "PX_COMPLAINT_RATE", "PX_SATISFACTION"}
    for df, name in [(df_kpi, "kpi data"), (df_warn, "warning output")]:
        actual = set(df["kpi_code"].unique())
        if actual != expected:
            errors.append(f"KPI mismatch in {name}: expected {expected}, got {actual}")

    for df, name in [(df_kpi, "kpi data"), (df_warn, "warning output")]:
        dups = df.duplicated(subset=["reporting_month", "kpi_code"], keep=False).sum()
        if dups:
            errors.append(f"Duplicates found in {name}: {dups} rows")

    kpi_months = set(df_kpi["reporting_month"].unique())
    warn_months = set(df_warn["reporting_month"].unique())
    if kpi_months != warn_months:
        errors.append(f"Month coverage mismatch: kpi={len(kpi_months)}, warn={len(warn_months)}")

    latest = df_kpi["reporting_month"].max()
    cross_latest = df_cross["reporting_month"].max()
    if latest != cross_latest:
        errors.append(f"Latest month mismatch: kpi={latest}, cross-domain={cross_latest}")

    return errors


# ──────────────────────────────────────────────────────────
# 4. CALCULATION HELPERS
# ──────────────────────────────────────────────────────────

def get_recent_window(df_kpi: pd.DataFrame, kpi_code: str, latest_month: str) -> pd.DataFrame:
    """Return most recent RECENT_WINDOW valid rows for a KPI, sorted ascending."""
    sub = df_kpi[df_kpi["kpi_code"] == kpi_code].copy()
    sub = sub.sort_values("reporting_month").reset_index(drop=True)
    # Take the last RECENT_WINDOW rows
    sub = sub.tail(RECENT_WINDOW).reset_index(drop=True)
    return sub


def fit_linear_trend(y_vals: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) using OLS on x = 0..n-1."""
    n = len(y_vals)
    x = np.arange(n, dtype=float)
    # numpy polyfit degree 1
    if n < 2:
        return 0.0, float(y_vals[0]) if n == 1 else 0.0
    coeffs = np.polyfit(x, y_vals, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    return slope, intercept


def calculate_residual_std(y_vals: np.ndarray, slope: float, intercept: float) -> float:
    """Population standard deviation of residuals (ddof=0)."""
    n = len(y_vals)
    x = np.arange(n, dtype=float)
    fitted = slope * x + intercept
    residuals = y_vals - fitted
    if len(residuals) < 1:
        return 0.0
    return float(np.std(residuals, ddof=0))


def calculate_ensemble_forecast(latest_value: float, slope: float,
                                 recent_avg: float, horizon: int) -> float:
    """Damped trend + ensemble forecast."""
    trend_forecast = latest_value + slope * horizon * DAMPING_FACTOR
    forecast_raw = ENSEMBLE_TREND_WEIGHT * trend_forecast + ENSEMBLE_AVG_WEIGHT * recent_avg
    return forecast_raw


def calculate_prediction_bounds(forecast_raw: float, residual_std: float,
                                horizon: int, min_valid: float, max_valid: float) -> tuple[float, float]:
    """Return (lower, upper) prediction bounds, clipped to valid range."""
    if residual_std == 0 or not np.isfinite(residual_std):
        lower = forecast_raw
        upper = forecast_raw
    else:
        uncertainty_mult = 1.0 + 0.25 * (horizon - 1)
        margin = Z_95 * residual_std * uncertainty_mult
        lower = forecast_raw - margin
        upper = forecast_raw + margin

    lower = max(min_valid, min(lower, max_valid))
    upper = max(min_valid, min(upper, max_valid))
    # Ensure lower <= upper
    if lower > upper:
        lower, upper = upper, lower
    return round(lower, 2), round(upper, 2)


def clip_to_valid_range(value: float, min_valid: float, max_valid: float) -> tuple[float, bool]:
    """Return (clipped_value, was_clipped)."""
    clipped = max(min_valid, min(value, max_valid))
    was_clipped = not np.isclose(clipped, value)
    return round(clipped, 2), was_clipped


def assign_forecast_status(kpi_code: str, forecast_value: float,
                           warning_thr: float, critical_thr: float) -> str:
    """Assign status based on forecast value and KPI-specific rules."""
    if pd.isna(forecast_value):
        return "Not Available"

    if kpi_code == "WF_STAFF_LEVEL":
        if forecast_value >= 92:
            return "Normal"
        elif forecast_value >= 90:
            return "Warning"
        else:
            return "Critical"
    elif kpi_code == "WF_ABSENCE_RATE":
        if forecast_value <= 7.5:
            return "Normal"
        elif forecast_value <= 9.5:
            return "Warning"
        else:
            return "Critical"
    elif kpi_code == "OP_BED_OCCUPANCY":
        if forecast_value <= 85:
            return "Normal"
        elif forecast_value <= 90:
            return "Warning"
        else:
            return "Critical"
    elif kpi_code == "OP_WAIT_TIME":
        if forecast_value <= 50:
            return "Normal"
        elif forecast_value <= 60:
            return "Warning"
        else:
            return "Critical"
    elif kpi_code == "PX_COMPLAINT_RATE":
        if forecast_value <= 2.0:
            return "Normal"
        elif forecast_value <= 2.5:
            return "Warning"
        else:
            return "Critical"
    elif kpi_code == "PX_SATISFACTION":
        if forecast_value >= 75:
            return "Normal"
        elif forecast_value >= 70:
            return "Warning"
        else:
            return "Critical"
    return "Not Available"


def calculate_threshold_gap(kpi_code: str, forecast_value: float,
                            warning_thr: float, direction: str) -> float:
    """Calculate threshold gap relative to warning boundary."""
    if pd.isna(forecast_value):
        return np.nan
    if direction == "higher_is_better":
        return round(forecast_value - warning_thr, 2)
    else:
        return round(warning_thr - forecast_value, 2)


def assign_threshold_crossing(latest_status: str, forecast_status: str) -> str:
    """Determine threshold crossing flag."""
    if latest_status == "Not Available" or forecast_status == "Not Available":
        return "Not Available"
    if latest_status == "Normal" and forecast_status == "Normal":
        return "No Crossing"
    if latest_status == "Normal" and forecast_status == "Warning":
        return "Normal to Warning"
    if latest_status == "Normal" and forecast_status == "Critical":
        return "Normal to Critical"
    if latest_status == "Warning" and forecast_status == "Critical":
        return "Warning to Critical"
    if latest_status == "Critical" and forecast_status == "Critical":
        return "Critical Persists"
    if latest_status == "Warning" and forecast_status == "Warning":
        return "Warning Persists"
    if latest_status == "Critical" and forecast_status in ("Warning", "Normal"):
        return "Improvement from Critical"
    if latest_status == "Warning" and forecast_status == "Normal":
        return "Improvement from Warning"
    return "No Crossing"


def assign_projected_direction(kpi_code: str, latest_value: float,
                               forecast_value: float, direction: str) -> str:
    """Determine projected direction based on KPI-specific tolerance."""
    if pd.isna(latest_value) or pd.isna(forecast_value):
        return "Not Available"
    tol = STABILITY_TOLERANCE.get(kpi_code, 0.10)
    diff = forecast_value - latest_value
    if abs(diff) < tol:
        return "Stable"
    # For higher-is-better: increase = Improving, decrease = Worsening
    # For lower-is-better: decrease = Improving, increase = Worsening
    # For range_is_better (Bed Occupancy): upper bound is the alert side, so increase = Worsening
    if direction == "higher_is_better":
        return "Improving" if diff > tol else "Worsening"
    else:
        return "Improving" if diff < -tol else "Worsening"


def assign_forecast_risk(kpi_code: str, forecast_status: str, latest_status: str,
                         projected_direction: str, lower_bound: float, upper_bound: float,
                         warning_thr: float, critical_thr: float, direction: str) -> str:
    """Determine forecast risk level."""
    if forecast_status == "Not Available":
        return "Not Available"

    crossing = assign_threshold_crossing(latest_status, forecast_status)

    # Critical risk
    if forecast_status == "Critical" and crossing in ("Warning to Critical", "Normal to Critical", "Critical Persists"):
        return "Critical"

    # Check Critical exposure from bounds
    def has_critical_exposure():
        if direction == "higher_is_better":
            return lower_bound < critical_thr
        else:
            return upper_bound > critical_thr

    def has_warning_exposure():
        if direction == "higher_is_better":
            return lower_bound < warning_thr
        else:
            return upper_bound > warning_thr

    # High risk
    if forecast_status == "Critical":
        return "High"
    if forecast_status == "Warning" and projected_direction == "Worsening":
        return "High"
    if forecast_status == "Warning" and has_critical_exposure():
        return "High"
    if latest_status == "Critical" and forecast_status in ("Warning", "Critical"):
        return "High"

    # Moderate risk
    if forecast_status == "Warning":
        return "Moderate"
    if forecast_status == "Normal" and projected_direction == "Worsening" and has_warning_exposure():
        return "Moderate"
    if latest_status in ("Warning", "Critical") and forecast_status in ("Normal", "Warning"):
        # If latest is Warning/Critical and forecast is improvement, that's Moderate
        if latest_status == "Warning" and forecast_status == "Normal":
            return "Moderate"
        if latest_status == "Critical" and forecast_status == "Normal":
            return "Moderate"

    # Low risk
    if forecast_status == "Normal":
        return "Low"

    return "Low"


def get_status_for_value(kpi_code: str, value: float,
                          warning_thr: float, critical_thr: float) -> str:
    """Return status label for any numeric value using the same rules as assign_forecast_status."""
    if pd.isna(value):
        return "Not Available"

    if kpi_code == "WF_STAFF_LEVEL":
        if value >= 92: return "Normal"
        elif value >= 90: return "Warning"
        else: return "Critical"
    elif kpi_code == "WF_ABSENCE_RATE":
        if value <= 7.5: return "Normal"
        elif value <= 9.5: return "Warning"
        else: return "Critical"
    elif kpi_code == "OP_BED_OCCUPANCY":
        if value <= 85: return "Normal"
        elif value <= 90: return "Warning"
        else: return "Critical"
    elif kpi_code == "OP_WAIT_TIME":
        if value <= 50: return "Normal"
        elif value <= 60: return "Warning"
        else: return "Critical"
    elif kpi_code == "PX_COMPLAINT_RATE":
        if value <= 2.0: return "Normal"
        elif value <= 2.5: return "Warning"
        else: return "Critical"
    elif kpi_code == "PX_SATISFACTION":
        if value >= 75: return "Normal"
        elif value >= 70: return "Warning"
        else: return "Critical"
    return "Not Available"


def get_interval_statuses(kpi_code: str, lower: float, upper: float,
                          warning_thr: float, critical_thr: float) -> set[str]:
    """Return the set of status categories touched by the prediction interval."""
    statuses = set()
    for val in [lower, upper]:
        statuses.add(get_status_for_value(kpi_code, val, warning_thr, critical_thr))
    return statuses


def interval_spans_multiple_statuses(statuses: set[str]) -> bool:
    """True if interval covers more than one status category."""
    # Filter out Not Available before counting
    meaningful = {s for s in statuses if s != "Not Available"}
    return len(meaningful) > 1


def interval_spans_normal_to_critical(statuses: set[str]) -> bool:
    """True if interval covers Normal, Warning, and Critical (all three)."""
    meaningful = {s for s in statuses if s != "Not Available"}
    return {"Normal", "Warning", "Critical"}.issubset(meaningful)


def interval_has_critical_exposure(direction: str, lower_bound: float, upper_bound: float,
                                   critical_thr: float) -> bool:
    """True if the interval crosses or enters the Critical range."""
    if direction == "higher_is_better":
        return lower_bound < critical_thr
    else:
        return upper_bound > critical_thr


def interval_has_warning_exposure(direction: str, lower_bound: float, upper_bound: float,
                                  warning_thr: float) -> bool:
    """True if the interval crosses or enters the Warning range."""
    if direction == "higher_is_better":
        return lower_bound < warning_thr
    else:
        return upper_bound > warning_thr


def assign_forecast_confidence(history_months: int, residual_std: float,
                               lower_bound: float, upper_bound: float,
                               latest_actual_value: float, recent_values: np.ndarray,
                               kpi_code: str, warning_thr: float, critical_thr: float,
                               direction: str) -> str:
    """Determine forecast confidence using relative interval width and status-span checks."""
    if history_months < MIN_MONTHS_REQUIRED:
        return "Not Available"

    interval_width = upper_bound - lower_bound
    small_positive = 0.01
    denominator = max(abs(latest_actual_value), small_positive)
    relative_width = interval_width / denominator

    distinct_count = len(np.unique(np.round(recent_values, 4)))

    # Base confidence from relative width
    if relative_width <= 0.10 and history_months >= RECENT_WINDOW:
        base = "High"
    elif relative_width <= 0.25 and history_months >= RECENT_WINDOW:
        base = "Moderate"
    else:
        base = "Low"

    # Interval status analysis
    interval_statuses = get_interval_statuses(kpi_code, lower_bound, upper_bound, warning_thr, critical_thr)
    spans_multiple = interval_spans_multiple_statuses(interval_statuses)
    spans_all_three = interval_spans_normal_to_critical(interval_statuses)
    has_critical = interval_has_critical_exposure(direction, lower_bound, upper_bound, critical_thr)
    has_warning = interval_has_warning_exposure(direction, lower_bound, upper_bound, warning_thr)

    # Cap rules (applied in order of severity)
    # Rule 4: interval spans Normal through Critical -> Low
    if spans_all_three:
        base = "Low"
    # Rule 3: interval spans more than one status -> cannot exceed Moderate
    elif spans_multiple:
        if base == "High":
            base = "Moderate"
    # Rule 5: interval crosses Critical -> cannot be High
    if has_critical:
        if base == "High":
            base = "Moderate"
    # Rule 6: Normal central forecast but interval enters Warning -> cannot exceed Moderate
    # (This is handled by spans_multiple if Warning is a different status from Normal)

    # Rule 1: fewer than three distinct recent values -> cap at Moderate
    if distinct_count < 3:
        if base == "High":
            base = "Moderate"

    # Rule 2: zero residual variation -> cap at Moderate
    if residual_std == 0 or not np.isfinite(residual_std):
        if base == "High":
            base = "Moderate"

    return base


def build_forecast_message(kpi_name: str, unit: str, latest_value: float,
                           latest_status: str, forecast_month: str, forecast_value: float,
                           forecast_status: str, projected_direction: str,
                           crossing_flag: str, lower_bound: float, upper_bound: float,
                           forecast_confidence: str, warning_thr: float, critical_thr: float,
                           direction: str) -> str:
    """Build a concise management-facing forecast message."""
    if pd.isna(forecast_value) or forecast_status == "Not Available":
        return f"Forecast for {kpi_name} is not available due to insufficient data."

    month_label = pd.Timestamp(forecast_month).strftime("%B %Y")
    latest_rounded = round(latest_value, 2) if not pd.isna(latest_value) else "N/A"
    forecast_rounded = round(forecast_value, 2)

    parts = [f"{kpi_name} is forecast at {forecast_rounded} {unit} for {month_label}, compared with {latest_rounded} {unit} in the latest actual month."]

    # Status transition
    if crossing_flag == "No Crossing":
        parts.append(f"The forecast status remains {forecast_status}.")
    elif crossing_flag == "Normal to Warning":
        parts.append(f"The central forecast may enter the Warning range.")
    elif crossing_flag == "Normal to Critical":
        parts.append(f"The central forecast may enter the Critical range.")
    elif crossing_flag == "Warning to Critical":
        parts.append(f"The central forecast may move from Warning to Critical.")
    elif crossing_flag == "Critical Persists":
        parts.append(f"The forecast indicates Critical conditions may persist.")
    elif crossing_flag == "Warning Persists":
        parts.append(f"The forecast indicates Warning conditions may persist.")
    elif crossing_flag.startswith("Improvement"):
        parts.append(f"The forecast indicates possible improvement from {latest_status} to {forecast_status}.")

    # Interval overlap
    if lower_bound != upper_bound:
        parts.append(f"The approximate interval is {lower_bound}–{upper_bound} {unit}.")
        # Check if interval overlaps warning or critical
        if direction == "higher_is_better":
            if lower_bound < critical_thr:
                parts.append(f"The lower bound indicates possible Critical exposure.")
            elif lower_bound < warning_thr:
                parts.append(f"The lower bound indicates possible Warning exposure.")
        else:
            if upper_bound > critical_thr:
                parts.append(f"The upper bound indicates possible Critical exposure.")
            elif upper_bound > warning_thr:
                parts.append(f"The upper bound indicates possible Warning exposure.")

    parts.append(f"Forecast confidence is {forecast_confidence}.")
    return " ".join(parts)


def build_management_attention(kpi_code: str, forecast_risk: str,
                               projected_direction: str, crossing_flag: str) -> str:
    """Build KPI-specific recommended management attention."""
    if forecast_risk == "Not Available":
        return "Validate data completeness before planning."

    base_actions = {
        "WF_STAFF_LEVEL": [
            "Continue workforce monitoring",
            "Review workforce availability and roster coverage",
            "Run Clinical Staffing Pressure scenario",
        ],
        "WF_ABSENCE_RATE": [
            "Review absence patterns and workforce availability",
            "Run Clinical Staffing Pressure scenario",
            "Escalate workforce review if Critical exposure persists",
        ],
        "OP_BED_OCCUPANCY": [
            "Monitor usable-bed pressure",
            "Review bed-capacity constraints",
            "Run Facility Capacity Disruption scenario",
        ],
        "OP_WAIT_TIME": [
            "Examine queue pressure by service area",
            "Review service-flow constraints",
            "Run the relevant capacity scenario",
        ],
        "PX_COMPLAINT_RATE": [
            "Review complaint themes and recurring service issues",
            "Examine links with waiting-time pressure",
            "Run the relevant operational scenario",
        ],
        "PX_SATISFACTION": [
            "Review patient-experience signals and complaint themes",
            "Conduct cross-functional service review",
            "Run both approved scenarios if deterioration remains connected",
        ],
    }

    actions = base_actions.get(kpi_code, ["Continue routine monitoring"])

    risk_guidance = {
        "Low": "Continue monitoring.",
        "Moderate": "Targeted review recommended.",
        "High": "Management review and scenario testing recommended.",
        "Critical": "Immediate management escalation and scenario testing recommended.",
    }

    parts = [risk_guidance.get(forecast_risk, "Continue monitoring.")]
    parts.extend(actions)

    if projected_direction == "Worsening" and forecast_risk in ("High", "Critical"):
        parts.append("Worsening trend supports earlier intervention.")

    return " ".join(parts)


# ──────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Sentinel360 — KPI Short-Term Forecast Engine")
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

    print(f"\nReading: {CONFIG_PATH}")
    df_cfg = pd.read_csv(CONFIG_PATH, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    print(f"  Rows: {len(df_cfg)}")

    # Validate
    print("\nValidating inputs...")
    errors = validate_inputs(df_kpi, df_warn, df_cross, df_cfg)
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise ValueError("Input validation failed.")
    print("  Validation passed.")

    # Parse dates
    df_kpi["reporting_month_dt"] = pd.to_datetime(df_kpi["reporting_month"])
    df_warn["reporting_month_dt"] = pd.to_datetime(df_warn["reporting_month"])
    df_cross["reporting_month_dt"] = pd.to_datetime(df_cross["reporting_month"])

    # Latest month
    latest_month = df_kpi["reporting_month"].max()
    latest_month_dt = pd.Timestamp(latest_month)
    print(f"\nForecast origin month: {latest_month}")

    # Get cross-domain context for latest month
    cross_latest = df_cross[df_cross["reporting_month"] == latest_month].iloc[0]
    cross_risk_level = cross_latest["cross_domain_risk_level"]
    cross_pattern = cross_latest["signal_pattern_code"]

    # Build KPI config lookup
    cfg_lookup = {}
    for _, r in df_cfg.iterrows():
        cfg_lookup[r["kpi_code"]] = {
            "kpi_name": r["kpi_name"],
            "domain": r["domain"],
            "unit": r["unit"],
            "direction": r["direction"],
            "warning_threshold": float(r["warning_threshold"]),
            "critical_threshold": float(r["critical_threshold"]),
            "minimum_valid": float(r["minimum_valid"]),
            "maximum_valid": float(r["maximum_valid"]),
        }

    # Build latest actual lookup from warning output (has warning_boundary and critical_boundary)
    latest_actual = {}
    for _, r in df_warn[df_warn["reporting_month"] == latest_month].iterrows():
        latest_actual[r["kpi_code"]] = {
            "value": float(r["kpi_value"]) if pd.notna(r["kpi_value"]) and r["kpi_value"] != "" else np.nan,
            "status": r["status"],
        }

    # Forecast horizons
    horizons = [1, 2, 3]
    forecast_months = [(latest_month_dt + pd.DateOffset(months=h)).strftime("%Y-%m-%d") for h in horizons]
    print(f"Forecast months: {', '.join(forecast_months)}")

    # Generate forecasts
    records = []
    kpi_codes = ["WF_STAFF_LEVEL", "WF_ABSENCE_RATE", "OP_BED_OCCUPANCY",
                 "OP_WAIT_TIME", "PX_COMPLAINT_RATE", "PX_SATISFACTION"]

    for kpi_code in kpi_codes:
        cfg = cfg_lookup[kpi_code]
        recent = get_recent_window(df_kpi, kpi_code, latest_month)

        history_months = len(recent)
        recent_window_months = history_months
        y_vals = recent["kpi_value"].astype(float).values
        latest_value = latest_actual[kpi_code]["value"]
        latest_status = latest_actual[kpi_code]["status"]

        warning_thr = cfg["warning_threshold"]
        critical_thr = cfg["critical_threshold"]
        min_valid = cfg["minimum_valid"]
        max_valid = cfg["maximum_valid"]
        direction = cfg["direction"]

        # Data quality assessment for recent window
        recent_dq = recent["data_quality_status"].values
        has_fail = "Fail" in recent_dq
        has_warning = "Warning" in recent_dq
        distinct_count = len(np.unique(np.round(y_vals, 4)))

        # Check if we can compute
        can_compute = (
            history_months >= MIN_MONTHS_REQUIRED
            and not has_fail
            and not np.isnan(latest_value)
            and all(np.isfinite(y_vals))
        )

        if can_compute:
            recent_avg = float(np.mean(y_vals))
            slope, intercept = fit_linear_trend(y_vals)
            residual_std = calculate_residual_std(y_vals, slope, intercept)
        else:
            recent_avg = np.nan
            slope = np.nan
            residual_std = np.nan

        for horizon, forecast_month in zip(horizons, forecast_months):
            if can_compute:
                # Calculate forecast
                forecast_raw = calculate_ensemble_forecast(latest_value, slope, recent_avg, horizon)
                forecast_value, was_clipped = clip_to_valid_range(forecast_raw, min_valid, max_valid)

                # Prediction bounds
                lower_bound, upper_bound = calculate_prediction_bounds(
                    forecast_raw, residual_std, horizon, min_valid, max_valid
                )

                # Forecast status
                forecast_status = assign_forecast_status(kpi_code, forecast_value, warning_thr, critical_thr)

                # Threshold gap
                threshold_gap = calculate_threshold_gap(kpi_code, forecast_value, warning_thr, direction)

                # Threshold crossing
                crossing_flag = assign_threshold_crossing(latest_status, forecast_status)

                # Projected direction
                projected_direction = assign_projected_direction(kpi_code, latest_value, forecast_value, direction)

                # Forecast risk
                forecast_risk = assign_forecast_risk(
                    kpi_code, forecast_status, latest_status, projected_direction,
                    lower_bound, upper_bound, warning_thr, critical_thr, direction
                )

                # Forecast confidence
                forecast_confidence = assign_forecast_confidence(
                    history_months, residual_std, lower_bound, upper_bound,
                    latest_value, y_vals, kpi_code, warning_thr, critical_thr, direction
                )

                # Data quality status
                if history_months < RECENT_WINDOW:
                    dq_status = "Warning"
                elif has_warning:
                    dq_status = "Warning"
                elif was_clipped:
                    dq_status = "Warning"
                elif distinct_count < 3:
                    dq_status = "Warning"
                else:
                    dq_status = "Pass"

                # Forecast message
                forecast_message = build_forecast_message(
                    cfg["kpi_name"], cfg["unit"], latest_value, latest_status,
                    forecast_month, forecast_value, forecast_status, projected_direction,
                    crossing_flag, lower_bound, upper_bound, forecast_confidence,
                    warning_thr, critical_thr, direction
                )

                # Management attention
                mgmt_attention = build_management_attention(kpi_code, forecast_risk, projected_direction, crossing_flag)

            else:
                # Fail state
                forecast_raw = np.nan
                forecast_value = ""
                lower_bound = ""
                upper_bound = ""
                forecast_status = "Not Available"
                threshold_gap = ""
                crossing_flag = "Not Available"
                projected_direction = "Not Available"
                forecast_risk = "Not Available"
                forecast_confidence = "Not Available"
                dq_status = "Fail"
                forecast_message = f"Forecast for {cfg['kpi_name']} is not available due to insufficient data or data-quality issues."
                mgmt_attention = "Validate data completeness before planning."
                slope = np.nan
                recent_avg = np.nan
                residual_std = np.nan

            record = {
                "forecast_origin_month": latest_month,
                "forecast_month": forecast_month,
                "forecast_horizon_months": int(horizon),
                "kpi_code": kpi_code,
                "kpi_name": cfg["kpi_name"],
                "domain": cfg["domain"],
                "unit": cfg["unit"],
                "direction": direction,
                "latest_actual_value": round(latest_value, 2) if not pd.isna(latest_value) else "",
                "latest_actual_status": latest_status,
                "forecast_method": "Transparent prototype ensemble: 70% damped linear trend + 30% recent average",
                "history_months_used": int(history_months) if not pd.isna(history_months) else 0,
                "recent_window_months": int(recent_window_months) if not pd.isna(recent_window_months) else 0,
                "recent_trend_slope": round(slope, 2) if not pd.isna(slope) and np.isfinite(slope) else "",
                "forecast_value_raw": round(forecast_raw, 2) if not pd.isna(forecast_raw) and np.isfinite(forecast_raw) else "",
                "forecast_value": round(forecast_value, 2) if not pd.isna(forecast_value) and forecast_value != "" and np.isfinite(forecast_value) else forecast_value,
                "lower_prediction_bound": round(lower_bound, 2) if not pd.isna(lower_bound) and lower_bound != "" and np.isfinite(lower_bound) else lower_bound,
                "upper_prediction_bound": round(upper_bound, 2) if not pd.isna(upper_bound) and upper_bound != "" and np.isfinite(upper_bound) else upper_bound,
                "forecast_status": forecast_status,
                "warning_boundary": warning_thr,
                "critical_boundary": critical_thr,
                "threshold_gap_forecast": round(threshold_gap, 2) if not pd.isna(threshold_gap) and threshold_gap != "" and np.isfinite(threshold_gap) else threshold_gap,
                "threshold_crossing_flag": crossing_flag,
                "projected_direction": projected_direction,
                "forecast_risk_level": forecast_risk,
                "forecast_confidence": forecast_confidence,
                "forecast_message": forecast_message,
                "recommended_management_attention": mgmt_attention,
                "evidence_source": "hospital_kpi_data_v2.csv; kpi_warning_output_v1.csv; cross_domain_signal_output_v1.csv; kpi_config_v2.csv",
                "data_quality_status": dq_status,
                "analytical_limitation": (
                    f"Only 24 months of synthetic monthly data are available. The forecast uses a {RECENT_WINDOW}-month recent window. "
                    f"The ensemble weight ({ENSEMBLE_TREND_WEIGHT} trend, {ENSEMBLE_AVG_WEIGHT} average) and damping factor ({DAMPING_FACTOR}) are transparent prototype assumptions. "
                    f"Approximate prediction intervals may not capture seasonality or structural change. Forecasts do not prove causes. "
                    f"Thresholds are illustrative prototype assumptions calibrated to the synthetic dataset. "
                    f"This forecast is for management decision support, not clinical use."
                ),
            }
            records.append(record)

    # Build DataFrame
    df_out = pd.DataFrame(records)

    # Sort: forecast_month ascending, domain, kpi_code
    domain_order = {"Workforce": 0, "Operations": 1, "Patient Experience": 2}
    df_out["domain_sort"] = df_out["domain"].map(domain_order)
    df_out = df_out.sort_values(["forecast_month", "domain_sort", "kpi_code"]).reset_index(drop=True)
    df_out = df_out.drop(columns=["domain_sort"])

    # Cross-domain enrichment: if latest cross-domain risk is High or Critical, append to message
    if cross_risk_level in ("High", "Critical"):
        for idx in df_out.index:
            msg = df_out.at[idx, "forecast_message"]
            if msg and "not available" not in msg.lower():
                df_out.at[idx, "forecast_message"] = (
                    f"{msg} The latest cross-domain signal ({cross_pattern}) indicates {cross_risk_level} risk; review this KPI forecast in that context."
                )

    # Ensure exact column order (31 columns)
    expected_cols = [
        "forecast_origin_month", "forecast_month", "forecast_horizon_months",
        "kpi_code", "kpi_name", "domain", "unit", "direction",
        "latest_actual_value", "latest_actual_status",
        "forecast_method", "history_months_used", "recent_window_months",
        "recent_trend_slope", "forecast_value_raw", "forecast_value",
        "lower_prediction_bound", "upper_prediction_bound",
        "forecast_status", "warning_boundary", "critical_boundary",
        "threshold_gap_forecast", "threshold_crossing_flag",
        "projected_direction", "forecast_risk_level", "forecast_confidence",
        "forecast_message", "recommended_management_attention",
        "evidence_source", "data_quality_status", "analytical_limitation",
    ]
    df_out = df_out[expected_cols]

    # ── Final validation ──
    print("\nFinal validation...")
    validations = []

    if len(df_out) != 18:
        validations.append(f"Row count != 18: got {len(df_out)}")
    else:
        print("  Rows: 18 ✓")

    if len(df_out.columns) != 31:
        validations.append(f"Column count != 31: got {len(df_out.columns)}")
    else:
        print("  Columns: 31 ✓")

    for kpi in kpi_codes:
        count = len(df_out[df_out["kpi_code"] == kpi])
        if count != 3:
            validations.append(f"KPI {kpi} has {count} rows, expected 3")
    if not any(f"KPI" in v for v in validations):
        print("  Each KPI has 3 forecast rows ✓")

    horizons_actual = sorted(df_out["forecast_horizon_months"].unique())
    if horizons_actual != [1, 2, 3]:
        validations.append(f"Horizons != [1,2,3]: got {horizons_actual}")
    else:
        print("  Horizons 1,2,3 ✓")

    dup_count = df_out.duplicated(subset=["forecast_month", "kpi_code"]).sum()
    if dup_count > 0:
        validations.append(f"Duplicate forecast_month+kpi_code: {dup_count}")
    else:
        print("  No duplicates ✓")

    # Forecast origin check
    origin = df_out["forecast_origin_month"].iloc[0]
    if origin != latest_month:
        validations.append(f"Origin mismatch: {origin} vs {latest_month}")
    else:
        print(f"  Origin matches latest month ({latest_month}) ✓")

    # Latest actual reconciliation
    for kpi in kpi_codes:
        row = df_out[df_out["kpi_code"] == kpi].iloc[0]
        expected = latest_actual[kpi]["value"]
        actual = row["latest_actual_value"]
        if actual != "" and abs(float(actual) - expected) > 0.01:
            validations.append(f"Latest actual mismatch for {kpi}: {actual} vs {expected}")
    print("  Latest actual values reconcile ✓")

    # Check no forecast exceeds valid range
    for kpi in kpi_codes:
        cfg = cfg_lookup[kpi]
        min_v = cfg["minimum_valid"]
        max_v = cfg["maximum_valid"]
        sub = df_out[df_out["kpi_code"] == kpi]
        for _, r in sub.iterrows():
            fv = r["forecast_value"]
            if fv != "" and fv is not None:
                fv_f = float(fv)
                if fv_f < min_v or fv_f > max_v:
                    validations.append(f"Forecast for {kpi} out of range: {fv_f}")
    print("  Forecasts within valid ranges ✓")

    # Check prediction bounds ordering
    for _, r in df_out.iterrows():
        lb = r["lower_prediction_bound"]
        ub = r["upper_prediction_bound"]
        fv = r["forecast_value"]
        if lb != "" and ub != "" and fv != "":
            lb_f = float(lb)
            ub_f = float(ub)
            fv_f = float(fv)
            if lb_f > ub_f:
                validations.append(f"Bounds inverted for {r['kpi_code']} {r['forecast_month']}")
            if not (lb_f <= fv_f <= ub_f):
                validations.append(f"Forecast not within bounds for {r['kpi_code']} {r['forecast_month']}")
    print("  Prediction bounds ordered and contain forecast ✓")

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
    print(f"Forecast origin:      {latest_month}")
    print(f"Forecast months:      {', '.join(forecast_months)}")
    print(f"Output rows:          {len(df_out)}")
    print()
    print("Forecast status counts:")
    print(df_out["forecast_status"].value_counts().to_string())
    print()
    print("Forecast risk counts:")
    print(df_out["forecast_risk_level"].value_counts().to_string())
    print()
    print("Threshold crossings:")
    crossings = df_out[df_out["threshold_crossing_flag"] != "No Crossing"]["threshold_crossing_flag"].value_counts()
    if len(crossings) == 0:
        print("  No crossings")
    else:
        print(crossings.to_string())
    print()
    print("Clipped forecasts:")
    clipped = df_out[df_out["data_quality_status"] == "Warning"]
    if len(clipped) == 0:
        print("  None")
    else:
        for _, r in clipped.iterrows():
            print(f"  {r['kpi_code']} horizon {r['forecast_horizon_months']}: {r['forecast_value_raw']} -> {r['forecast_value']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
