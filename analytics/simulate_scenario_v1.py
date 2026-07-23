"""
simulate_scenario_v1.py

Sentinel360 Healthcare — Scenario Simulation Engine
Computes live what-if projections for the two approved planning scenarios
(Clinical Staffing Pressure, Facility Capacity Disruption).

Reuses the forecast mathematics imported from
generate_kpi_forecast_output_v1.py — it does NOT read any pre-computed
output CSV. The baseline is a fresh one-month-ahead ensemble forecast,
and scenario parameter effects are applied on top via a transparent
linear coefficient table (IMPACT_COEFFICIENTS).

Inputs (passed in as DataFrames by the caller):
    hospital_kpi_data_v2.csv    — monthly KPI history
    kpi_config_v2.csv           — thresholds, direction, valid ranges
    scenario_baselines_v1.csv   — parameter definitions (loaded by the UI)

Author: Sentinel360 prototype team
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Reuse live forecast math — package import when called from app.py,
# plain import when run directly from the analytics folder.
try:
    from analytics.generate_kpi_forecast_output_v1 import (
        get_recent_window,
        fit_linear_trend,
        calculate_ensemble_forecast,
        clip_to_valid_range,
        get_status_for_value,
    )
except ImportError:
    from generate_kpi_forecast_output_v1 import (
        get_recent_window,
        fit_linear_trend,
        calculate_ensemble_forecast,
        clip_to_valid_range,
        get_status_for_value,
    )


# ──────────────────────────────────────────────────────────
# 1. CONSTANTS
# ──────────────────────────────────────────────────────────

KPI_CODES = ["WF_STAFF_LEVEL", "WF_ABSENCE_RATE", "OP_BED_OCCUPANCY",
             "OP_WAIT_TIME", "PX_COMPLAINT_RATE", "PX_SATISFACTION"]

REFERENCE_MONTH_DAYS = 30.0

# Transparent prototype impact coefficients:
# KPI-unit change per one unit of the scenario parameter, before duration
# scaling. Signs follow each KPI's direction (e.g. a positive coefficient
# on OP_WAIT_TIME means the parameter worsens waiting time).
# Only KPIs listed in scenario_baselines_v1.csv `affected_kpis` appear
# for each parameter. These are illustrative prototype assumptions
# calibrated to the synthetic dataset, not validated hospital elasticities.
IMPACT_COEFFICIENTS = {
    "STAFF_PRESSURE": {
        "STAFF_REDUCTION_PCT": {            # per 1 % staff unavailable
            "WF_STAFF_LEVEL": -1.00,
            "OP_WAIT_TIME": +1.20,
            "PX_COMPLAINT_RATE": +0.030,
            "PX_SATISFACTION": -0.40,
        },
        "ABSENTEEISM_INCREASE_PCT_POINT": {  # per 1 percentage point
            "WF_ABSENCE_RATE": +1.00,
            "WF_STAFF_LEVEL": -1.00,
            "OP_WAIT_TIME": +1.50,
        },
        "PATIENT_DEMAND_INCREASE_PCT": {     # per 1 % extra demand
            "OP_BED_OCCUPANCY": +0.80,
            "OP_WAIT_TIME": +1.00,
            "PX_COMPLAINT_RATE": +0.020,
        },
        "TEMPORARY_STAFF_ADDED": {           # per temporary staff member
            "WF_STAFF_LEVEL": +0.40,
            "OP_WAIT_TIME": -0.80,
            "PX_SATISFACTION": +0.15,
        },
        "EXTENDED_SERVICE_HOURS": {          # per extra service hour/day
            "OP_WAIT_TIME": -3.00,
            "PX_COMPLAINT_RATE": -0.050,
            "PX_SATISFACTION": +0.50,
        },
    },
    "CAPACITY_DISRUPTION": {
        "CAPACITY_REDUCTION_PCT": {          # per 1 % usable capacity lost
            "OP_BED_OCCUPANCY": +0.90,
            "OP_WAIT_TIME": +1.00,
            "PX_COMPLAINT_RATE": +0.030,
            "PX_SATISFACTION": -0.35,
        },
        "PATIENT_DEMAND_INCREASE_PCT": {     # per 1 % extra demand
            "OP_BED_OCCUPANCY": +0.80,
            "OP_WAIT_TIME": +1.00,
            "PX_COMPLAINT_RATE": +0.020,
        },
        "SERVICE_HOURS_LOST": {              # per service hour lost/day
            "OP_WAIT_TIME": +4.00,
            "PX_COMPLAINT_RATE": +0.060,
            "PX_SATISFACTION": -0.60,
        },
        "TEMPORARY_CAPACITY_RESTORED_PCT": { # per 1 % capacity restored
            "OP_BED_OCCUPANCY": -0.70,
            "OP_WAIT_TIME": -0.80,
            "PX_SATISFACTION": +0.25,
        },
        "PATIENTS_REDIRECTED_PCT": {         # per 1 % patients redirected
            "OP_BED_OCCUPANCY": -0.60,
            "OP_WAIT_TIME": -0.70,
            "PX_COMPLAINT_RATE": -0.010,
        },
        "EXTENDED_SERVICE_HOURS": {          # per extra service hour/day
            "OP_WAIT_TIME": -3.00,
            "PX_COMPLAINT_RATE": -0.050,
            "PX_SATISFACTION": +0.50,
        },
    },
}

ANALYTICAL_LIMITATION = (
    "Scenario projections use a transparent linear coefficient model on top of "
    "a live one-month-ahead ensemble forecast. Coefficients are illustrative "
    "prototype assumptions calibrated to the synthetic dataset, not validated "
    "hospital elasticities. Effects are scaled linearly by scenario duration "
    "over a 30-day month and clipped to each KPI's valid range. Projections "
    "support management discussion, not clinical or financial commitment."
)


# ──────────────────────────────────────────────────────────
# 2. CORE CALCULATIONS
# ──────────────────────────────────────────────────────────

def build_config_lookup(df_config: pd.DataFrame) -> dict:
    """Map kpi_code to the threshold/range fields needed for simulation."""
    lookup = {}
    for _, r in df_config.iterrows():
        lookup[r["kpi_code"]] = {
            "kpi_name": r["kpi_name"],
            "domain": r["domain"],
            "unit": r["unit"],
            "direction": r["direction"],
            "warning_threshold": float(r["warning_threshold"]),
            "critical_threshold": float(r["critical_threshold"]),
            "minimum_valid": float(r["minimum_valid"]),
            "maximum_valid": float(r["maximum_valid"]),
        }
    return lookup


def forecast_baseline_value(df_kpi: pd.DataFrame, kpi_code: str,
                            latest_month: str, min_valid: float,
                            max_valid: float) -> float:
    """One-month-ahead ensemble forecast using the live forecast engine math."""
    recent = get_recent_window(df_kpi, kpi_code, latest_month)
    y_vals = recent["kpi_value"].astype(float).values
    slope, _ = fit_linear_trend(y_vals)
    latest_value = float(y_vals[-1])
    recent_avg = float(np.mean(y_vals))
    raw = calculate_ensemble_forecast(latest_value, slope, recent_avg, horizon=1)
    clipped, _ = clip_to_valid_range(raw, min_valid, max_valid)
    return clipped


def calc_scenario_delta(scenario_code: str, kpi_code: str,
                        params: dict, duration_days: float) -> float:
    """Summed coefficient effect on one KPI, scaled by duration share of a month."""
    duration_factor = min(duration_days, REFERENCE_MONTH_DAYS) / REFERENCE_MONTH_DAYS
    delta = 0.0
    for param_code, kpi_effects in IMPACT_COEFFICIENTS[scenario_code].items():
        value = float(params.get(param_code, 0.0))
        delta += kpi_effects.get(kpi_code, 0.0) * value
    return delta * duration_factor


def calc_mitigation_cost(scenario_code: str, params: dict) -> float:
    """Estimated mitigation cost in MYR for the scenario duration."""
    duration_days = float(params.get("SCENARIO_DURATION_DAYS", 0.0))
    if scenario_code == "STAFF_PRESSURE":
        staff = float(params.get("TEMPORARY_STAFF_ADDED", 0.0))
        cost_per_day = float(params.get("TEMP_STAFF_COST_PER_DAY", 0.0))
        return staff * cost_per_day * duration_days
    if scenario_code == "CAPACITY_DISRUPTION":
        restored = float(params.get("TEMPORARY_CAPACITY_RESTORED_PCT", 0.0))
        cost_per_day = float(params.get("TEMP_CAPACITY_COST_PER_DAY", 0.0))
        # ponytail: flat daily cost while any temporary capacity is active;
        # scale by restored share if finer costing is ever needed
        return cost_per_day * duration_days if restored > 0 else 0.0
    return 0.0


def simulate_scenario(df_kpi: pd.DataFrame, df_config: pd.DataFrame,
                      scenario_code: str, params: dict) -> dict:
    """
    Run one scenario. `params` maps parameter_code to the slider value.

    Returns a dict with the origin month, one row per KPI comparing the
    live baseline forecast against the scenario projection, the estimated
    mitigation cost, and the analytical limitation statement.
    """
    if scenario_code not in IMPACT_COEFFICIENTS:
        raise ValueError(f"Unknown scenario_code: {scenario_code}")

    latest_month = df_kpi["reporting_month"].max()
    duration_days = float(params.get("SCENARIO_DURATION_DAYS", REFERENCE_MONTH_DAYS))
    cfg_lookup = build_config_lookup(df_config)

    rows = []
    for kpi_code in KPI_CODES:
        cfg = cfg_lookup[kpi_code]
        baseline = forecast_baseline_value(
            df_kpi, kpi_code, latest_month,
            cfg["minimum_valid"], cfg["maximum_valid"],
        )
        delta = calc_scenario_delta(scenario_code, kpi_code, params, duration_days)
        scenario_value, was_clipped = clip_to_valid_range(
            baseline + delta, cfg["minimum_valid"], cfg["maximum_valid"]
        )
        rows.append({
            "kpi_code": kpi_code,
            "kpi_name": cfg["kpi_name"],
            "domain": cfg["domain"],
            "unit": cfg["unit"],
            "baseline_forecast": baseline,
            "baseline_status": get_status_for_value(
                kpi_code, baseline,
                cfg["warning_threshold"], cfg["critical_threshold"]),
            "scenario_value": scenario_value,
            "scenario_status": get_status_for_value(
                kpi_code, scenario_value,
                cfg["warning_threshold"], cfg["critical_threshold"]),
            "impact_delta": round(scenario_value - baseline, 2),
            "value_clipped_to_valid_range": was_clipped,
        })

    return {
        "scenario_code": scenario_code,
        "forecast_origin_month": latest_month,
        "duration_days": duration_days,
        "rows": rows,
        "estimated_mitigation_cost_myr": calc_mitigation_cost(scenario_code, params),
        "analytical_limitation": ANALYTICAL_LIMITATION,
    }


# ──────────────────────────────────────────────────────────
# 3. SELF-CHECK
# ──────────────────────────────────────────────────────────

def demo() -> None:
    """Runnable self-check against the real project inputs."""
    base_dir = Path(__file__).parent.parent.resolve()
    df_kpi = pd.read_csv(base_dir / "data" / "processed" / "hospital_kpi_data_v2.csv",
                         encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    df_config = pd.read_csv(base_dir / "config" / "kpi_config_v2.csv",
                            encoding="utf-8-sig", keep_default_na=False, na_values=[""])

    # All-zero parameters must reproduce the baseline exactly.
    idle = simulate_scenario(df_kpi, df_config, "STAFF_PRESSURE",
                             {"SCENARIO_DURATION_DAYS": 14})
    for row in idle["rows"]:
        assert row["scenario_value"] == row["baseline_forecast"], row
        assert row["scenario_status"] == row["baseline_status"], row
    assert idle["estimated_mitigation_cost_myr"] == 0.0

    # A staffing shock must reduce staffing level and raise waiting time.
    shock = simulate_scenario(df_kpi, df_config, "STAFF_PRESSURE", {
        "STAFF_REDUCTION_PCT": 15, "ABSENTEEISM_INCREASE_PCT_POINT": 4,
        "PATIENT_DEMAND_INCREASE_PCT": 15, "SCENARIO_DURATION_DAYS": 30,
    })
    by_code = {r["kpi_code"]: r for r in shock["rows"]}
    assert by_code["WF_STAFF_LEVEL"]["impact_delta"] < 0
    assert by_code["OP_WAIT_TIME"]["impact_delta"] > 0

    # Mitigation must cost money and improve waiting time vs no intervention.
    mitigated = simulate_scenario(df_kpi, df_config, "STAFF_PRESSURE", {
        "STAFF_REDUCTION_PCT": 15, "ABSENTEEISM_INCREASE_PCT_POINT": 4,
        "PATIENT_DEMAND_INCREASE_PCT": 15, "TEMPORARY_STAFF_ADDED": 8,
        "EXTENDED_SERVICE_HOURS": 2, "SCENARIO_DURATION_DAYS": 30,
        "TEMP_STAFF_COST_PER_DAY": 900,
    })
    mit_by_code = {r["kpi_code"]: r for r in mitigated["rows"]}
    assert mit_by_code["OP_WAIT_TIME"]["scenario_value"] < by_code["OP_WAIT_TIME"]["scenario_value"]
    assert mitigated["estimated_mitigation_cost_myr"] == 8 * 900 * 30

    # Capacity scenario: occupancy rises under capacity loss + demand growth.
    cap = simulate_scenario(df_kpi, df_config, "CAPACITY_DISRUPTION", {
        "CAPACITY_REDUCTION_PCT": 20, "PATIENT_DEMAND_INCREASE_PCT": 10,
        "SERVICE_HOURS_LOST": 4, "SCENARIO_DURATION_DAYS": 14,
    })
    cap_by_code = {r["kpi_code"]: r for r in cap["rows"]}
    assert cap_by_code["OP_BED_OCCUPANCY"]["impact_delta"] > 0

    # All values must respect each KPI's valid range.
    cfg_lookup = build_config_lookup(df_config)
    for result in (idle, shock, mitigated, cap):
        for row in result["rows"]:
            cfg = cfg_lookup[row["kpi_code"]]
            assert cfg["minimum_valid"] <= row["scenario_value"] <= cfg["maximum_valid"], row

    print("simulate_scenario_v1 self-check passed.")
    print(f"Forecast origin month: {idle['forecast_origin_month']}")
    for row in shock["rows"]:
        print(f"  {row['kpi_code']}: baseline {row['baseline_forecast']} "
              f"({row['baseline_status']}) -> scenario {row['scenario_value']} "
              f"({row['scenario_status']})")


if __name__ == "__main__":
    demo()
