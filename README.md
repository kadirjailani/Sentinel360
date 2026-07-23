# Sentinel360 Healthcare

Executive early-warning prototype for hospital KPI monitoring. It turns raw operational records (rosters, attendance, patient queues, surveys, complaints, bed occupancy) into a monthly KPI dataset, then runs a four-stage analytics pipeline that detects warnings, connects signals across domains, forecasts the next 1–3 months, and ranks the KPIs by executive priority. A Streamlit dashboard presents the results.

> **Prototype notice:** All data is synthetic (single hospital, `KPJ_01` / JCORP Specialist Hospital, 24 monthly periods: July 2024 – June 2026). Thresholds, weights, and scoring rules are transparent prototype assumptions calibrated to the synthetic dataset — not validated hospital standards. Outputs support management decisions; they are not for clinical use.

## KPIs monitored

Six headline KPIs across three domains, defined in `config/kpi_config_v2.csv` (formulas, thresholds, direction, valid ranges, assumptions, and limitations per KPI):

| Domain | KPI code | KPI name | Direction |
|---|---|---|---|
| Workforce | `WF_STAFF_LEVEL` | Staffing Level | higher is better |
| Workforce | `WF_ABSENCE_RATE` | Staff Absenteeism Rate | lower is better |
| Operations | `OP_BED_OCCUPANCY` | Bed Occupancy Rate | range is better |
| Operations | `OP_WAIT_TIME` | Average Patient Waiting Time | lower is better |
| Patient Experience | `PX_COMPLAINT_RATE` | Patient Complaint Rate | lower is better |
| Patient Experience | `PX_SATISFACTION` | Patient Satisfaction Score | higher is better |

## Project structure

```
Sentinel360/
├── app.py                                      # Streamlit dashboard (reads outputs/ only, no recalculation)
├── analytics/
│   ├── generate_kpi_warning_output_v1.py       # Stage 1 — warnings, streaks, anomalies, alert priority
│   ├── generate_cross_domain_signal_output_v1.py  # Stage 2 — domain scores, signal patterns, risk score
│   ├── generate_kpi_forecast_output_v1.py      # Stage 3 — 1–3 month transparent forecast
│   └── generate_executive_risk_ranking_v1.py   # Stage 4 — executive priority ranking of the 6 KPIs
├── config/
│   ├── kpi_config_v2.csv                       # KPI definitions, thresholds, formulas, limitations
│   └── scenario_baselines_v1.csv               # Scenario parameters (Staffing Pressure, Capacity Disruption)
├── data/processed/
│   ├── hospital_kpi_data_v2.csv                # Monthly KPI dataset (6 KPIs × 24 months) — pipeline input
│   ├── staff_roster_v1.csv                     # Planned shifts + required headcount (~135k rows)
│   ├── staff_attendance_v1.csv                 # Actual attendance vs roster (~135k rows)
│   ├── staff_master_v1.csv                     # Staff directory
│   ├── patient_queue_records_v1.csv            # Per-visit waiting times (~298k rows)
│   ├── patient_encounters_v1.csv               # Daily encounter volumes by service area
│   ├── patient_survey_v1.csv                   # Survey responses with scores (~61k rows)
│   ├── patient_complaints_v1.csv               # Complaints with theme/sentiment/urgency
│   └── bed_occupancy_records_v1.csv            # Daily ward-level bed occupancy
├── outputs/                                    # Pipeline results (consumed by app.py)
│   ├── kpi_warning_output_v1.csv
│   ├── cross_domain_signal_output_v1.csv
│   ├── kpi_forecast_output_v1.csv
│   └── executive_risk_ranking_v1.csv
└── validation/
    └── ground_truth_v2.xlsx                    # Ground truth for validating pipeline outputs
```

## Analytics pipeline

Each stage reads the outputs of earlier stages, so **run order matters**:

```
hospital_kpi_data_v2.csv + kpi_config_v2.csv
        │
        ▼
[1] kpi_warning ──► [2] cross_domain ──► [3] forecast ──► [4] risk_ranking ──► app.py
```

### Stage 1 — KPI Warning & Anomaly Detection
`analytics/generate_kpi_warning_output_v1.py`

For every KPI-month: threshold breach classification, consecutive warning/critical streaks, deterioration streaks, rolling six-month z-score anomaly detection, and a combined alert priority. Emits a plain-language alert title, explanation, and recommended management attention per row.

### Stage 2 — Cross-Domain Signal Detection
`analytics/generate_cross_domain_signal_output_v1.py`

Scores each domain (Normal = 0, Warning = 1, Critical = 2 points per KPI), counts connected domains, classifies the month into a signal pattern (e.g. `PATTERN_STABLE`, `PATTERN_ALL_DOMAINS`), and produces a cross-domain risk score with management interpretation. Concurrent movement is flagged as association, not causation.

### Stage 3 — Short-Term KPI Forecast
`analytics/generate_kpi_forecast_output_v1.py`

Transparent 1–3 month forecast per KPI: an ensemble of 70% damped linear trend (damping factor 0.7 over a 6-month recent window) + 30% recent average, with approximate 95% prediction intervals. Classifies each forecast month against warning/critical boundaries and flags threshold crossings.

### Stage 4 — Executive Risk Ranking
`analytics/generate_executive_risk_ranking_v1.py`

Combines all prior outputs into a single ranked list of the six KPIs. The executive priority score sums transparent components: current severity, persistence, deterioration, anomaly, forecast exposure, cross-domain involvement, and a data-quality penalty. Each row carries an executive summary, recommended action, and suggested scenario to run.

Every output row includes its evidence sources, data-quality status, and an explicit `analytical_limitation` statement.

## Dashboard

`app.py` is a read-only Streamlit app over `outputs/` — it never recalculates results. Five pages:

- **Executive Overview** — implemented: latest-month status, ranking, and signals
- **Warning Evidence**, **Forecast**, **Scenario Lab**, **Data & Validation** — placeholders, content coming soon

The Scenario Lab will use `config/scenario_baselines_v1.csv`, which defines two what-if scenarios: **Clinical Staffing Pressure** (`STAFF_PRESSURE`) and **Facility Capacity Disruption** (`CAPACITY_DISRUPTION`).

## Getting started

Requires Python 3.10+ with `pandas`, `numpy`, and `streamlit`:

```bash
pip install pandas numpy streamlit
```

Regenerate outputs (only needed if inputs change — the repo ships with pre-computed outputs), in order:

```bash
python analytics/generate_kpi_warning_output_v1.py
python analytics/generate_cross_domain_signal_output_v1.py
python analytics/generate_kpi_forecast_output_v1.py
python analytics/generate_executive_risk_ranking_v1.py
```

Launch the dashboard from the project root:

```bash
streamlit run app.py
```
