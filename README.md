# PMU Source Identification
A Python pipeline for preprocessing and classifying synchrophasor (PMU) measurement data from the Victorian power network, using LLM-based and sequence-modelling approaches for source identification.

---

## Project Overview

Modern power systems generate large volumes of heterogeneous measurement data from Phasor Measurement Units (PMUs). Ensuring the authenticity and integrity of these measurements is critical for reliable grid operation and defence against data-driven cyber attacks.

This project explores the use of large language models (LLMs) and related foundation models for **PMU source identification** — given a window of synchrophasor measurements, predict which of ~50 PMUs it originated from. This is a 50-class fingerprinting problem operating on high-frequency time-series data (50 samples/second) sourced from Victorian distribution network PMUs operated by AusNet Services.

---

## Project Structure

```
power-anomaly-detection/
├── data/                          # Raw PMU data files (one per unit)
├── src/
│   ├── load_data.py               # Data ingestion and column renaming
│   └── process_data.py            # Preprocessing pipeline
│
├── notebooks/
│   └── exploration.ipynb          # Exploratory data analysis
├── main.py                        # Pipeline entry point
├── README.md
└── requirements.txt
```

---

## Dataset

- **Source:** AusNet Services Victorian distribution network
- **Coverage:** 3-hour window (October 2021)
- **Units:** ~50 PMUs
- **Granularity:** 20ms intervals (50 Hz)
- **Format:** One CSV/Excel file per PMU, columns include timestamp, frequency, GPS flag, voltage phasors (A/B/C magnitude and angle), and current phasors (A/B/C/N magnitude and angle)

### Key Columns

| Column | Description |
|---|---|
| Timestamp | Sample time (`DD/MM/YYYY HH:MM:SS.ffffff`) |
| FREQ | Fundamental frequency (Hz) |
| FLAG | GPS lock status — bitwise field per IEC C37.118; 64/128 = good, ≥1000 = bad lock |
| UA/UB/UC MAG+ANG | Phase voltage magnitude and angle (A, B, C) |
| UR/US/UT MAG+ANG | Secondary voltage phasors |
| IA/IB/IC/IN MAG+ANG | Phase and neutral current magnitude and angle |

---

## Preprocessing Pipeline

All preprocessing is handled in `src/process_data.py`. The pipeline runs per-PMU in the following order:

### 1. Duplicate Removal
Duplicate timestamps are dropped, keeping the first occurrence. This must occur before reindexing as pandas cannot reindex over a non-unique index.

### 2. Reindex to Uniform Grid
The dataframe is reindexed to a strict 20ms `DatetimeIndex` using `pd.date_range`. Rows absent from the original data appear as NaN after this step.

### 3. Interpolate Short Gaps
Contiguous runs of missing rows up to `MAX_INTERP_GAP` samples are filled using linear interpolation. Runs exceeding this threshold are left as NaN and dropped downstream.

### 4. GPS Loss Detection and Flagging
The FLAG column is scanned for contiguous runs of bad GPS lock values (FLAG ≥ `MAX_FLAG`). Runs exceeding `MAX_INTERP_GAP` are flagged as NaN across all data columns. Short runs are linearly interpolated using surrounding good-lock values.

> **Why GPS loss matters:** When a PMU loses GPS lock, it free-runs on its internal clock. Timestamps during this window can drift and are not trustworthy for time-series alignment. Interpolating across these windows would fabricate data at timestamps that may not correspond to reality.

### 5. Drop Unfillable Rows
Rows still containing NaN across data columns after steps 3–4 are dropped via `dropna()`.

### 6. Segment Assignment
A `segment_id` column is assigned that increments at every discontinuity (gap left by GPS loss or unfilled missing rows). This ensures sliding windows used for model training never straddle a real-world time gap.

### Key Constants

| Constant | Value | Description |
|---|---|---|
| `SAMPLE_PERIOD_MS` | 20 | Sample interval in milliseconds |
| `MAX_INTERP_GAP` | 10 | Max consecutive samples to interpolate |
| `MAX_FLAG` | 1000 | Threshold above which FLAG indicates bad GPS lock |
| `GOOD_FLAG_VALUES` | {64, 128} | Confirmed good GPS lock values |

> **Note:** `MAX_FLAG` is empirically derived from the observed FLAG distribution (64, 128, 270784, 270800, 270816). The bad values are likely a bitmask per IEC C37.118 PMU status word conventions — this has not been verified against the device specification and is noted as a known limitation.

---

## Feature Stages

Features are added incrementally as model experiments progress:

| Stage | Features |
|---|---|
| Stage 1 | `FREQ` only |
| Stage 2 | `FREQ` + voltage phasors (UA/UB/UC MAG+ANG) |
| Stage 3 | `FREQ` + voltage + current phasors (IA/IB/IC/IN MAG+ANG) |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ethanharveyuq/power-anomaly-detection.git
cd power-anomaly-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add data files
Place PMU data files in the `data/` folder.

### 4. Run the pipeline
```bash
python main.py
```

---

## Requirements

- Python 3.8+
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- openpyxl

---

## Key Design Decisions

- **GPS loss is excluded, not interpolated over** — timestamps during GPS loss windows are untrustworthy and fabricating values at those positions would corrupt any time-aligned model
- **Segment IDs prevent cross-gap training windows** — any sequence model operating on sliding windows must respect segment boundaries to avoid treating discontinuous data as continuous signal
- **Incremental feature addition** — starting from frequency only and adding voltage/current progressively allows controlled attribution of classification performance improvements to specific signal types
- **Per-PMU preprocessing** — each unit is cleaned independently before any cross-unit alignment, preventing timestamp mismatches from propagating across the dataset

---

## Possible Extensions

- Sliding window feature extraction and windowed dataset construction
- Transformer-based sequence classifier (PatchTST, iTransformer)
- LLM backbone fine-tuning for time-series (GPT4TS / One Fits All approach)
- Cross-PMU alignment for spatial feature extraction
- Evaluation against spoofed/injected data for FDIA detection

---

## Author

Ethan Harvey — Undergraduate Research Project, University of Queensland
