# Power Anomaly Detection

A Python-based anomaly detection pipeline for identifying irregular 
electricity consumption patterns in time-series household power data.

---

## Project Structure
power-anomaly-detection/
├── data/
│   └── household_power_consumption.txt
├── src/
│   ├── load_data.py           # Data ingestion and preprocessing
│   ├── inject_attacks.py   # Synthetic anomaly injection
|   ├── time_segmentation.py   # Add time and day categories
│   ├── anomaly_detection.py   # Rolling window + same-hour detection
│   ├── correlation_checks.py  # Cross-variable correlation anomalies
│   ├── anomaly_summaries.py # JSON summary generation
│   └── visualisation.py       # Plotting and visual output
| 
├── notebooks/
│   └── exploration.ipynb      # Exploratory data analysis
├── main.py                    # Pipeline entry point
├── README.md
└── requirements.txt

---

## Dataset
- **Source:** UCI Machine Learning Repository — Individual Household 
  Electric Power Consumption https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set/data
- **Coverage:** December 2006 – November 2010
- **Granularity:** 1-minute intervals (~2 million readings)
- **Features:** Global active power, voltage, current, 
  and 3 sub-metering channels

---

## Detection Methods

### Time-Aware Anomaly Detection
- **Rolling Window Z-Score** — Compares each reading to its local 
  neighbourhood to catch sudden spikes
- **Same-Hour Comparison** — Compares each reading to the same hour 
  on the same day type (weekday/weekend) to account for daily patterns

### Correlation Checks
- **Ratio Tracking** — Monitors the unmetered power remainder 
  (total power minus sub meter sum) for unexpected deviations
- **Rolling Correlation** — Tracks correlation between global intensity 
  and active power, flagging divergence between physically linked variables

### Simulated Attacks
- Random spikes
- Gradual drift
- Sensor flatline
- Noise injection

---

## How to Run

### 1. Clone the repository
git clone https://github.com/ethanharveyuq/power-anomaly-detection.git
cd power-anomaly-detection

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add the dataset
Place household_power.csv in the data/ folder.

### 4. Run the pipeline
python main.py

Output is saved to output.json.

---

## Requirements
- Python 3.8+
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib

---

## Key Design Decisions
- Anomalies are only flagged when multiple detection methods agree, 
  reducing false positives
- Weekday and weekend periods are treated separately since consumption 
  patterns differ significantly
- All simulated attacks are applied to a copy of the data, preserving 
  the original for comparison

---

## Possible Extensions
- Real-time streaming anomaly detection
- Additional sub-meter correlation checks
- Dashboard visualisation with Plotly or Streamlit

---

## Author
Ethan Harvey