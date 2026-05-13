# Prediction-calibration-framework-using-Python-and-environmental-dataset
# Automated Prediction & Calibration Framework
## Next-Day Maximum Temperature Prediction for Bengaluru, India

This project presents an end-to-end machine learning pipeline for predicting the **next-day maximum temperature** in Bengaluru, Karnataka, using 1000 days of historical weather data.

The framework follows a structured workflow:

**Collect Data → Engineer Features → Train Model → Calibrate Hyperparameters → Evaluate Performance → Generate Visual Reports**

The project demonstrates how predictive modeling and calibration techniques can be applied to environmental datasets in a fully automated and reproducible way.

---

## Project Highlights

- 1000-day Bengaluru weather dataset (May 2023 – February 2026)
- 19 engineered features including lag variables and rolling statistics
- Baseline model: Random Forest Regressor
- Hyperparameter calibration using GridSearchCV
- Automated visual report generation
- BBMP ward-level temperature heat map
- Export-ready PDF report

---

## Objective

To build a robust machine learning pipeline that predicts tomorrow’s maximum temperature using current and historical weather variables such as:

- Maximum temperature
- Minimum temperature
- Mean temperature
- Relative humidity
- Precipitation
- Wind speed

---

## Dataset Information

| Parameter | Value |
|----------|-------|
| Location | Bengaluru, Karnataka, India |
| Coordinates | 12.97°N, 77.59°E |
| Time Period | May 13, 2023 – Feb 5, 2026 |
| Number of Days | 1000 |
| Data Source | Open-Meteo Archive API |
| Target Variable | Next-day maximum temperature (°C) |

---

## Feature Engineering

The model uses 19 predictive features.

### Raw Features
- `temp_max_C`
- `temp_min_C`
- `temp_mean_C`
- `humidity_pct`
- `precip_mm`
- `wind_kmh`

### Lag Features
- Previous day values (`lag1`)
- Two-day lag for mean temperature (`lag2`)

### Rolling Features
- 7-day rolling mean of temperature
- 7-day rolling standard deviation
- 7-day cumulative precipitation

### Calendar Features
- Month
- Day of year
- Monsoon indicator (June–September)

---

## Model Architecture

### Baseline Model
`RandomForestRegressor` from scikit-learn

### Calibration
`GridSearchCV` with 5-fold cross-validation.

### Hyperparameter Search Space

```python
{
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"]
}
```

Total combinations tested: **24**

---

## Model Performance

| Metric | Baseline RF | Tuned RF |
|------:|------------:|---------:|
| RMSE (°C) | 1.4876 | 1.4923 |
| R² Score | 0.6658 | 0.6637 |
| CV Folds | — | 5 |
| Parameter Combinations | — | 24 |

### Interpretation

- Average prediction error is approximately **1.49°C**
- Model explains approximately **66%** of variance in next-day maximum temperature
- Hyperparameter tuning provided similar performance to the default model, indicating the baseline model was already near-optimal

---

## Key Findings

1. **7-day rolling mean temperature** is the most important predictor.
2. Recent temperature observations dominate forecasting accuracy.
3. Precipitation and monsoon indicators have lower importance.
4. Bengaluru exhibits strong temperature persistence, making short-term forecasting effective.

---

## Feature Importance (Top Predictors)

| Rank | Feature |
|----:|---------|
| 1 | `temp_roll7_mean` |
| 2 | `temp_mean_C` |
| 3 | `temp_mean_C_lag2` |
| 4 | `temp_mean_C_lag1` |
| 5 | `day_of_year` |

---

## Visual Outputs Generated

The pipeline automatically creates:

- `01_eda.png` — Exploratory Data Analysis
- `02_seasonal.png` — Seasonal Climate Patterns
- `03_prediction.png` — Actual vs Predicted Results
- `04_feature_importance.png` — Feature Importance
- `05_bbmp_heatmap.png` — BBMP Ward-Level Heat Map
- `06_results_table.png` — Performance Summary Table
- `Bengaluru_Temperature_Prediction_Report_Final.pdf`

---

## BBMP Heat Map

The project includes a ward-level temperature heat map using Bengaluru’s 243 BBMP wards.

Since the input data is a single weather station time series, a synthetic spatial gradient is used to approximate intra-city temperature variation based on:
- Latitude
- Longitude
- Random noise

This visualization is intended for demonstration and presentation purposes.

---

## Repository Structure

```text
Prediction-calibration-framework-using-Python-and-environmental-dataset/
│
├── bengaluru_temp_prediction.py
├── project1_bengaluru_weather_1000days.csv
├── BBMP.shp
├── 01_eda.png
├── 02_seasonal.png
├── 03_prediction.png
├── 04_feature_importance.png
├── 05_bbmp_heatmap.png
├── 06_results_table.png
├── Bengaluru_Temperature_Prediction_Report_Final.pdf
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/OmkarSarode2005/Prediction-calibration-framework-using-Python-and-environmental-dataset.git
cd Prediction-calibration-framework-using-Python-and-environmental-dataset
```

### Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Run the Project

```bash
python bengaluru_temp_prediction.py
```

---

## Technologies Used

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn

---

## Future Improvements

- Integrate Numerical Weather Prediction (NWP) outputs
- Test LSTM and Transformer models
- Build a FastAPI REST API
- Containerize with Docker
- Extend to multi-output forecasting (temperature, humidity, rainfall)

---

## Applications

- Urban climate analytics
- Smart city planning
- Agricultural decision support
- Energy demand forecasting
- Environmental monitoring

---

## Author

**Omkar Sarode**

Cyber Physical Systems Student, MIT Manipal

GitHub: https://github.com/OmkarSarode2005

---

## Repository Link

https://github.com/OmkarSarode2005/Prediction-calibration-framework-using-Python-and-environmental-dataset

-
