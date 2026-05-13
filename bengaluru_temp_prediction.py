"""
Automated Prediction & Calibration Framework
Next-Day Maximum Temperature Prediction — Bengaluru, Karnataka
Dataset: May 13 2023 – Feb 5 2026 (1000 days)
Model: Random Forest + GridSearchCV
"""

import struct
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
CSV_PATH  = "project1_bengaluru_weather_1000days.csv"
SHP_PATH  = "BBMP.shp"


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_eda(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Bengaluru Weather — Exploratory Analysis (1000 Days)", fontsize=14, fontweight="bold")

    # temperature over time
    ax = axes[0, 0]
    ax.fill_between(df["date"], df["temp_min_C"], df["temp_max_C"], alpha=0.3, color="salmon", label="Min-Max range")
    ax.plot(df["date"], df["temp_mean_C"], color="firebrick", lw=0.8, label="Mean temp")
    ax.set_title("Temperature Over Time")
    ax.set_ylabel("°C")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # daily precipitation
    ax = axes[0, 1]
    ax.bar(df["date"], df["precip_mm"], color="steelblue", width=1, alpha=0.7)
    ax.set_title("Daily Precipitation (Monsoon visible Jun–Sep)")
    ax.set_ylabel("mm")
    ax.grid(True, alpha=0.3)

    # monthly max temp box plot
    ax = axes[1, 0]
    df["month_name"] = df["date"].dt.strftime("%b")
    df["month_num"]  = df["date"].dt.month
    monthly_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    grouped = [df[df["month_name"] == m]["temp_max_C"].dropna().values for m in monthly_order]
    ax.boxplot(grouped, labels=monthly_order, patch_artist=True,
               boxprops=dict(facecolor="mistyrose", color="firebrick"),
               medianprops=dict(color="firebrick", lw=2))
    ax.set_title("Monthly Max Temperature Distribution")
    ax.set_ylabel("Max Temp (°C)")
    ax.grid(True, alpha=0.3, axis="y")

    # correlation heatmap
    ax = axes[1, 1]
    cols = ["temp_max_C","temp_min_C","temp_mean_C","humidity_pct","precip_mm","wind_kmh"]
    labels = ["Tmax","Tmin","Tmean","Hum","Precip","Wind"]
    corr = df[cols].corr().values
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i,j]) > 0.6 else "black")
    ax.set_title("Feature Correlation Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("01_eda.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[1/6] EDA plot saved → 01_eda.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEASONAL CLIMATE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def plot_seasonal(df):
    df["month_num"] = df["date"].dt.month
    monthly = df.groupby("month_num").agg(
        tmax=("temp_max_C","mean"),
        tmin=("temp_min_C","mean"),
        precip=("precip_mm","sum"),
        humidity=("humidity_pct","mean")
    ).reset_index()
    # normalise precip to ~monthly (dataset spans ~2.75 years)
    monthly["precip"] = monthly["precip"] / 2.75

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    x = range(1, 13)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Bengaluru Seasonal Climate Patterns (1000-Day Dataset)", fontsize=13, fontweight="bold")

    # temperature
    ax1.fill_between(monthly["month_num"], monthly["tmin"], monthly["tmax"], alpha=0.25, color="salmon", label="Min-Max range")
    ax1.plot(monthly["month_num"], monthly["tmax"], "o-", color="firebrick", lw=2, label="Max Temp")
    ax1.plot(monthly["month_num"], monthly["tmin"], "s-", color="steelblue", lw=2, label="Min Temp")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(months, fontsize=8)
    ax1.set_ylabel("Temperature (°C)"); ax1.set_title("Monthly Temperature Range")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # rainfall & humidity
    ax2b = ax2.twinx()
    ax2.bar(monthly["month_num"], monthly["precip"], color="steelblue", alpha=0.7, label="Est. Monthly Rainfall (mm)")
    ax2b.plot(monthly["month_num"], monthly["humidity"], "o-", color="darkorange", lw=2, label="Humidity %")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(months, fontsize=8)
    ax2.set_ylabel("Est. Monthly Rainfall (mm)"); ax2b.set_ylabel("Humidity (%)")
    ax2.set_title("Monthly Rainfall & Humidity")
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("02_seasonal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[2/6] Seasonal plot saved → 02_seasonal.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df):
    d = df.copy()

    # lag features (t-1, t-2)
    for col in ["temp_max_C","temp_min_C","temp_mean_C","humidity_pct","precip_mm","wind_kmh"]:
        d[f"{col}_lag1"] = d[col].shift(1)

    d["temp_mean_C_lag2"] = d["temp_mean_C"].shift(2)
    d["precip_mm_lag1"]   = d["precip_mm"].shift(1)

    # rolling stats (7-day window)
    d["temp_roll7_mean"] = d["temp_mean_C"].rolling(7).mean()
    d["temp_roll7_std"]  = d["temp_mean_C"].rolling(7).std()
    d["precip_roll7"]    = d["precip_mm"].rolling(7).sum()

    # calendar features
    d["month"]      = d["date"].dt.month
    d["day_of_year"] = d["date"].dt.dayofyear
    d["is_monsoon"] = d["month"].isin([6, 7, 8, 9]).astype(int)

    # target: next-day maximum temperature
    d["target"] = d["temp_max_C"].shift(-1)

    d = d.dropna().reset_index(drop=True)
    return d


FEATURES = [
    "temp_max_C","temp_min_C","temp_mean_C","humidity_pct","precip_mm","wind_kmh",
    "temp_max_C_lag1","temp_min_C_lag1","temp_mean_C_lag1","humidity_pct_lag1",
    "precip_mm_lag1","wind_kmh_lag1","temp_mean_C_lag2",
    "temp_roll7_mean","temp_roll7_std","precip_roll7",
    "month","day_of_year","is_monsoon"
]


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT (chronological 80/20)
# ─────────────────────────────────────────────────────────────────────────────

def split_data(df):
    split = int(len(df) * 0.80)
    train = df.iloc[:split]
    test  = df.iloc[split:]
    X_train = train[FEATURES];  y_train = train["target"]
    X_test  = test[FEATURES];   y_test  = test["target"]
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 6. BASELINE RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline(X_train, y_train):
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


# ─────────────────────────────────────────────────────────────────────────────
# 7. CALIBRATION — GridSearchCV
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(X_train, y_train):
    param_grid = {
        "n_estimators":     [100, 200],
        "max_depth":        [None, 10, 20],
        "min_samples_split":[2, 5],
        "max_features":     ["sqrt", "log2"]
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0
    )
    gs.fit(X_train, y_train)
    print(f"    Best params : {gs.best_params_}")
    print(f"    CV RMSE     : {-gs.best_score_:.4f} °C")
    return gs.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# 8. EVALUATION & PREDICTION PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(rf_base, rf_tuned, X_train, y_train, X_test, y_test):
    # metrics
    pred_base  = rf_base.predict(X_test)
    pred_tuned = rf_tuned.predict(X_test)

    rmse_base  = np.sqrt(mean_squared_error(y_test, pred_base))
    rmse_tuned = np.sqrt(mean_squared_error(y_test, pred_tuned))
    r2_base    = r2_score(y_test, pred_base)
    r2_tuned   = r2_score(y_test, pred_tuned)

    print("\n    ── Results ──────────────────────────────────")
    print(f"    Baseline RF  → RMSE: {rmse_base:.4f} °C  |  R²: {r2_base:.4f}")
    print(f"    Tuned RF     → RMSE: {rmse_tuned:.4f} °C  |  R²: {r2_tuned:.4f}")
    print(f"    Improvement  → RMSE: ↓{rmse_base-rmse_tuned:.4f}  |  R²: ↑{r2_tuned-r2_base:.4f}")

    # time-series + scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Bengaluru Next-Day Max Temperature Prediction\n(Random Forest · GridSearchCV Calibration)",
                 fontsize=12, fontweight="bold")

    idx = range(len(y_test))
    ax1.plot(idx, y_test.values,    color="black",     lw=0.9, label="Actual")
    ax1.plot(idx, pred_base,        color="tomato",    lw=1.0, alpha=0.7, label="Baseline RF")
    ax1.plot(idx, pred_tuned,       color="seagreen",  lw=1.0, alpha=0.8, label="Tuned RF (GridCV)")
    ax1.set_xlabel("Day Index"); ax1.set_ylabel("Max Temperature (°C)")
    ax1.set_title(f"Actual vs Predicted (Test Set — {len(y_test)} days)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    lo = min(y_test.min(), pred_tuned.min()) - 0.5
    hi = max(y_test.max(), pred_tuned.max()) + 0.5
    ax2.scatter(y_test, pred_tuned, color="steelblue", alpha=0.5, s=20, label="Predictions")
    ax2.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit (y=x)")
    ax2.set_xlabel("Actual Max Temperature (°C)"); ax2.set_ylabel("Predicted Max Temperature (°C)")
    ax2.set_title(f"Scatter: Actual vs Predicted  R²={r2_tuned:.3f}")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("03_prediction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[3/6] Prediction plot saved → 03_prediction.png")

    return pred_tuned, rmse_tuned, r2_tuned


# ─────────────────────────────────────────────────────────────────────────────
# 9. FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(rf_tuned):
    fi = pd.Series(rf_tuned.feature_importances_, index=FEATURES).sort_values(ascending=True)

    colors = ["#d3d3d3" if v < 0.04 else ("#5ba4cf" if v < 0.08 else "#2ecc71") for v in fi.values]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(fi.index, fi.values, color=colors, edgecolor="none")
    for bar, val in zip(bars, fi.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label=">8% importance"),
        mpatches.Patch(color="#5ba4cf", label="4–8% importance"),
        mpatches.Patch(color="#d3d3d3", label="<4% importance"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    ax.set_xlabel("Importance Score"); ax.set_xlim(0, fi.max() * 1.15)
    ax.set_title("Feature Importance — Tuned Random Forest\n(Bengaluru Temperature Prediction)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("04_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[4/6] Feature importance plot saved → 04_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. BBMP WARD-LEVEL HEAT MAP
#     The shapefile covers Bengaluru's 243 BBMP wards.  Since the input data
#     is a single-station time series (one lat/lon), we simulate plausible
#     intra-city temperature variation using a spatial gradient (south-west
#     wards are typically warmer; north-east is cooler and higher elevation).
#     The gradient seed is the dataset's climatological mean max temperature.
# ─────────────────────────────────────────────────────────────────────────────

def read_shapefile_polygons(path):
    """Parse ESRI Shapefile (type 5 – Polygon) without external geo libraries."""
    polygons = []
    centroids = []
    with open(path, "rb") as f:
        f.seek(100)  # skip file header
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            content_len = struct.unpack(">i", hdr[4:])[0]
            content = f.read(content_len * 2)
            if len(content) < 4:
                break
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type != 5:
                continue

            num_parts, num_points = struct.unpack("<ii", content[36:44])
            parts = [struct.unpack("<i", content[44+i*4:48+i*4])[0] for i in range(num_parts)]
            pts_start = 44 + num_parts * 4
            points = [
                struct.unpack("<dd", content[pts_start+i*16:pts_start+i*16+16])
                for i in range(num_points)
            ]

            rings = []
            for i, p in enumerate(parts):
                end = parts[i+1] if i+1 < num_parts else num_points
                rings.append(np.array(points[p:end]))

            polygons.append(rings)
            # centroid from the outer ring
            outer = rings[0]
            centroids.append((outer[:, 0].mean(), outer[:, 1].mean()))

    return polygons, np.array(centroids)


def plot_bbmp_heatmap(df, polygons, centroids):
    mean_tmax = df["temp_max_C"].mean()    # dataset climatological mean

    # spatial temperature proxy:
    # southern wards (lower lat) tend to be slightly warmer; western wards also warmer
    lat_norm = (centroids[:, 1] - centroids[:, 1].min()) / (np.ptp(centroids[:, 1]) + 1e-9)
    lon_norm = (centroids[:, 0] - centroids[:, 0].min()) / (np.ptp(centroids[:, 0]) + 1e-9)

    # temperature field: warmer in SW, cooler in NE; ±2°C spread around mean
    np.random.seed(42)
    noise = np.random.normal(0, 0.3, len(centroids))
    ward_temps = mean_tmax - 1.5 * lat_norm - 0.8 * lon_norm + 2.0 + noise

    cmap = cm.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=ward_temps.min(), vmax=ward_temps.max())

    fig, ax = plt.subplots(figsize=(10, 10))

    patches = []
    face_colors = []
    for rings, temp in zip(polygons, ward_temps):
        outer = rings[0]  # only outer boundary for fill
        poly = MplPolygon(outer, closed=True)
        patches.append(poly)
        face_colors.append(cmap(norm(temp)))

    pc = PatchCollection(patches, facecolor=face_colors, edgecolor="#555555", linewidth=0.3)
    ax.add_collection(pc)

    # colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Est. Mean Max Temperature (°C)", fontsize=10)

    ax.set_xlim(centroids[:, 0].min() - 0.02, centroids[:, 0].max() + 0.02)
    ax.set_ylim(centroids[:, 1].min() - 0.02, centroids[:, 1].max() + 0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title(
        "Bengaluru BBMP Ward-Level Temperature Heat Map\n"
        "(Spatial gradient derived from 1000-day climatology · 243 wards)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("05_bbmp_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[5/6] BBMP heat map saved → 05_bbmp_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. RESULTS SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_table(rf_base, rf_tuned, X_test, y_test, rmse_tuned, r2_tuned):
    pred_base  = rf_base.predict(X_test)
    rmse_base  = np.sqrt(mean_squared_error(y_test, pred_base))
    r2_base    = r2_score(y_test, pred_base)

    rows = [
        ["RMSE (°C)",          f"{rmse_base:.4f}",  f"{rmse_tuned:.4f}", f"↓ {rmse_base-rmse_tuned:.4f}"],
        ["R² Score",           f"{r2_base:.4f}",    f"{r2_tuned:.4f}",   f"↑ {r2_tuned-r2_base:.4f}"],
        ["CV Folds",           "—",                 "5",                 "—"],
        ["Param combos tested","—",                 "24",                "—"],
    ]
    cols = ["Metric", "Baseline RF (default)", "Tuned RF (GridSearchCV)", "Improvement"]

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows)+1):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(bg)

    ax.set_title("Model Results Summary", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig("06_results_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[6/6] Results table saved → 06_results_table.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n── Bengaluru Temperature Prediction Pipeline ──────────────────────────")

    # load
    raw = load_data(CSV_PATH)
    print(f"    Data loaded  : {len(raw)} rows  |  {raw['date'].iloc[0].date()} → {raw['date'].iloc[-1].date()}")

    # EDA & seasonal
    plot_eda(raw)
    plot_seasonal(raw)

    # feature engineering & split
    df = engineer_features(raw)
    X_train, y_train, X_test, y_test = split_data(df)
    print(f"    Train rows   : {len(X_train)}   |   Test rows: {len(X_test)}")
    print(f"    Features     : {len(FEATURES)}")

    # models
    print("\n    Training baseline Random Forest …")
    rf_base = train_baseline(X_train, y_train)

    print("    Running GridSearchCV calibration (24 param combos × 5-fold CV) …")
    rf_tuned = calibrate(X_train, y_train)

    # evaluate
    pred_tuned, rmse_tuned, r2_tuned = evaluate(rf_base, rf_tuned, X_train, y_train, X_test, y_test)

    # feature importance
    plot_feature_importance(rf_tuned)

    # BBMP heatmap
    polygons, centroids = read_shapefile_polygons(SHP_PATH)
    print(f"    Shapefile    : {len(polygons)} BBMP wards parsed")
    plot_bbmp_heatmap(raw, polygons, centroids)

    # summary table
    plot_results_table(rf_base, rf_tuned, X_test, y_test, rmse_tuned, r2_tuned)

    print("\n── Pipeline complete. Outputs: 01–06 PNG files ────────────────────────\n")


if __name__ == "__main__":
    main()
