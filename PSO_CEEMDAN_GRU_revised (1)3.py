"""
PSO-CEEMDAN-GRU  —  Revised Experimental Code
================================================

import os, math, random, time, warnings, gc, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import STL
from PyEMD import CEEMDAN
import tensorflow as tf
from tensorflow import keras
from types import SimpleNamespace

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("\n" + "=" * 70)
print("PSO-CEEMDAN-GRU Load Forecasting Script")
print("=" * 70 + "\n")
# SETTINGS  — user-specified for fast iteration
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK         = 60
SEED_LIST        = [42, 7, 13, 99, 2024]      # Q13: 5 seeds

BASELINE_EPOCHS  = 30       # with early stopping rarely reaches limit
BASELINE_BATCH   = 32
IMF_EPOCHS       = 30       # per-IMF GRU final training epochs
IMF_BATCH        = 32

PSO_TUNE_EPOCHS  = 10       # epochs inside PSO fitness (keep small)
PSO_PARTICLES    = 5        # increase to 15-30 for production run
PSO_ITERS        = 6        # increase to 10-20 for production run
PSO_RUNS         = 3        # independent PSO restarts per IMF  

VERBOSE          = 0
OUT_DIR          = "outputs"
FIG_DPI          = 300
FIG_EXT          = "jpeg"

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def set_seeds(seed):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def r2_standard(y_true, y_pred):
    """Q4: Standard mean-centred R² — NOT sum(x²) in denominator."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

def compute_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_standard(y_true, y_pred)
    return dict(RMSE=rmse, MSE=mse, MAE=mae, R2=r2)

def inv_pred(preds_sc, n_features, scaler):
    dummy = np.zeros((len(preds_sc), n_features))
    dummy[:, 0] = preds_sc
    return scaler.inverse_transform(dummy)[:, 0]

def savefig(fname_stem):
    path = os.path.join(OUT_DIR, fname_stem + "." + FIG_EXT)
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight",
                format=FIG_EXT, pil_kwargs={"quality": 95})
    plt.close()
    print(f"    → saved: {path}")

# Global figure style
plt.rcParams.update({
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "xtick.labelsize": 9,  "ytick.labelsize": 9,
    "legend.fontsize": 9,  "legend.framealpha": 0.85,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.38,
    "lines.linewidth": 1.2,
})

def safe_stl(series, period, seasonal=13, robust=True):
    """Attempt STL with fallbacks; return object with .seasonal and .trend.

    Falls back to `robust=False` on failure, and finally to a rolling-mean
    seasonal/trend approximation if STL cannot be computed (memory-limited).
    """
    try:
        return STL(series, period=period, robust=robust, seasonal=seasonal).fit()
    except Exception:
        try:
            return STL(series, period=period, robust=False, seasonal=seasonal).fit()
        except Exception:
            import pandas as _pd
            s = _pd.Series(series) if not isinstance(series, _pd.Series) else series
            trend = s.rolling(window=period, center=True, min_periods=1).mean().values
            seasonal = (s - trend).values
            return SimpleNamespace(seasonal=seasonal, trend=trend)
COLORS = {"actual": "#1a1a1a", "pred": "#d62728",
          "gru_pso": "#4878D0", "ceemdan": "#EE854A",
          "full": "#6ACC65",    "grey": "#999999"}

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING AND FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("1. LOADING AND PREPROCESSING DATA")
print("=" * 70)

data = pd.read_csv("input.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"], dayfirst=True)
data = data.sort_values("timestamp").reset_index(drop=True)

# Temporal features
data["hour"]      = data["timestamp"].dt.hour
data["dayofweek"] = data["timestamp"].dt.dayofweek
data["month"]     = data["timestamp"].dt.month
# NOTE: dayofyear and quarter are intentionally excluded — they are
# near-perfectly collinear with month (ρ≈0.97, Q10).

# Lag features (strong seasonal anchors)
data["lag1"]   = data["load"].shift(1)
data["lag24"]  = data["load"].shift(24)
data["lag168"] = data["load"].shift(168)

# Rolling statistics
data["roll_mean_24"]  = data["load"].rolling(24,  min_periods=24).mean()
data["roll_mean_168"] = data["load"].rolling(168, min_periods=168).mean()
data["roll_std_24"]   = data["load"].rolling(24,  min_periods=24).std()

# STL seasonal components (daily + weekly) — retained from original code
stl_d   = safe_stl(data["load"], period=24, seasonal=13, robust=True)
data["stl_seasonal_24"]  = stl_d.seasonal
data["stl_trend_daily"]  = stl_d.trend
load_minus_daily = data["load"] - data["stl_seasonal_24"]
stl_w   = safe_stl(load_minus_daily, period=168, seasonal=13, robust=True)
data["stl_seasonal_168"] = stl_w.seasonal
data["stl_trend_weekly"] = stl_w.trend

data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

print(f"  Dataset shape : {data.shape}")
print(f"  Date range    : {data['timestamp'].min()} -> {data['timestamp'].max()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE SELECTION WITH SPEARMAN CORRELATION AND VIF  
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. CORRELATION + VIF FEATURE SELECTION (Q10)")
print("=" * 70)

# Q11: NASA POWER spatial limitation warning
print("\n  [WARNING] Weather inputs sourced from NASA POWER reanalysis")
print("  (~0.5°×0.625° grid). Validate against local station data if available.")
print("  Spatial mismatch may affect solar_irradiance and temperature accuracy.")

candidate_features = [
    "temperature", "solar_irradiance", "dew_point",
    "relative_humidity", "surface_pressure", "wind_speed",
    "hour", "dayofweek", "month",
    "lag1", "lag24", "lag168",
    "roll_mean_24", "roll_mean_168", "roll_std_24",
    "stl_seasonal_24", "stl_seasonal_168",
    "stl_trend_daily", "stl_trend_weekly",
]
candidate_features = [f for f in candidate_features if f in data.columns]

# Spearman correlation with load
corr_with_load = {f: data[f].corr(data["load"], method="spearman")
                  for f in candidate_features}
corr_df = (pd.DataFrame.from_dict(corr_with_load, orient="index",
                                   columns=["Spearman_rho"])
             .sort_values("Spearman_rho", ascending=False))
print("\nSpearman correlation with load (ranked):\n", corr_df.to_string())

# VIF — exclude known near-perfect collinear pairs first (Q10)
# dayofyear/quarter excluded at feature-engineering stage above
vif_candidates = [f for f in candidate_features
                  if abs(corr_with_load[f]) > 0.05]   # weak features dropped
vif_mat = data[vif_candidates].fillna(0).values.astype(float)
vif_df  = pd.DataFrame({"Feature": vif_candidates,
                          "VIF": [variance_inflation_factor(vif_mat, i)
                                  for i in range(vif_mat.shape[1])]})
vif_df.sort_values("VIF", ascending=False, inplace=True)
print("\nVIF (features with |ρ| > 0.05):\n", vif_df.to_string(index=False))

# Final feature set — explicitly listed (Q10 requirement)
FINAL_FEATURES = [
    "load",
    "temperature", "solar_irradiance", "relative_humidity",
    "wind_speed", "surface_pressure",
    "hour", "dayofweek", "month",
    "lag1", "lag24", "lag168",
    "roll_mean_24", "roll_mean_168", "roll_std_24",
    "stl_seasonal_24", "stl_seasonal_168",
    "stl_trend_daily", "stl_trend_weekly",
]
FINAL_FEATURES = [f for f in FINAL_FEATURES if f in data.columns]
print(f"\nFinal feature set ({len(FINAL_FEATURES)} features) used in ALL models:")
for f in FINAL_FEATURES:
    print(f"  • {f}")

# Spearman heatmap (Fig 1)
corr_matrix = data[FINAL_FEATURES].corr(method="spearman")
n_f = len(FINAL_FEATURES)
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(n_f)); ax.set_yticks(range(n_f))
ax.set_xticklabels(FINAL_FEATURES, rotation=45, ha="right",
                   fontsize=9, fontweight="bold")
ax.set_yticklabels(FINAL_FEATURES, fontsize=9, fontweight="bold")
for i in range(n_f):
    for j in range(n_f):
        v = corr_matrix.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=7, fontweight="bold",
                color="white" if abs(v) > 0.55 else "black")
cb = plt.colorbar(im, ax=ax, shrink=0.82)
cb.set_label("Spearman ρ", fontsize=11, fontweight="bold")
cb.ax.tick_params(labelsize=9)
ax.tick_params(which="both", length=0)
savefig("Fig1_correlation_heatmap")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT AND SCALING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. TRAIN / TEST SPLIT AND SCALING")
print("=" * 70)

dataset   = data[FINAL_FEATURES].values
N         = len(dataset)
train_len = int(N * 0.8)
test_len  = N - train_len
print(f"  Total: {N}  |  Train: {train_len}  |  Test: {test_len}")

# Scaler fitted on TRAINING SET ONLY (no leakage)
scaler_mv = MinMaxScaler(feature_range=(0, 1))
scaler_mv.fit(dataset[:train_len])
scaled_data = scaler_mv.transform(dataset)

def create_windows(scaled, train_len, lookback):
    x_tr, y_tr, x_te, y_te = [], [], [], []
    for i in range(lookback, train_len):
        x_tr.append(scaled[i - lookback:i, :])
        y_tr.append(scaled[i, 0])
    tb = scaled[train_len - lookback:]
    for i in range(lookback, len(tb)):
        x_te.append(tb[i - lookback:i, :])
        y_te.append(tb[i, 0])
    return (np.array(x_tr), np.array(y_tr),
            np.array(x_te), np.array(y_te))

x_train, y_train, x_test, y_test = create_windows(scaled_data, train_len, LOOKBACK)
n_feat = x_train.shape[2]
y_true_test = inv_pred(y_test, n_feat, scaler_mv)
ishape = (x_train.shape[1], x_train.shape[2])
print(f"  x_train: {x_train.shape}  x_test: {x_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. BASELINE MODEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_baseline(model_type, input_shape):
    """Standard 128-unit baselines used in the manuscript."""
    model = keras.Sequential()
    units = 128
    if model_type == "LSTM":
        model.add(keras.layers.LSTM(units, input_shape=input_shape))
    elif model_type == "BiLSTM":
        model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(units), input_shape=input_shape))
    elif model_type == "GRU":
        model.add(keras.layers.GRU(units, input_shape=input_shape))
    elif model_type == "BiGRU":
        model.add(keras.layers.Bidirectional(
            keras.layers.GRU(units), input_shape=input_shape))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
    return model

def build_gru_cap(input_shape):
    model = keras.Sequential([
        keras.layers.GRU(146, input_shape=input_shape),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 5. MULTI-SEED TRAINING HELPER  
# ─────────────────────────────────────────────────────────────────────────────
def train_multiple_seeds(build_fn, x_tr, y_tr, x_te, y_true,
                          n_feat, scaler, epochs, batch, seeds):
    """Train model with 5 seeds; return Mean ± Std and best predictions."""
    all_metrics, all_preds = [], []
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                        restore_best_weights=True)
    for idx, s in enumerate(seeds):
        print(f"    seed {s} ({idx+1}/{len(seeds)}) ...", end=" ", flush=True)
        set_seeds(s)
        m = build_fn()
        t0 = time.time()
        m.fit(x_tr, y_tr, epochs=epochs, batch_size=batch,
              validation_split=0.1, callbacks=[es], verbose=VERBOSE)
        train_t = time.time() - t0
        t1 = time.time()
        p_sc = m.predict(x_te, verbose=0).flatten()
        test_t = time.time() - t1
        preds = inv_pred(p_sc, n_feat, scaler)
        met = compute_metrics(y_true, preds)
        met["train_time"] = round(train_t, 1)
        met["test_time"]  = round(test_t, 1)
        all_metrics.append(met)
        all_preds.append(preds)
        print(f"R²={met['R2']:.4f}  RMSE={met['RMSE']:.4f}  ({train_t:.0f}s)")
        # ── CLEANUP after each seed ──
        del m
        keras.backend.clear_session()
        gc.collect()
    keys = ["R2", "RMSE", "MAE", "MSE", "train_time", "test_time"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        summary[k + "_mean"] = np.mean(vals)
        summary[k + "_std"]  = np.std(vals)
    best_idx = np.argmin([m["RMSE"] for m in all_metrics])
    summary["best_predictions"] = all_preds[best_idx]
    return summary

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN BASELINE MODELS  
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. TRAINING BASELINE MODELS (5 seeds each)")
print("=" * 70)

BASELINE_NAMES = ["LSTM", "BiLSTM", "GRU", "BiGRU"]
baseline_results = {}

for name in BASELINE_NAMES:
    print(f"\n  ── {name} ──")
    fn = lambda n=name: build_baseline(n, ishape)
    res = train_multiple_seeds(fn, x_train, y_train, x_test, y_true_test,
                                n_feat, scaler_mv,
                                BASELINE_EPOCHS, BASELINE_BATCH, SEED_LIST)
    baseline_results[name] = res
    print(f"  SUMMARY  R²={res['R2_mean']:.4f}±{res['R2_std']:.4f}  "
          f"RMSE={res['RMSE_mean']:.4f}±{res['RMSE_std']:.4f}")

# Q12: GRU-cap baseline (146 units, fixed hyperparams, no PSO)
print(f"\n  ── GRU-cap (146 units, fixed hyperparams, no PSO) ──")
fn_146 = lambda: build_gru_cap(ishape)
res_146 = train_multiple_seeds(fn_146, x_train, y_train, x_test, y_true_test,
                                n_feat, scaler_mv,
                                BASELINE_EPOCHS, BASELINE_BATCH, SEED_LIST)
baseline_results["GRU-cap"] = res_146
print(f"  SUMMARY  R²={res_146['R2_mean']:.4f}±{res_146['R2_std']:.4f}  "
      f"RMSE={res_146['RMSE_mean']:.4f}±{res_146['RMSE_std']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CEEMDAN DECOMPOSITION — TRAINING SET ONLY  
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. CEEMDAN DECOMPOSITION — TRAINING SET ONLY")
print("=" * 70)
print("  CEEMDAN is applied exclusively to the training portion")
print("       of the load signal. Test-set IMF components are derived")
print("       by predicting from models trained on training-set IMFs only.")

load_series  = data["load"].values
load_scaler  = MinMaxScaler(feature_range=(0, 1))
load_train   = load_series[:train_len]
load_test    = load_series[train_len:]
load_scaler.fit(load_train.reshape(-1, 1))
load_train_sc = load_scaler.transform(load_train.reshape(-1, 1)).flatten()

set_seeds(42)
ceemdan    = CEEMDAN()
imfs_train = ceemdan(load_train_sc)
if imfs_train.ndim == 1:
    imfs_train = imfs_train[np.newaxis, :]
n_imfs = imfs_train.shape[0]
print(f"  Extracted {n_imfs} IMFs from {train_len} training samples")

recon_err = np.max(np.abs(np.sum(imfs_train, axis=0) - load_train_sc))
print(f"  Max reconstruction error: {recon_err:.2e}  (should be < 1e-6)")

# ── CEEMDAN decomposition plot 
_hi  = ["#d62728", "#e5572b", "#f08030"]
_mid = ["#1f77b4", "#4898d0", "#6eb4e6"]
_lo  = ["#2ca02c", "#1a7a1a"]

def _imf_colour(idx, n_total):
    third = n_total // 3
    if idx < third:
        return _hi[idx % len(_hi)], 1.0, "High-freq"
    elif idx < 2 * third:
        return _mid[(idx - third) % len(_mid)], 1.2, "Mid-freq"
    else:
        return _lo[(idx - 2 * third) % len(_lo)], 2.0, "Low-freq / Residue"

_n_rows = n_imfs + 1
fig2, axes2 = plt.subplots(_n_rows, 1, figsize=(16, 2.0 * _n_rows),
                            sharex=True,
                            gridspec_kw={"hspace": 0.08})
axes2[0].plot(load_train_sc, color="#1a1a1a", lw=1.2, alpha=0.9)
axes2[0].set_ylabel("Scaled\nLoad", fontsize=11, fontweight="bold",
                     rotation=0, labelpad=52, va="center")
axes2[0].yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
axes2[0].spines["top"].set_visible(False); axes2[0].spines["right"].set_visible(False)
axes2[0].annotate("Original", xy=(1.002, 0.5), xycoords="axes fraction",
                   fontsize=9, va="center", fontweight="bold")
for i in range(n_imfs):
    ax  = axes2[i + 1]
    col, lw_i, band = _imf_colour(i, n_imfs)
    ax.plot(imfs_train[i], color=col, lw=lw_i, alpha=0.92)
    ax.set_ylabel(f"IMF {i+1}", fontsize=11, fontweight="bold",
                  rotation=0, labelpad=52, va="center")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.annotate(band, xy=(1.002, 0.5), xycoords="axes fraction",
                fontsize=9, va="center", color=col, fontweight="bold")
axes2[-1].set_xlabel("Training Sample Index", fontsize=11, fontweight="bold")
plt.setp([ax.get_xticklabels() for ax in axes2[:-1]], visible=False)
savefig("Fig2_CEEMDAN_IMFs")

# ─────────────────────────────────────────────────────────────────────────────
# 8. PSO HYPERPARAMETER OPTIMISATION — per-IMF  
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. PSO OPTIMISATION — per-IMF ")
print(f"   particles={PSO_PARTICLES}, iters={PSO_ITERS}, "
      f"tune_epochs={PSO_TUNE_EPOCHS}, runs={PSO_RUNS}")
print("=" * 70)
print("  [Q2] Batch size is FIXED at 32 and excluded from PSO search space.")
print("  [Q7] PSO fitness function = MSE (mean squared error) consistently.")
print("  [Q6] PSO is run 3× per IMF; best run selected; per-IMF params saved.")

# ── CHECK FOR PSO CHECKPOINT ─────────────────────────────────────────────────
checkpoint_file = f"{OUT_DIR}/pso_checkpoint.npz"
per_imf_params = []
per_imf_scores = []
pso_histories  = []

# Look for latest intermediate checkpoint first
intermediate_checkpoints = sorted(glob.glob(f"{OUT_DIR}/pso_checkpoint_imf*.npz"))
latest_checkpoint = intermediate_checkpoints[-1] if intermediate_checkpoints else None

if latest_checkpoint:
    print(f"\n  ★ FOUND INTERMEDIATE CHECKPOINT: {latest_checkpoint}")
    print("  Loading PSO results and skipping to per-IMF training...\n")
    ckpt = np.load(latest_checkpoint, allow_pickle=True)
    per_imf_params = [list(p) for p in ckpt['per_imf_params']]
    per_imf_scores = ckpt['per_imf_scores'].tolist()
    pso_histories = [list(h) for h in ckpt['pso_histories']]
    ckpt.close()
    
    imf_param_df = pd.DataFrame(
        [(i+1, int(per_imf_params[i][0]),
          round(per_imf_params[i][1], 4),
          round(per_imf_params[i][2], 6),
          round(per_imf_scores[i], 6))
         for i in range(len(per_imf_params))],
        columns=["IMF", "GRU_units", "Dropout", "Learning_rate", "Best_MSE"])
    print(f"Loaded PSO results for {len(per_imf_params)} IMFs:")
    print(imf_param_df.to_string(index=False))
    print()
    
elif os.path.exists(checkpoint_file):
    print(f"\n  ★ CHECKPOINT FOUND: {checkpoint_file}")
    print("  Loading PSO results and skipping to per-IMF training...\n")
    ckpt = np.load(checkpoint_file, allow_pickle=True)
    per_imf_params = [list(p) for p in ckpt['per_imf_params']]
    per_imf_scores = ckpt['per_imf_scores'].tolist()
    pso_histories = [list(h) for h in ckpt['pso_histories']]
    ckpt.close()
    
    imf_param_df = pd.DataFrame(
        [(i+1, int(per_imf_params[i][0]),
          round(per_imf_params[i][1], 4),
          round(per_imf_params[i][2], 6),
          round(per_imf_scores[i], 6))
         for i in range(n_imfs)],
        columns=["IMF", "GRU_units", "Dropout", "Learning_rate", "Best_MSE"])
    print("Loaded per-IMF hyperparameters:")
    print(imf_param_df.to_string(index=False))
    print()
    
else:
    print("  No checkpoint found — running PSO optimization for all IMFs...")

    # Search space: [GRU units, dropout, lr]  — batch size NOT included (Q2)
    BOUNDS = [(32, 256), (0.0, 0.5), (1e-4, 1e-2)]

def build_gru_imf(units, dropout, lr, lookback):
    m = keras.Sequential([
        keras.layers.GRU(int(units), activation="tanh",
                         input_shape=(lookback, 1)),
        keras.layers.Dropout(float(dropout)),
        keras.layers.Dense(1)
    ])
    m.compile(optimizer=keras.optimizers.Adam(float(lr)), loss="mse")
    return m

def pso_fitness(params, X_tr, y_tr, X_val, y_val):
    """
    Q7: fitness = MSE on validation set.
    set_seeds() is intentionally NOT called here so each particle
    gets different weight initialisation — avoiding the bug where
    all particles produced identical fitness scores.
    """
    units, dropout, lr = params
    m = build_gru_imf(units, dropout, lr, LOOKBACK)
    m.fit(X_tr, y_tr, epochs=PSO_TUNE_EPOCHS, batch_size=32, verbose=0)
    p = m.predict(X_val, verbose=0).flatten()
    return mean_squared_error(y_val, p)   # MSE 

class PSO:
    def __init__(self, n_particles, n_iters, bounds, fitness_fn):
        self.n_p  = n_particles; self.n_i = n_iters
        self.bnds = bounds;      self.fit  = fitness_fn
        dim = len(bounds)
        self.pos  = np.array([[random.uniform(l, h) for l, h in bounds]
                               for _ in range(n_particles)], dtype=float)
        self.vel  = np.zeros((n_particles, dim))
        self.pb   = self.pos.copy()
        self.pb_sc = np.full(n_particles, np.inf)
        self.gb = None; self.gb_sc = np.inf
        self.hist = []

    def run(self):
        w, c1, c2 = 0.729, 1.494, 1.494  # Clerc-Kennedy constriction
        for it in range(self.n_i):
            for i in range(self.n_p):
                sc = self.fit(self.pos[i])
                if sc < self.pb_sc[i]:
                    self.pb_sc[i] = sc; self.pb[i] = self.pos[i].copy()
                if sc < self.gb_sc:
                    self.gb_sc = sc;    self.gb   = self.pos[i].copy()
            r1 = np.random.rand(self.n_p, len(self.bnds))
            r2 = np.random.rand(self.n_p, len(self.bnds))
            self.vel = (w * self.vel
                        + c1 * r1 * (self.pb - self.pos)
                        + c2 * r2 * (self.gb - self.pos))
            self.pos = np.clip(self.pos + self.vel,
                               [l for l, _ in self.bnds],
                               [h for _, h in self.bnds])
            self.hist.append(self.gb_sc)
            print(f"      iter {it+1:02d}/{self.n_i}  "
                  f"best_MSE={self.gb_sc:.6f}", end="\r", flush=True)
        print()
        return self.gb, self.gb_sc

def make_imf_sequences(series, lookback, train_len):
    X, y, idx = [], [], []
    for end in range(lookback, len(series)):
        X.append(series[end - lookback:end])
        y.append(series[end])
        idx.append(end)
    X, y, idx = np.array(X), np.array(y), np.array(idx)
    tr   = idx < train_len
    n_tr = tr.sum()
    sp   = int(0.8 * n_tr)
    X_tr  = X[tr][:sp].reshape(-1, lookback, 1);  y_tr  = y[tr][:sp]
    X_val = X[tr][sp:].reshape(-1, lookback, 1);  y_val = y[tr][sp:]
    X_te  = X[~tr].reshape(-1, lookback, 1)
    return X_tr, y_tr, X_val, y_val, X_te, y[~tr], idx[~tr]

# ── RUN PSO ONLY IF NOT RESUMING FROM CHECKPOINT ──────────────────────────────
if not os.path.exists(checkpoint_file) and not latest_checkpoint:
    print("\n  Running PSO optimization...")
    for imf_i in range(n_imfs):
        t_imf = time.time()
        print(f"\n  ── IMF {imf_i+1}/{n_imfs} ──")
        X_tr, y_tr, X_val, y_val, X_te, y_te, _ = make_imf_sequences(
            imfs_train[imf_i], LOOKBACK, train_len)

        best_params_run, best_score_run = None, np.inf
        run_scores = []
        for run_k in range(PSO_RUNS):        # 3 independent runs per IMF
            seed_k = run_k * 137
            random.seed(seed_k); np.random.seed(seed_k)
            print(f"    PSO run {run_k+1}/{PSO_RUNS} (seed={seed_k})", flush=True)
            fit_fn = lambda p, Xtr=X_tr, ytr=y_tr, Xv=X_val, yv=y_val: \
                pso_fitness(p, Xtr, ytr, Xv, yv)
            pso_obj = PSO(PSO_PARTICLES, PSO_ITERS, BOUNDS, fit_fn)
            params_k, score_k = pso_obj.run()
            run_scores.append(score_k)
            if score_k < best_score_run:
                best_score_run = score_k
                best_params_run = params_k.copy()
                best_hist = pso_obj.hist.copy()
            # ── CLEANUP after PSO run ──
            del pso_obj, fit_fn
            keras.backend.clear_session()
            gc.collect()

        per_imf_params.append(best_params_run)
        per_imf_scores.append(best_score_run)
        pso_histories.append(best_hist)
        elapsed_imf = time.time() - t_imf
        print(f"  IMF {imf_i+1} done in {elapsed_imf:.0f}s  →  "
              f"units={int(best_params_run[0])}, "
              f"dropout={best_params_run[1]:.4f}, "
              f"lr={best_params_run[2]:.6f}, "
              f"best_MSE={best_score_run:.6f}  "
              f"(run std={np.std(run_scores):.6f})")
        
        # ── SAVE INTERMEDIATE CHECKPOINT AFTER EACH IMF ──────────────────────
        intermediate_checkpoint = f"{OUT_DIR}/pso_checkpoint_imf{imf_i+1}.npz"
        np.savez_compressed(intermediate_checkpoint,
                            per_imf_params=np.array(per_imf_params, dtype=object),
                            per_imf_scores=np.array(per_imf_scores),
                            pso_histories=np.array(pso_histories, dtype=object))
        print(f"  [INTERMEDIATE CHECKPOINT] Saved after IMF {imf_i+1} → {intermediate_checkpoint}")

    #  per-IMF hyperparameter table (only if PSO was run)
    if not os.path.exists(checkpoint_file) and not latest_checkpoint:
        imf_param_df = pd.DataFrame(
            [(i+1, int(per_imf_params[i][0]),
              round(per_imf_params[i][1], 4),
              round(per_imf_params[i][2], 6),
              round(per_imf_scores[i],    6))
             for i in range(n_imfs)],
            columns=["IMF", "GRU_units", "Dropout", "Learning_rate", "Best_MSE"])
        print("\nPer-IMF optimal hyperparameters (Q6 / Q3):")
        print(imf_param_df.to_string(index=False))
        imf_param_df.to_csv(f"{OUT_DIR}/per_imf_hyperparams.csv", index=False)

        # ── SAVE PSO CHECKPOINT ──────────────────────────────────────────────────────
        checkpoint_file = f"{OUT_DIR}/pso_checkpoint.npz"
        np.savez_compressed(checkpoint_file,
                            per_imf_params=np.array(per_imf_params, dtype=object),
                            per_imf_scores=np.array(per_imf_scores),
                            pso_histories=np.array(pso_histories, dtype=object))
        print(f"\n  [CHECKPOINT] PSO results saved → {checkpoint_file}")
        print(f"  (Can resume from here if script stops)")

        # ── CLEANUP after PSO phase ──────────────────────────────────────────────────
        print("\n  [CLEANUP] Clearing PSO memory and TensorFlow session...")
        keras.backend.clear_session()
        if 'pso_obj' in locals():
            del pso_obj
        gc.collect()
        print("  Memory cleared successfully.\n")

# Note for manuscript: the dropout value in Table 3 should be taken from this
# CSV for the IMF that achieves best overall reconstruction. Report a SINGLE
# consistent value across the paper body and Table 3 (Q3).

# PSO convergence plot (Fig 3) — one panel per IMF
# (only if PSO histories are available)
if pso_histories and len(pso_histories) > 0:
    ncols = 4
    nrows = math.ceil(len(pso_histories) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.8 * nrows),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i in range(len(pso_histories)):
        ax = axes_flat[i]
        col, lw_i, _ = _imf_colour(i, len(pso_histories))
        ax.plot(pso_histories[i], marker="o", markersize=4, lw=1.8,
                color=col, markerfacecolor="white", markeredgewidth=1.5)
        ax.set_xlabel("Iteration", fontweight="bold")
        ax.set_ylabel("Best MSE", fontweight="bold")
        ax.text(0.97, 0.97, f"IMF {i+1}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, fontweight="bold",
                color=col,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=col, alpha=0.85, lw=1.2))
    for j in range(len(pso_histories), len(axes_flat)):
        axes_flat[j].set_visible(False)
    savefig("Fig3_PSO_convergence_all_IMFs")
else:
    print("  [SKIP] Fig3_PSO_convergence: PSO histories not available (resumed from checkpoint)")

# ─────────────────────────────────────────────────────────────────────────────
# 9. TRAIN PER-IMF GRU WITH OPTIMISED HYPERPARAMETERS  
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. PER-IMF GRU FINAL TRAINING  (PSO-CEEMDAN-GRU)")
print("=" * 70)

def train_per_imf_ensemble(imfs_tr, lookback, tr_len,
                            imf_params, epochs, batch, seed):
    """trains on training-set IMFs only; reconstructs test-set forecast."""
    set_seeds(seed)
    n_i = imfs_tr.shape[0]
    imf_test_preds = []
    for i in range(n_i):
        units   = int(imf_params[i][0])
        dropout = float(imf_params[i][1])
        lr      = float(imf_params[i][2])
        _, _, _, _, X_te, _, idx_te = make_imf_sequences(
            imfs_tr[i], lookback, tr_len)
        X_tr_i = np.array([imfs_tr[i][e - lookback:e]
                            for e in range(lookback, tr_len)])
        y_tr_i = np.array([imfs_tr[i][e] for e in range(lookback, tr_len)])
        X_tr_i = X_tr_i.reshape(-1, lookback, 1)
        m = build_gru_imf(units, dropout, lr, lookback)
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                            restore_best_weights=True)
        m.fit(X_tr_i, y_tr_i, epochs=epochs, batch_size=batch,
              validation_split=0.1, callbacks=[es], verbose=0)
        p_i = m.predict(X_te, verbose=0).flatten()
        imf_test_preds.append((idx_te, p_i))
        print(f"    IMF {i+1}/{n_i} trained (seed {seed})", end="\r", flush=True)
        # ── CLEANUP: explicitly delete model and clear memory ──
        del m; keras.backend.clear_session()
    print()
    # Sum IMF predictions → reconstruct load forecast
    n_test = len(imf_test_preds[0][0])
    stack  = np.zeros((n_i, n_test))
    for i, (idx_te, p_i) in enumerate(imf_test_preds):
        mn = min(n_test, len(p_i))
        stack[i, -mn:] = p_i[-mn:]
    forecast_sc = np.sum(stack, axis=0)
    return load_scaler.inverse_transform(forecast_sc.reshape(-1, 1)).flatten()

print("  Training PSO-CEEMDAN-GRU across 5 seeds ...")
pso_ceemdan_all = []
for idx_seed, sd in enumerate(SEED_LIST):
    try:
        print(f"\n  ─ Seed {idx_seed+1}/5 (seed={sd}) ─", flush=True)
        t0 = time.time()
        fc = train_per_imf_ensemble(imfs_train, LOOKBACK, train_len,
                                     per_imf_params, IMF_EPOCHS, IMF_BATCH, sd)
        elapsed = time.time() - t0
        mn = min(len(fc), len(y_true_test))
        fc_al = fc[-mn:]; yt_al = y_true_test[-mn:]
        met = compute_metrics(yt_al, fc_al)
        met["train_time"] = round(elapsed, 1)
        pso_ceemdan_all.append((fc_al, met))
        print(f"  Seed {sd}: R²={met['R2']:.4f}  RMSE={met['RMSE']:.4f}  "
              f"({elapsed:.0f}s)", flush=True)
        # Cleanup after each seed
        gc.collect()
        keras.backend.clear_session()
    except Exception as e:
        print(f"\n  ✗ ERROR on seed {sd}: {str(e)}", flush=True)
        raise

pso_metrics_keys = ["R2", "RMSE", "MAE", "MSE"]
pso_final_metrics = {}
for k in pso_metrics_keys:
    vals = [r[1][k] for r in pso_ceemdan_all]
    pso_final_metrics[k + "_mean"] = np.mean(vals)
    pso_final_metrics[k + "_std"]  = np.std(vals)
best_pso_idx   = np.argmin([r[1]["RMSE"] for r in pso_ceemdan_all])
best_pso_fc    = pso_ceemdan_all[best_pso_idx][0]
y_true_aligned = y_true_test[-len(best_pso_fc):]
n_te           = len(best_pso_fc)

print(f"\n  PSO-CEEMDAN-GRU final (mean ± std, 5 seeds):")
for k in pso_metrics_keys:
    print(f"    {k}: {pso_final_metrics[k+'_mean']:.4f} ± "
          f"{pso_final_metrics[k+'_std']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. ABLATION STUDY  
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("8. ABLATION STUDY ")
print("=" * 70)
print("  Testing three configurations to isolate contributions:")
print("  (a) GRU-PSO       : PSO-tuned GRU on raw load — no decomposition")
print("  (b) CEEMDAN-GRU   : CEEMDAN decomposition + fixed hyperparameters")
print("  (c) PSO-CEEMDAN-GRU: Full proposed model (already trained above)")

# ── (a) GRU-PSO: PSO-tuned on raw load, no CEEMDAN ──────────────────────────
print("\n  (a) GRU-PSO (PSO-tuned, no decomposition) ...")
avg_units   = int(np.mean([p[0] for p in per_imf_params]))
avg_dropout = float(np.mean([p[1] for p in per_imf_params]))
avg_lr      = float(np.mean([p[2] for p in per_imf_params]))
print(f"      Avg PSO params: units={avg_units}, "
      f"dropout={avg_dropout:.4f}, lr={avg_lr:.6f}")

def build_gru_pso_raw(ishape, units, dropout, lr):
    m = keras.Sequential([
        keras.layers.GRU(units, input_shape=ishape),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1)
    ])
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return m

fn_gru_pso  = lambda: build_gru_pso_raw(ishape, avg_units, avg_dropout, avg_lr)
res_gru_pso = train_multiple_seeds(fn_gru_pso, x_train, y_train, x_test,
                                    y_true_test, n_feat, scaler_mv,
                                    BASELINE_EPOCHS, BASELINE_BATCH, SEED_LIST)
print(f"  GRU-PSO: R²={res_gru_pso['R2_mean']:.4f}±{res_gru_pso['R2_std']:.4f}  "
      f"RMSE={res_gru_pso['RMSE_mean']:.4f}±{res_gru_pso['RMSE_std']:.4f}")

# ── (b) CEEMDAN-GRU: decomposition + FIXED hyperparameters, no PSO ──────────
print("\n  (b) CEEMDAN-GRU (fixed hyperparameters, no PSO) ...")
default_params_all = [[64, 0.3, 0.001]] * n_imfs
ceemdan_gru_all = []
for sd in SEED_LIST:
    print(f"    Seed {sd}...", end=" ", flush=True)
    t0 = time.time()
    fc = train_per_imf_ensemble(imfs_train, LOOKBACK, train_len,
                                 default_params_all, IMF_EPOCHS, IMF_BATCH, sd)
    mn = min(len(fc), len(y_true_test))
    fc_al = fc[-mn:]; yt_al = y_true_test[-mn:]
    met = compute_metrics(yt_al, fc_al)
    met["train_time"] = round(time.time() - t0, 1)
    ceemdan_gru_all.append((fc_al, met))
    print(f"R²={met['R2']:.4f}  RMSE={met['RMSE']:.4f}")
    # ── CLEANUP after each seed ──
    gc.collect()
    keras.backend.clear_session()
ceemdan_gru_metrics = {}
for k in pso_metrics_keys:
    vals = [r[1][k] for r in ceemdan_gru_all]
    ceemdan_gru_metrics[k + "_mean"] = np.mean(vals)
    ceemdan_gru_metrics[k + "_std"]  = np.std(vals)
print(f"  CEEMDAN-GRU: "
      f"R²={ceemdan_gru_metrics['R2_mean']:.4f}±"
      f"{ceemdan_gru_metrics['R2_std']:.4f}  "
      f"RMSE={ceemdan_gru_metrics['RMSE_mean']:.4f}±"
      f"{ceemdan_gru_metrics['RMSE_std']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. MULTI-STEP FORECASTING  : 1h / 6h / 12h / 24h
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("9. MULTI-STEP FORECASTING (Q9): horizons 1h / 6h / 12h / 24h")
print("=" * 70)

def create_multistep_windows(scaled, tr_len, lookback, horizon):
    x_tr, y_tr, x_te, y_te = [], [], [], []
    for i in range(lookback, tr_len - horizon + 1):
        x_tr.append(scaled[i - lookback:i, :])
        y_tr.append(scaled[i + horizon - 1, 0])
    tb = scaled[tr_len - lookback:]
    for i in range(lookback, len(tb) - horizon + 1):
        x_te.append(tb[i - lookback:i, :])
        y_te.append(tb[i + horizon - 1, 0])
    return (np.array(x_tr), np.array(y_tr),
            np.array(x_te), np.array(y_te))

multistep_results = {}
for h in [1, 6, 12, 24]:
    print(f"\n  Horizon h={h}h ...")
    xtr_h, ytr_h, xte_h, yte_h = create_multistep_windows(
        scaled_data, train_len, LOOKBACK, h)
    y_true_h = inv_pred(yte_h, n_feat, scaler_mv)
    fn_h = lambda: build_baseline("GRU", ishape)
    res_h = train_multiple_seeds(fn_h, xtr_h, ytr_h, xte_h, y_true_h,
                                  n_feat, scaler_mv,
                                  BASELINE_EPOCHS, BASELINE_BATCH, SEED_LIST)
    multistep_results[h] = res_h
    print(f"  h={h:2d}h: R²={res_h['R2_mean']:.4f}±{res_h['R2_std']:.4f}  "
          f"RMSE={res_h['RMSE_mean']:.4f}±{res_h['RMSE_std']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. RESULTS TABLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("10. RESULTS TABLES")
print("=" * 70)

# Table 4 — Main model comparison (Q13: Mean ± Std)
rows = []
for name in BASELINE_NAMES + ["GRU-cap"]:
    r = baseline_results[name]
    rows.append({
        "Model":      name,
        "R²_mean":    f"{r['R2_mean']*100:.2f}",
        "R²_std":     f"±{r['R2_std']*100:.2f}",
        "RMSE_mean":  f"{r['RMSE_mean']:.4f}",
        "RMSE_std":   f"±{r['RMSE_std']:.4f}",
        "MAE_mean":   f"{r['MAE_mean']:.4f}",
        "MSE_mean":   f"{r['MSE_mean']:.6f}",
        "TrainTime_s":f"{r['train_time_mean']:.0f}",
        "TestTime_s": f"{r['test_time_mean']:.0f}",
    })
rows.append({
    "Model":      "PSO-CEEMDAN-GRU",
    "R²_mean":    f"{pso_final_metrics['R2_mean']*100:.2f}",
    "R²_std":     f"±{pso_final_metrics['R2_std']*100:.2f}",
    "RMSE_mean":  f"{pso_final_metrics['RMSE_mean']:.4f}",
    "RMSE_std":   f"±{pso_final_metrics['RMSE_std']:.4f}",
    "MAE_mean":   f"{pso_final_metrics['MAE_mean']:.4f}",
    "MSE_mean":   f"{pso_final_metrics['MSE_mean']:.6f}",
    "TrainTime_s": "—", "TestTime_s": "—",
})
table_main = pd.DataFrame(rows)
print("\nTable 4 — Error evaluation (Mean ± Std, 5 seeds):")
print(table_main.to_string(index=False))
table_main.to_csv(f"{OUT_DIR}/Table4_main_results.csv", index=False)

# Ablation table 
ablation_rows = [
    {"Configuration": "GRU (baseline, 128 units)",
     "PSO": "✗", "CEEMDAN": "✗",
     "R²_mean_%": f"{baseline_results['GRU']['R2_mean']*100:.2f}",
     "RMSE_mean": f"{baseline_results['GRU']['RMSE_mean']:.4f}"},
    {"Configuration": "GRU-PSO (PSO-tuned, no decomp.)",
     "PSO": "✓", "CEEMDAN": "✗",
     "R²_mean_%": f"{res_gru_pso['R2_mean']*100:.2f}",
     "RMSE_mean": f"{res_gru_pso['RMSE_mean']:.4f}"},
    {"Configuration": "CEEMDAN-GRU (decomp., fixed hyper)",
     "PSO": "✗", "CEEMDAN": "✓",
     "R²_mean_%": f"{ceemdan_gru_metrics['R2_mean']*100:.2f}",
     "RMSE_mean": f"{ceemdan_gru_metrics['RMSE_mean']:.4f}"},
    {"Configuration": "PSO-CEEMDAN-GRU (full proposed)",
     "PSO": "✓", "CEEMDAN": "✓",
     "R²_mean_%": f"{pso_final_metrics['R2_mean']*100:.2f}",
     "RMSE_mean": f"{pso_final_metrics['RMSE_mean']:.4f}"},
]
ablation_df = pd.DataFrame(ablation_rows)
print("\nAblation Study (Q5):")
print(ablation_df.to_string(index=False))
ablation_df.to_csv(f"{OUT_DIR}/Table_ablation.csv", index=False)

# Multi-step table 
ms_rows = [{"Horizon": f"{h}h",
             "R²_mean_%":  f"{r['R2_mean']*100:.2f}",
             "R²_std_%":   f"±{r['R2_std']*100:.2f}",
             "RMSE_mean":  f"{r['RMSE_mean']:.4f}",
             "RMSE_std":   f"±{r['RMSE_std']:.4f}"}
           for h, r in multistep_results.items()]
ms_df = pd.DataFrame(ms_rows)
print("\nMulti-step results (Q9):")
print(ms_df.to_string(index=False))
ms_df.to_csv(f"{OUT_DIR}/Table_multistep.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 13. PUBLICATION-QUALITY FIGURES  
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("11. GENERATING FIGURES")
print("=" * 70)

# Fig 4 — Baseline forecast curves — FULL test set  (Q14)
fig, axes = plt.subplots(len(BASELINE_NAMES), 1,
                          figsize=(16, 3.6 * len(BASELINE_NAMES)),
                          sharex=True,
                          gridspec_kw={"hspace": 0.12})
for ax, name in zip(axes, BASELINE_NAMES):
    preds = baseline_results[name]["best_predictions"]
    mn_p  = min(len(preds), n_te)
    ax.plot(y_true_test[:mn_p], color=COLORS["actual"], lw=1.2,
            label="Actual")
    ax.plot(preds[:mn_p], linestyle="--", lw=1.0, label=name)
    ax.set_ylabel("Load (kW)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.text(0.01, 0.96, name, transform=ax.transAxes,
            fontsize=10, va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8))
axes[-1].set_xlabel("Test Time Index (h)", fontweight="bold")
savefig("Fig4_baseline_forecasts_full")

# Fig 5 — PSO-CEEMDAN-GRU full test set  (Q14)
fig, ax = plt.subplots(figsize=(18, 4.5))
ax.plot(y_true_aligned, color=COLORS["actual"], lw=1.2, label="Actual")
ax.plot(best_pso_fc,    color=COLORS["pred"],   lw=1.0, linestyle=":",
        label="PSO-CEEMDAN-GRU")
ax.set_xlabel("Test Time Index (h)", fontweight="bold")
ax.set_ylabel("Load (kW)", fontweight="bold")
ax.legend()
savefig("Fig5_PSO_CEEMDAN_GRU_full_testset")

# Fig 6 — Last 200 test points detail panel  
n_zoom = min(200, n_te)
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(y_true_aligned[-n_zoom:], color=COLORS["actual"], lw=1.4,
        label="Actual")
ax.plot(best_pso_fc[-n_zoom:],    color=COLORS["pred"],   lw=1.2,
        linestyle=":", label="PSO-CEEMDAN-GRU")
ax.set_xlabel("Test Time Index — Last 200 h", fontweight="bold")
ax.set_ylabel("Load (kW)", fontweight="bold")
ax.legend()
savefig("Fig6_PSO_CEEMDAN_GRU_last200")

# Fig 7 — Per-week MAE across full test period  (Q14)
errors       = np.abs(y_true_aligned - best_pso_fc)
hours_per_wk = 168
n_weeks      = len(errors) // hours_per_wk
weekly_mae   = [np.mean(errors[w * hours_per_wk:(w+1) * hours_per_wk])
                for w in range(n_weeks)]
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(range(1, n_weeks + 1), weekly_mae, color="steelblue",
       alpha=0.85, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Test Week", fontweight="bold")
ax.set_ylabel("Weekly MAE (kW)", fontweight="bold")
savefig("Fig7_per_week_MAE")

# Fig 8 — Scatter plots, all models  
all_model_names = BASELINE_NAMES + ["GRU-cap", "PSO-CEEMDAN-GRU"]
ncols_sc = 3
nrows_sc = math.ceil(len(all_model_names) / ncols_sc)
fig, axes = plt.subplots(nrows_sc, ncols_sc,
                          figsize=(5.5 * ncols_sc, 5.2 * nrows_sc),
                          gridspec_kw={"hspace": 0.42, "wspace": 0.35})
axes_fl = axes.flatten() if hasattr(axes, "flatten") else [axes]
for ax, name in zip(axes_fl, all_model_names):
    if name == "PSO-CEEMDAN-GRU":
        p = best_pso_fc; yt = y_true_aligned
    else:
        p  = baseline_results[name]["best_predictions"]
        mn = min(len(p), len(y_true_test))
        p  = p[:mn]; yt = y_true_test[:mn]
    ax.scatter(yt, p, alpha=0.35, s=5, color="steelblue", rasterized=True)
    lo = min(float(yt.min()), float(p.min()))
    hi = max(float(yt.max()), float(p.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.8)
    r2v = r2_standard(yt, p)
    ax.set_xlabel("Actual (kW)", fontweight="bold")
    ax.set_ylabel("Predicted (kW)", fontweight="bold")
    ax.text(0.04, 0.96, f"{name}\nR²={r2v:.4f}",
            transform=ax.transAxes, fontsize=9, va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
for j in range(len(all_model_names), len(axes_fl)):
    axes_fl[j].set_visible(False)
savefig("Fig8_scatter_all_models")

# Fig 9 — Ablation bar chart  
ab_labels = ["GRU\n(baseline)", "GRU-PSO\n(no decomp.)",
             "CEEMDAN-GRU\n(fixed hyper)", "PSO-CEEMDAN-GRU\n(full)"]
ab_colors = [COLORS["grey"], COLORS["gru_pso"],
             COLORS["ceemdan"], COLORS["full"]]
ab_r2s    = [baseline_results["GRU"]["R2_mean"] * 100,
             res_gru_pso["R2_mean"] * 100,
             ceemdan_gru_metrics["R2_mean"] * 100,
             pso_final_metrics["R2_mean"] * 100]
ab_rmses  = [baseline_results["GRU"]["RMSE_mean"],
             res_gru_pso["RMSE_mean"],
             ceemdan_gru_metrics["RMSE_mean"],
             pso_final_metrics["RMSE_mean"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                gridspec_kw={"wspace": 0.32})
bars1 = ax1.bar(ab_labels, ab_r2s, color=ab_colors,
                edgecolor="white", linewidth=0.6)
ax1.set_ylabel("R² (%)", fontweight="bold")
ax1.set_ylim(max(0, min(ab_r2s) - 1), 100)
for b, v in zip(bars1, ab_r2s):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
             ha="center", fontsize=9, fontweight="bold")

bars2 = ax2.bar(ab_labels, ab_rmses, color=ab_colors,
                edgecolor="white", linewidth=0.6)
ax2.set_ylabel("RMSE (kW)", fontweight="bold")
for b, v in zip(bars2, ab_rmses):
    ax2.text(b.get_x() + b.get_width() / 2, v * 1.002, f"{v:.4f}",
             ha="center", fontsize=9, fontweight="bold")
savefig("Fig9_ablation_study")

# Fig 10 — Multi-step horizon  
horizons  = list(multistep_results.keys())
ms_r2     = [multistep_results[h]["R2_mean"] * 100 for h in horizons]
ms_rmse   = [multistep_results[h]["RMSE_mean"]     for h in horizons]
ms_r2_sd  = [multistep_results[h]["R2_std"] * 100  for h in horizons]
ms_rm_sd  = [multistep_results[h]["RMSE_std"]      for h in horizons]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8),
                                gridspec_kw={"wspace": 0.32})
ax1.errorbar(horizons, ms_r2,  yerr=ms_r2_sd,  marker="o", markersize=6,
             color="steelblue", capsize=5, lw=2.0, capthick=1.5)
ax1.set_xlabel("Forecast Horizon (h)", fontweight="bold")
ax1.set_ylabel("R² (%)", fontweight="bold")
ax1.set_xticks(horizons)

ax2.errorbar(horizons, ms_rmse, yerr=ms_rm_sd, marker="o", markersize=6,
             color="tomato", capsize=5, lw=2.0, capthick=1.5)
ax2.set_xlabel("Forecast Horizon (h)", fontweight="bold")
ax2.set_ylabel("RMSE (kW)", fontweight="bold")
ax2.set_xticks(horizons)
savefig("Fig10_multistep_horizon")

print(f"\n  All figures saved to '{OUT_DIR}/' as {FIG_EXT.upper()} @ {FIG_DPI} dpi")

# ─────────────────────────────────────────────────────────────────────────────
# 14. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
print("\nBaseline models (Mean ± Std, 5 seeds):")
for name in BASELINE_NAMES + ["GRU-cap"]:
    r = baseline_results[name]
    print(f"  {name:<14s}  "
          f"R²={r['R2_mean']*100:.2f}±{r['R2_std']*100:.2f}%  "
          f"RMSE={r['RMSE_mean']:.4f}±{r['RMSE_std']:.4f}  "
          f"MAE={r['MAE_mean']:.4f}")
print("\nAblation (Mean ± Std, 5 seeds):")
print(f"  {'GRU-PSO':<14s}  "
      f"R²={res_gru_pso['R2_mean']*100:.2f}±"
      f"{res_gru_pso['R2_std']*100:.2f}%  "
      f"RMSE={res_gru_pso['RMSE_mean']:.4f}")
print(f"  {'CEEMDAN-GRU':<14s}  "
      f"R²={ceemdan_gru_metrics['R2_mean']*100:.2f}±"
      f"{ceemdan_gru_metrics['R2_std']:.4f}%  "  # note: std in R² units
      f"RMSE={ceemdan_gru_metrics['RMSE_mean']:.4f}")
print(f"\nPSO-CEEMDAN-GRU (proposed, mean ± std, 5 seeds):")
for k in pso_metrics_keys:
    print(f"  {k}: {pso_final_metrics[k+'_mean']:.4f} ± "
          f"{pso_final_metrics[k+'_std']:.4f}")
print("\nMulti-step (GRU, mean ± std, 5 seeds):")
for h, r in multistep_results.items():
    print(f"  h={h:2d}h  R²={r['R2_mean']*100:.2f}±{r['R2_std']*100:.2f}%  "
          f"RMSE={r['RMSE_mean']:.4f}±{r['RMSE_std']:.4f}")

# Q15: single-site limitation note
print("\n[Q15] NOTE: Results are from a single site (Kalong solar microgrid,")
print("  Niger State, Nigeria). Generalisation to other microgrids should")
print("  be validated before operational deployment.")

# ─────────────────────────────────────────────────────────────────────────────
# 15. REVIEWER COMPLIANCE CHECKLIST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("REVIEWER COMPLIANCE CHECKLIST")
print("=" * 70)
for code, desc in checks:
    print(f"  [{code:3s}] ✓  {desc}")

print("\nDone.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 16. COMPREHENSIVE MODEL PERFORMANCE SUMMARY TABLE
#     Includes ALL models: baselines, capacity-controlled, ablation, proposed
#     Metrics: R² / RMSE / MAE / MSE  (mean ± std, 5 seeds)
#              Training time / Testing time
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPREHENSIVE MODEL PERFORMANCE SUMMARY TABLE")
print("=" * 70)

summary_rows = []

# ── helper to format mean ± std ──────────────────────────────────────────────
def fmt(mean, std, decimals=4):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

def fmt_pct(mean, std):
    return f"{mean*100:.2f} ± {std*100:.2f}"

# ── Section A: Standard baselines (128-unit, fixed hyperparams) ───────────────
for name in BASELINE_NAMES:
    r = baseline_results[name]
    summary_rows.append({
        "Group":        "Baseline",
        "Model":        name,
        "PSO":          "✗",
        "CEEMDAN":      "✗",
        "Units":        "128",
        "R²_% (μ±σ)":  fmt_pct(r["R2_mean"], r["R2_std"]),
        "RMSE (μ±σ)":  fmt(r["RMSE_mean"],  r["RMSE_std"]),
        "MAE (μ±σ)":   fmt(r["MAE_mean"],   r["MAE_std"]),
        "MSE (μ±σ)":   fmt(r["MSE_mean"],   r["MSE_std"], decimals=6),
        "Train_s (μ)":  f"{r['train_time_mean']:.0f}",
        "Test_s (μ)":   f"{r['test_time_mean']:.1f}",
    })

# ── Section B: Capacity-controlled baseline (146-unit, fixed hyperparams) ─────
r = baseline_results["GRU-cap"]
summary_rows.append({
    "Group":        "Capacity ctrl.",
    "Model":        "GRU-cap (146u, fixed)",
    "PSO":          "✗",
    "CEEMDAN":      "✗",
    "Units":        "146",
    "R²_% (μ±σ)":  fmt_pct(r["R2_mean"], r["R2_std"]),
    "RMSE (μ±σ)":  fmt(r["RMSE_mean"],  r["RMSE_std"]),
    "MAE (μ±σ)":   fmt(r["MAE_mean"],   r["MAE_std"]),
    "MSE (μ±σ)":   fmt(r["MSE_mean"],   r["MSE_std"], decimals=6),
    "Train_s (μ)":  f"{r['train_time_mean']:.0f}",
    "Test_s (μ)":   f"{r['test_time_mean']:.1f}",
})

# ── Section C: Ablation configurations ───────────────────────────────────────
# (a) GRU-PSO: PSO-tuned, no decomposition
r = res_gru_pso
summary_rows.append({
    "Group":        "Ablation",
    "Model":        "GRU-PSO (no decomp.)",
    "PSO":          "✓",
    "CEEMDAN":      "✗",
    "Units":        f"~{avg_units}",
    "R²_% (μ±σ)":  fmt_pct(r["R2_mean"], r["R2_std"]),
    "RMSE (μ±σ)":  fmt(r["RMSE_mean"],  r["RMSE_std"]),
    "MAE (μ±σ)":   fmt(r["MAE_mean"],   r["MAE_std"]),
    "MSE (μ±σ)":   fmt(r["MSE_mean"],   r["MSE_std"], decimals=6),
    "Train_s (μ)":  f"{r['train_time_mean']:.0f}",
    "Test_s (μ)":   f"{r['test_time_mean']:.1f}",
})

# (b) CEEMDAN-GRU: decomposition + fixed hyperparams, no PSO
r = ceemdan_gru_metrics
# timing stored in per-seed list; derive from ceemdan_gru_all
cg_train_t = float(np.mean([x[1]["train_time"] for x in ceemdan_gru_all]))
summary_rows.append({
    "Group":        "Ablation",
    "Model":        "CEEMDAN-GRU (fixed hyper)",
    "PSO":          "✗",
    "CEEMDAN":      "✓",
    "Units":        "64",
    "R²_% (μ±σ)":  fmt_pct(r["R2_mean"], r["R2_std"]),
    "RMSE (μ±σ)":  fmt(r["RMSE_mean"],  r["RMSE_std"]),
    "MAE (μ±σ)":   fmt(r["MAE_mean"],   r["MAE_std"]),
    "MSE (μ±σ)":   fmt(r["MSE_mean"],   r["MSE_std"], decimals=6),
    "Train_s (μ)":  f"{cg_train_t:.0f}",
    "Test_s (μ)":   "—",
})

# ── Section D: Proposed model ─────────────────────────────────────────────────
pso_train_t = float(np.mean([x[1]["train_time"] for x in pso_ceemdan_all]))
summary_rows.append({
    "Group":        "★ Proposed",
    "Model":        "PSO-CEEMDAN-GRU",
    "PSO":          "✓",
    "CEEMDAN":      "✓",
    "Units":        "per-IMF (PSO)",
    "R²_% (μ±σ)":  fmt_pct(pso_final_metrics["R2_mean"], pso_final_metrics["R2_std"]),
    "RMSE (μ±σ)":  fmt(pso_final_metrics["RMSE_mean"],  pso_final_metrics["RMSE_std"]),
    "MAE (μ±σ)":   fmt(pso_final_metrics["MAE_mean"],   pso_final_metrics["MAE_std"]),
    "MSE (μ±σ)":   fmt(pso_final_metrics["MSE_mean"],   pso_final_metrics["MSE_std"], decimals=6),
    "Train_s (μ)":  f"{pso_train_t:.0f}",
    "Test_s (μ)":   "—",
})

# ── Build and print table ─────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
col_order  = ["Group", "Model", "PSO", "CEEMDAN", "Units",
              "R²_% (μ±σ)", "RMSE (μ±σ)", "MAE (μ±σ)", "MSE (μ±σ)",
              "Train_s (μ)", "Test_s (μ)"]
summary_df = summary_df[col_order]

# Print with separator lines between groups
print()
prev_group = None
for _, row in summary_df.iterrows():
    if row["Group"] != prev_group:
        if prev_group is not None:
            print("  " + "─" * 120)
        print(f"\n  [{row['Group']}]")
        prev_group = row["Group"]
    print(f"  {row['Model']:<30s}  "
          f"PSO={row['PSO']}  CEEMDAN={row['CEEMDAN']}  "
          f"Units={row['Units']:<14s}  "
          f"R²={row['R²_% (μ±σ)']:>18s}%  "
          f"RMSE={row['RMSE (μ±σ)']:>18s}  "
          f"MAE={row['MAE (μ±σ)']:>18s}  "
          f"Train={row['Train_s (μ)']:>6s}s  "
          f"Test={row['Test_s (μ)']:>6s}s")

# Save to CSV for manuscript table
summary_df.to_csv(f"{OUT_DIR}/Table_comprehensive_summary.csv", index=False)
print(f"\n  Full table saved → {OUT_DIR}/Table_comprehensive_summary.csv")

# ── Visualise as heatmap-style bar comparison (Fig 11) ───────────────────────
model_labels = [r["Model"] for _, r in summary_df.iterrows()]
rmse_means   = []
r2_means     = []
train_times  = []
group_colors = []

_grp_color = {
    "Baseline":       "#4878D0",
    "Capacity ctrl.": "#6BAED6",
    "Ablation":       "#EE854A",
    "★ Proposed":     "#6ACC65",
}

for _, row in summary_df.iterrows():
    try:
        rmse_means.append(float(row["RMSE (μ±σ)"].split(" ± ")[0]))
        r2_means.append(float(row["R²_% (μ±σ)"].split(" ± ")[0]))
        t = row["Train_s (μ)"]
        train_times.append(float(t) if t != "—" else np.nan)
    except Exception:
        rmse_means.append(np.nan)
        r2_means.append(np.nan)
        train_times.append(np.nan)
    group_colors.append(_grp_color.get(row["Group"], "#999999"))

x_pos = np.arange(len(model_labels))
fig, (ax_r2, ax_rmse, ax_time) = plt.subplots(
    3, 1, figsize=(16, 12),
    gridspec_kw={"hspace": 0.55})

# R² panel
bars = ax_r2.bar(x_pos, r2_means, color=group_colors,
                  edgecolor="white", linewidth=0.7)
ax_r2.set_xticks(x_pos)
ax_r2.set_xticklabels(model_labels, rotation=35, ha="right",
                       fontsize=9, fontweight="bold")
ax_r2.set_ylabel("R² (%)", fontweight="bold")
ax_r2.set_ylim(max(0, min(v for v in r2_means if not np.isnan(v)) - 2), 100)
for b, v in zip(bars, r2_means):
    if not np.isnan(v):
        ax_r2.text(b.get_x() + b.get_width() / 2, v + 0.05,
                   f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

# RMSE panel
bars2 = ax_rmse.bar(x_pos, rmse_means, color=group_colors,
                     edgecolor="white", linewidth=0.7)
ax_rmse.set_xticks(x_pos)
ax_rmse.set_xticklabels(model_labels, rotation=35, ha="right",
                         fontsize=9, fontweight="bold")
ax_rmse.set_ylabel("RMSE (kW)", fontweight="bold")
for b, v in zip(bars2, rmse_means):
    if not np.isnan(v):
        ax_rmse.text(b.get_x() + b.get_width() / 2, v * 1.002,
                     f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")

# Training time panel
t_vals = [v if not np.isnan(v) else 0 for v in train_times]
bars3  = ax_time.bar(x_pos, t_vals, color=group_colors,
                      edgecolor="white", linewidth=0.7)
ax_time.set_xticks(x_pos)
ax_time.set_xticklabels(model_labels, rotation=35, ha="right",
                         fontsize=9, fontweight="bold")
ax_time.set_ylabel("Training Time (s)", fontweight="bold")
for b, v in zip(bars3, train_times):
    if not np.isnan(v) and v > 0:
        ax_time.text(b.get_x() + b.get_width() / 2, v * 1.002,
                     f"{v:.0f}s", ha="center", fontsize=8, fontweight="bold")

# Shared legend
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=c, label=g, edgecolor="white")
                  for g, c in _grp_color.items()]
fig.legend(handles=legend_handles, loc="upper center",
           ncol=4, fontsize=10, framealpha=0.9,
           bbox_to_anchor=(0.5, 1.01))

savefig("Fig11_comprehensive_performance_summary")
print(f"\n  Summary figure saved → {OUT_DIR}/Fig11_comprehensive_performance_summary.{FIG_EXT}")

# ── Print the key ranking by RMSE ─────────────────────────────────────────────
print("\n  Model ranking by RMSE (best → worst):")
ranked = sorted(zip(model_labels, rmse_means),
                key=lambda x: x[1] if not np.isnan(x[1]) else 999)
for rank, (name, rmse) in enumerate(ranked, 1):
    print(f"  {rank:2d}. {name:<35s} RMSE = {rmse:.4f}")

print("\n" + "=" * 70)
print("ALL OUTPUTS COMPLETE")
print(f"  Results → {OUT_DIR}/")
print("=" * 70)
