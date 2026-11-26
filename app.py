# ==============================================================
# WEBSITE ANALISIS PERBANDINGAN DECISION TREE DAN RANDOM FOREST
# DATASET: TOP INSTAGRAM INFLUENCERS | TARGET: INFLUENCE SCORE
# Jalankan dengan:  streamlit run app.py
# ==============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Decision Tree vs Random Forest - Instagram Influencers",
    page_icon="🌳",
    layout="wide",
)


# ---------------------- TEMA INSTAGRAM (oranye, merah, putih) ----------------------
ORANGE, RED = "#F77737", "#E4281F"
GRADIENT = "linear-gradient(90deg, #FCAF45 0%, #F77737 40%, #E4281F 100%)"

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.5rem; }}
h1 {{
    background: {GRADIENT};
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800;
}}
h2, h3 {{ color: {RED}; }}
[data-testid="stMetric"] {{
    background: #FFFFFF; border: 1px solid #FFD9C2; border-left: 6px solid {ORANGE};
    border-radius: 12px; padding: 12px 16px;
    box-shadow: 0 2px 8px rgba(228, 40, 31, 0.08);
}}
[data-testid="stMetricValue"] {{ color: {RED}; font-weight: 700; }}
button[data-baseweb="tab"] {{ font-weight: 600; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {RED}; }}
div[data-baseweb="tab-highlight"] {{ background: {GRADIENT}; height: 3px; }}
[data-testid="stSidebar"] {{ background: #FFF3EA; border-right: 3px solid {ORANGE}; }}
[data-testid="stSidebar"] h2 {{ color: {RED}; }}
[data-testid="stFileUploaderDropzone"] {{ border: 2px dashed {ORANGE}; background: #FFFFFF; }}
</style>
""", unsafe_allow_html=True)

COLOR_DT, COLOR_RF = ORANGE, RED
plt.rcParams.update({"axes.prop_cycle": plt.cycler(color=[RED, ORANGE, "#FCAF45"])})

DEFAULT_DATASET = Path(__file__).parent / "top_insta_influencers_data.csv"

NUMERIC_COLUMNS = [
    "posts", "followers", "avg_likes",
    "60_day_eng_rate", "new_post_avg_like", "total_likes",
]
NUMERIC_FEATURES = list(NUMERIC_COLUMNS)
CATEGORICAL_FEATURES = ["country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "influence_score"
RANDOM_STATES = [10, 20, 30, 40, 50]


# ---------------------- PREPROCESSING ----------------------

def convert_value(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if value == "":
        return np.nan
    if value.endswith("%"):
        return float(value[:-1])
    multiplier = 1
    if value.endswith("k"):
        multiplier, value = 1_000, value[:-1]
    elif value.endswith("m"):
        multiplier, value = 1_000_000, value[:-1]
    elif value.endswith("b"):
        multiplier, value = 1_000_000_000, value[:-1]
    return float(value.replace(",", "")) * multiplier


def build_preprocessor():
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def build_dt():
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", DecisionTreeRegressor(
            max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42)),
    ])


def build_rf():
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_split=4,
            min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ])


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2 Score": r2_score(y_true, y_pred),
    }


def prediction_table(y_test, pred):
    hasil = pd.DataFrame({"Nilai Aktual": y_test.values, "Nilai Prediksi": pred})
    hasil["Selisih"] = (hasil["Nilai Aktual"] - hasil["Nilai Prediksi"]).abs()
    hasil["Persentase Prediksi"] = np.where(
        hasil["Nilai Aktual"] != 0,
        hasil["Nilai Prediksi"] / hasil["Nilai Aktual"] * 100,
        np.nan,
    )
    return hasil


# ---------------------- PROSES UTAMA (di-cache) ----------------------

@st.cache_data(show_spinner=False)
def load_raw(source):
    if isinstance(source, (str, Path)):
        return pd.read_csv(source, encoding="utf-8-sig")
    return pd.read_csv(source, encoding="utf-8-sig")


def prepare(df_raw):
    df = df_raw.copy()
    missing = [c for c in NUMERIC_COLUMNS + CATEGORICAL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Kolom berikut tidak ditemukan di dataset: " + ", ".join(missing))
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].apply(convert_value)
    X = df[FEATURES].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.notna()
    return df, X.loc[valid], y.loc[valid]


@st.cache_data(show_spinner="Melatih model...")
def run_analysis(X, y):
    out = {}

    # --- Pembagian data utama (random_state=42) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    out["n_train"], out["n_test"] = len(X_train), len(X_test)
    out["y_test"] = y_test

    dt = build_dt().fit(X_train, y_train)
    rf = build_rf().fit(X_train, y_train)
    pred_dt, pred_rf = dt.predict(X_test), rf.predict(X_test)
    out["pred_dt"], out["pred_rf"] = pred_dt, pred_rf
    out["eval_dt"], out["eval_rf"] = evaluate(y_test, pred_dt), evaluate(y_test, pred_rf)

    # Figure pohon
    dt_names = dt.named_steps["preprocessor"].get_feature_names_out()
    fig1, ax1 = plt.subplots(figsize=(25, 12))
    plot_tree(dt.named_steps["model"], feature_names=dt_names, filled=True,
              rounded=True, max_depth=3, fontsize=9, ax=ax1)
    ax1.set_title("Decision Tree untuk Prediksi Influence Score", fontsize=16)
    out["fig_dt_tree"] = fig1

    rf_names = rf.named_steps["preprocessor"].get_feature_names_out()
    fig2, ax2 = plt.subplots(figsize=(25, 12))
    plot_tree(rf.named_steps["model"].estimators_[0], feature_names=rf_names,
              filled=True, rounded=True, max_depth=3, fontsize=9, ax=ax2)
    ax2.set_title("Salah Satu Pohon dalam Random Forest", fontsize=16)
    out["fig_rf_tree"] = fig2

    imp = pd.DataFrame({
        "Fitur": rf_names,
        "Importance": rf.named_steps["model"].feature_importances_,
    }).sort_values("Importance", ascending=False)
    out["importance"] = imp

    # --- 5 kali percobaan ---
    rows = []
    for nomor, seed in enumerate(RANDOM_STATES, start=1):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        e_dt = evaluate(yte, build_dt().fit(Xtr, ytr).predict(Xte))
        e_rf = evaluate(yte, build_rf().fit(Xtr, ytr).predict(Xte))
        rows.append({
            "Percobaan": nomor, "Random State": seed,
            "MAE Decision Tree": e_dt["MAE"], "RMSE Decision Tree": e_dt["RMSE"],
            "R2 Decision Tree": e_dt["R2 Score"],
            "MAE Random Forest": e_rf["MAE"], "RMSE Random Forest": e_rf["RMSE"],
            "R2 Random Forest": e_rf["R2 Score"],
        })
    out["trials"] = pd.DataFrame(rows)
    return out


# ---------------------- TAMPILAN ----------------------

st.title("🌳 Perbandingan Decision Tree dan Random Forest")
st.caption("Prediksi **Influence Score** pada dataset Top Instagram Influencers")

with st.sidebar:
    st.header("📂 Dataset")
    uploaded = st.file_uploader("Upload dataset CSV kamu sendiri (opsional)", type=["csv"])
    st.markdown(
        "Kalau tidak upload, web memakai **dataset bawaan** (Top Instagram Influencers). "
        "File upload harus punya kolom: `posts, followers, avg_likes, 60_day_eng_rate, "
        "new_post_avg_like, total_likes, country, influence_score`."
    )

try:
    if uploaded is not None:
        df_raw = load_raw(uploaded)
        st.success(f"Memakai dataset yang kamu upload: **{uploaded.name}**")
    else:
        df_raw = load_raw(DEFAULT_DATASET)
        st.info("Memakai dataset bawaan: **top_insta_influencers_data.csv**")
    df, X, y = prepare(df_raw)
except Exception as e:
    st.error(f"Dataset tidak bisa diproses: {e}")
    st.stop()

st.subheader("1. Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah baris", df_raw.shape[0])
c2.metric("Jumlah kolom", df_raw.shape[1])
c3.metric("Baris dipakai (target valid)", len(y))
st.dataframe(df_raw.head(20), width="stretch")

R = run_analysis(X, y)

st.subheader("2. Pembagian Data")
a, b = st.columns(2)
a.metric("Data Training (80%)", R["n_train"])
b.metric("Data Testing (20%)", R["n_test"])

tab_dt, tab_rf, tab_cmp, tab_exp = st.tabs([
    "🌲 Decision Tree", "🌳 Random Forest", "⚖️ Perbandingan", "🔁 5 Kali Percobaan",
])

# ----- Decision Tree -----
with tab_dt:
    st.markdown("### Hasil Evaluasi Decision Tree")
    m = st.columns(4)
    for col, k in zip(m, ["MAE", "MSE", "RMSE", "R2 Score"]):
        col.metric(k, f"{R['eval_dt'][k]:.4f}")
    st.markdown("**Visualisasi pohon (kedalaman 3 teratas)**")
    st.pyplot(R["fig_dt_tree"])
    st.markdown("**Tabel hasil prediksi**")
    st.dataframe(prediction_table(R["y_test"], R["pred_dt"]).round(2), width="stretch")

# ----- Random Forest -----
with tab_rf:
    st.markdown("### Hasil Evaluasi Random Forest")
    m = st.columns(4)
    for col, k in zip(m, ["MAE", "MSE", "RMSE", "R2 Score"]):
        col.metric(k, f"{R['eval_rf'][k]:.4f}")
    st.markdown("**Salah satu pohon dalam Random Forest**")
    st.pyplot(R["fig_rf_tree"])
    st.markdown("**Fitur paling berpengaruh (Top 10)**")
    top = R["importance"].head(10).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top["Fitur"], top["Importance"], color=COLOR_RF)
    ax.set_xlabel("Nilai Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    st.dataframe(R["importance"].head(10).round(4), width="stretch")
    st.markdown("**Tabel hasil prediksi**")
    st.dataframe(prediction_table(R["y_test"], R["pred_rf"]).round(2), width="stretch")

# ----- Perbandingan -----
with tab_cmp:
    st.markdown("### Tabel Perbandingan Kedua Algoritma")
    perb = pd.DataFrame({
        "Algoritma": ["Decision Tree", "Random Forest"],
        **{k: [R["eval_dt"][k], R["eval_rf"][k]] for k in ["MAE", "MSE", "RMSE", "R2 Score"]},
    })
    st.dataframe(perb.round(4), width="stretch", hide_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["MAE", "MSE", "RMSE"]):
        ax.bar(perb["Algoritma"], perb[metric], color=[COLOR_DT, COLOR_RF])
        ax.set_title("Perbandingan " + metric)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Perbandingan Error Decision Tree dan Random Forest")
    fig.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(perb["Algoritma"], perb["R2 Score"], color=[COLOR_DT, COLOR_RF])
    ax.set_title("Perbandingan R2 Score")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(R["y_test"].values, label="Nilai Aktual", marker="o", color="#262626")
    ax.plot(R["pred_dt"], label="Decision Tree", marker="x", color=COLOR_DT)
    ax.plot(R["pred_rf"], label="Random Forest", marker="s", color=COLOR_RF)
    ax.set_title("Perbandingan Nilai Aktual dan Hasil Prediksi")
    ax.set_xlabel("Data Testing")
    ax.set_ylabel("Influence Score")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # Kesimpulan
    rmse_dt, rmse_rf = R["eval_dt"]["RMSE"], R["eval_rf"]["RMSE"]
    if rmse_dt < rmse_rf:
        best, best_rmse, best_r2 = "Decision Tree", rmse_dt, R["eval_dt"]["R2 Score"]
    elif rmse_rf < rmse_dt:
        best, best_rmse, best_r2 = "Random Forest", rmse_rf, R["eval_rf"]["R2 Score"]
    else:
        best, best_rmse = "Keduanya memiliki RMSE yang sama", rmse_dt
        best_r2 = max(R["eval_dt"]["R2 Score"], R["eval_rf"]["R2 Score"])

    st.markdown("### Kesimpulan (Pembagian Data Utama)")
    st.success(f"Algoritma terbaik berdasarkan RMSE: **{best}** (RMSE {best_rmse:.4f}, R² {best_r2:.4f})")
    if rmse_dt > 0:
        imp_pct = (rmse_dt - rmse_rf) / rmse_dt * 100
        if imp_pct > 0:
            st.write(f"Random Forest memiliki RMSE **{imp_pct:.2f}% lebih rendah** dibandingkan Decision Tree.")
        elif imp_pct < 0:
            st.write(f"Decision Tree memiliki RMSE **{abs(imp_pct):.2f}% lebih rendah** dibandingkan Random Forest.")
        else:
            st.write("Kedua algoritma memiliki RMSE yang sama.")
    if best_r2 >= 0.7:
        st.write("Model memiliki kemampuan prediksi yang cukup baik.")
    elif best_r2 >= 0:
        st.write("Model memiliki kemampuan prediksi yang masih terbatas.")
    else:
        st.write("Model belum memiliki performa prediksi yang baik karena R² negatif.")

# ----- 5 percobaan -----
with tab_exp:
    T = R["trials"]
    st.markdown("### Pengujian Konsistensi: 5 Kali Percobaan")
    st.caption("Setiap percobaan memakai random state pembagian data yang berbeda: 10, 20, 30, 40, 50.")
    st.dataframe(T.round(4), width="stretch", hide_index=True)

    st.markdown("**Rata-rata hasil percobaan**")
    avg = T.drop(columns=["Percobaan", "Random State"]).mean().to_frame("Rata-rata")
    st.dataframe(avg.round(4), width="stretch")

    rf_menang = int((T["RMSE Random Forest"] < T["RMSE Decision Tree"]).sum())
    dt_menang = int((T["RMSE Decision Tree"] < T["RMSE Random Forest"]).sum())
    seri = 5 - rf_menang - dt_menang
    st.markdown("**Rekapitulasi hasil**")
    r1, r2, r3 = st.columns(3)
    r1.metric("Random Forest unggul", f"{rf_menang} kali")
    r2.metric("Decision Tree unggul", f"{dt_menang} kali")
    r3.metric("Hasil seri", f"{seri} kali")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(T["Percobaan"], T["RMSE Decision Tree"], marker="o", label="Decision Tree", color=COLOR_DT)
    ax.plot(T["Percobaan"], T["RMSE Random Forest"], marker="s", label="Random Forest", color=COLOR_RF)
    ax.set_title("Perbandingan RMSE dalam 5 Kali Percobaan")
    ax.set_xlabel("Percobaan")
    ax.set_ylabel("RMSE")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    dt_avg, rf_avg = T["RMSE Decision Tree"].mean(), T["RMSE Random Forest"].mean()
    st.markdown("### Kesimpulan Pengujian Konsistensi")
    st.write(f"Rata-rata RMSE Decision Tree: **{dt_avg:.4f}**")
    st.write(f"Rata-rata RMSE Random Forest: **{rf_avg:.4f}**")
    if rf_avg < dt_avg and rf_menang > dt_menang:
        st.success("Random Forest menunjukkan performa lebih konsisten. "
                   "Rata-rata RMSE lebih rendah dan lebih sering unggul.")
    elif rf_avg < dt_avg:
        st.warning("Random Forest memiliki rata-rata RMSE lebih rendah, namun keunggulannya "
                   "belum konsisten pada sebagian besar percobaan.")
    elif dt_avg < rf_avg:
        st.warning("Decision Tree memiliki rata-rata RMSE lebih rendah. "
                   "Random Forest belum menunjukkan keunggulan berdasarkan RMSE.")
    else:
        st.info("Kedua algoritma memiliki rata-rata RMSE yang sama.")
    st.caption("RMSE lebih rendah = kesalahan prediksi lebih kecil. "
               "R² negatif = kemampuan prediksi belum optimal. "
               "Hasil hanya menggambarkan lima pembagian data yang diuji.")
