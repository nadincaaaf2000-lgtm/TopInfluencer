import base64
import gzip
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree

st.set_page_config(page_title="Decision Tree vs Random Forest", page_icon="🌳", layout="wide")

ACCENT = "#F26B1D"
BLUE = "#2F7ED8"

# CSS hanya memakai warna aksen. Warna latar & teks mengikuti tema Streamlit
# (.streamlit/config.toml) sehingga tidak pernah bentrok.
st.markdown(
    f"""
    <style>
    h1 {{ color: {ACCENT} !important; font-weight: 800 !important; }}
    [data-testid="stSidebar"] {{ border-right: 3px solid {ACCENT}; }}
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {ACCENT} !important; }}
    [data-testid="stMetricValue"] {{ color: {ACCENT}; }}
    [data-testid="stSidebar"] code {{
        color: {ACCENT}; background: rgba(242,107,29,0.12);
        padding: 1px 6px; border-radius: 4px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Konstanta (sama dengan kode Colab)
# ---------------------------------------------------------------------------
CSV_NAME = "top_insta_influencers_data.csv"
NUMERIC_FEATURES = ["posts", "followers", "avg_likes", "60_day_eng_rate", "new_post_avg_like", "total_likes"]
CATEGORICAL_FEATURES = ["country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "influence_score"
TEST_SIZE = 0.2
SEED = 42
EXPERIMENT_SEEDS = [10, 20, 30, 40, 50]
DT_PARAMS = dict(max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42)
RF_PARAMS = dict(n_estimators=100, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42, n_jobs=-1)


# ---------------------------------------------------------------------------
# Load & preprocessing
# ---------------------------------------------------------------------------
def convert_value(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if value == "":
        return np.nan
    try:
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
    except ValueError:
        return np.nan


@st.cache_data(show_spinner=False)
def load_default():
    """CSV di repo dulu; kalau tidak ada, pakai salinan yang tertanam di default_data.py."""
    here = Path(__file__).resolve().parent
    for path in (here / CSV_NAME, here / "data" / CSV_NAME, Path.cwd() / CSV_NAME):
        if path.exists():
            return pd.read_csv(path, encoding="utf-8-sig"), f"file {path.name}"
    try:
        from default_data import CSV_GZ_B64

        raw = gzip.decompress(base64.b64decode(CSV_GZ_B64))
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig"), "dataset bawaan"
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"{CSV_NAME} tidak ditemukan dan default_data.py tidak bisa dibaca. "
            "Letakkan salah satunya di folder yang sama dengan app.py."
        ) from exc


def prepare(raw):
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in df.columns]
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Kolom wajib tidak ada: " + ", ".join(missing))
    for col in NUMERIC_FEATURES:
        df[col] = df[col].apply(convert_value)
    X = df[FEATURES].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.notna()
    return df, X.loc[valid], y.loc[valid]


def make_pipeline(model):
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocessor", pre), ("model", model)])


def scores(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {
        "MAE": mean_absolute_error(y_true, pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2 Score": r2_score(y_true, pred),
    }


def result_table(y_true, pred):
    t = pd.DataFrame({"Nilai Aktual": y_true.values, "Nilai Prediksi": pred})
    t["Selisih"] = (t["Nilai Aktual"] - t["Nilai Prediksi"]).abs()
    t["Persentase Prediksi"] = np.where(
        t["Nilai Aktual"] != 0, t["Nilai Prediksi"] / t["Nilai Aktual"] * 100, np.nan
    )
    return t.round(2)


@st.cache_data(show_spinner="Melatih model...")
def run_analysis(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    out = {"n_train": len(X_train), "n_test": len(X_test), "y_test": y_test}

    dt = make_pipeline(DecisionTreeRegressor(**DT_PARAMS)).fit(X_train, y_train)
    rf = make_pipeline(RandomForestRegressor(**RF_PARAMS)).fit(X_train, y_train)
    out["dt"], out["rf"] = dt, rf
    out["pred_dt"], out["pred_rf"] = dt.predict(X_test), rf.predict(X_test)
    out["s_dt"], out["s_rf"] = scores(y_test, out["pred_dt"]), scores(y_test, out["pred_rf"])

    rows = []
    for no, seed in enumerate(EXPERIMENT_SEEDS, start=1):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
        a = scores(yb, make_pipeline(DecisionTreeRegressor(**DT_PARAMS)).fit(Xa, ya).predict(Xb))
        b = scores(yb, make_pipeline(RandomForestRegressor(**RF_PARAMS)).fit(Xa, ya).predict(Xb))
        rows.append(
            {
                "Percobaan": no, "Random State": seed,
                "MAE Decision Tree": a["MAE"], "RMSE Decision Tree": a["RMSE"], "R2 Decision Tree": a["R2 Score"],
                "MAE Random Forest": b["MAE"], "RMSE Random Forest": b["RMSE"], "R2 Random Forest": b["R2 Score"],
            }
        )
    out["experiments"] = pd.DataFrame(rows)
    return out


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def tree_figure(model, names, title):
    fig = plt.figure(figsize=(25, 12))
    plot_tree(model, feature_names=names, filled=True, rounded=True, max_depth=3, fontsize=9)
    plt.title(title, fontsize=16)
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Dataset")
    uploaded = st.file_uploader("Upload dataset CSV kamu sendiri (opsional)", type="csv")
    st.markdown(
        "Kalau tidak upload, web memakai **dataset bawaan** (Top Instagram Influencers). "
        "File upload harus punya kolom: "
        + ", ".join(f"`{c}`" for c in FEATURES + [TARGET])
        + "."
    )
    st.header("⚙️ Parameter model")
    st.markdown(
        f"Sama dengan kode Colab:\n\n"
        f"- Data uji **{int(TEST_SIZE * 100)}%**, random state **{SEED}**\n"
        f"- Decision Tree: max depth **5**, min split **4**, min leaf **2**\n"
        f"- Random Forest: **100** pohon, max depth **5**, min split **4**, min leaf **2**"
    )

# ---------------------------------------------------------------------------
# Data (otomatis)
# ---------------------------------------------------------------------------
st.title("🌳 Perbandingan Decision Tree dan Random Forest")
try:
    if uploaded is not None:
        raw, source = pd.read_csv(uploaded, encoding="utf-8-sig"), f"upload: {uploaded.name}"
    else:
        raw, source = load_default()
    df, X, y = prepare(raw)
    if len(y) < 20:
        raise ValueError("Data terlalu sedikit (minimal 20 baris bernilai influence_score).")
except Exception as exc:  # noqa: BLE001
    st.error(f"Dataset tidak bisa diproses: {exc}")
    st.stop()

st.markdown(f"Prediksi **Influence Score** pada dataset Top Instagram Influencers. Sumber data: {source}.")
R = run_analysis(X, y)
s_dt, s_rf = R["s_dt"], R["s_rf"]

tabs = st.tabs(
    ["📊 Dataset", "🌲 Decision Tree", "🌳 Random Forest", "⚖️ Perbandingan & Kesimpulan", "🔁 Konsistensi 5 Percobaan", "🔮 Coba Prediksi"]
)

# ---- Dataset --------------------------------------------------------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah baris", df.shape[0])
    c2.metric("Jumlah kolom", df.shape[1])
    c3.metric("Data latih / uji", f"{R['n_train']} / {R['n_test']}")
    st.subheader("Cuplikan data")
    st.dataframe(df.head(), width="stretch")

# ---- Decision Tree / Random Forest ---------------------------------------
def model_tab(label, pipe, pred, s, tree_title, tree_model):
    st.subheader(f"Hasil evaluasi {label}")
    m = st.columns(4)
    for col, key in zip(m, ["MAE", "MSE", "RMSE", "R2 Score"]):
        col.metric(key, f"{s[key]:.4f}")
    st.subheader(f"Tabel hasil prediksi {label}")
    st.dataframe(result_table(R["y_test"], pred), width="stretch", height=320)
    st.subheader("Visualisasi pohon")
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    show(tree_figure(tree_model, names, tree_title))
    return names


with tabs[1]:
    model_tab(
        "Decision Tree", R["dt"], R["pred_dt"], s_dt,
        "Decision Tree untuk Prediksi Influence Score", R["dt"].named_steps["model"],
    )

with tabs[2]:
    rf_names = model_tab(
        "Random Forest", R["rf"], R["pred_rf"], s_rf,
        "Salah Satu Pohon dalam Random Forest", R["rf"].named_steps["model"].estimators_[0],
    )
    st.subheader("Fitur paling berpengaruh (Top 10)")
    imp = pd.DataFrame(
        {"Fitur": rf_names, "Importance": R["rf"].named_steps["model"].feature_importances_}
    ).sort_values("Importance", ascending=False)
    st.dataframe(imp.head(10).round(4), width="stretch")
    top = imp.head(10).sort_values("Importance")
    fig = plt.figure(figsize=(10, 6))
    plt.barh(top["Fitur"], top["Importance"], color=ACCENT)
    plt.title("Top 10 Feature Importance Random Forest")
    plt.xlabel("Nilai Importance")
    plt.ylabel("Fitur")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    show(fig)

# ---- Perbandingan & Kesimpulan -------------------------------------------
with tabs[3]:
    comp = pd.DataFrame(
        {
            "Algoritma": ["Decision Tree", "Random Forest"],
            **{k: [s_dt[k], s_rf[k]] for k in ["MAE", "MSE", "RMSE", "R2 Score"]},
        }
    )
    st.subheader("Tabel perbandingan kedua algoritma")
    st.dataframe(comp.round(4), width="stretch", hide_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["MAE", "MSE", "RMSE"]):
        ax.bar(comp["Algoritma"], comp[metric], color=[ACCENT, BLUE])
        ax.set_title("Perbandingan " + metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Perbandingan Error Decision Tree dan Random Forest")
    plt.tight_layout()
    show(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.bar(comp["Algoritma"], comp["R2 Score"], color=[ACCENT, BLUE])
    plt.title("Perbandingan R2 Score")
    plt.xlabel("Algoritma")
    plt.ylabel("R2 Score")
    plt.grid(axis="y", alpha=0.3)
    show(fig)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(R["y_test"].values, label="Nilai Aktual", marker="o", color="#555555")
    plt.plot(R["pred_dt"], label="Decision Tree", marker="x", color=ACCENT)
    plt.plot(R["pred_rf"], label="Random Forest", marker="s", color=BLUE)
    plt.title("Perbandingan Nilai Aktual dan Hasil Prediksi")
    plt.xlabel("Data Testing")
    plt.ylabel("Influence Score")
    plt.legend()
    plt.grid(alpha=0.3)
    show(fig)

    st.subheader("Kesimpulan hasil perbandingan")
    rmse_dt, rmse_rf = s_dt["RMSE"], s_rf["RMSE"]
    if rmse_dt < rmse_rf:
        best_name, best_rmse, best_r2 = "Decision Tree", rmse_dt, s_dt["R2 Score"]
    elif rmse_rf < rmse_dt:
        best_name, best_rmse, best_r2 = "Random Forest", rmse_rf, s_rf["R2 Score"]
    else:
        best_name, best_rmse, best_r2 = "Keduanya memiliki RMSE yang sama", rmse_dt, max(s_dt["R2 Score"], s_rf["R2 Score"])
    st.markdown(
        f"Algoritma terbaik berdasarkan RMSE: **{best_name}**  \n"
        f"RMSE terbaik: **{best_rmse:.4f}** · R2 Score: **{best_r2:.4f}**"
    )
    if rmse_dt > 0:
        imp_pct = (rmse_dt - rmse_rf) / rmse_dt * 100
        if imp_pct > 0:
            st.write(f"Random Forest memiliki RMSE {imp_pct:.2f}% lebih rendah dibandingkan Decision Tree.")
        elif imp_pct < 0:
            st.write(f"Decision Tree memiliki RMSE {abs(imp_pct):.2f}% lebih rendah dibandingkan Random Forest.")
        else:
            st.write("Kedua algoritma memiliki RMSE yang sama.")
   if best_r2 >= 0.7:
    st.success(
        f"{best_name} memiliki performa prediksi yang baik "
        f"dengan nilai R² sebesar {best_r2:.4f}."
    )
elif best_r2 >= 0:
    st.warning(
        f"{best_name} memiliki performa prediksi yang masih terbatas "
        f"dengan nilai R² sebesar {best_r2:.4f}."
    )
else:
    st.warning(
        f"{best_name} merupakan algoritma terbaik berdasarkan RMSE, "
        f"tetapi kemampuan prediksinya masih belum optimal "
        f"karena nilai R² sebesar {best_r2:.4f}."
    )

# ---- Konsistensi ----------------------------------------------------------
with tabs[4]:
    E = R["experiments"]
    st.subheader("Tabel hasil 5 kali percobaan")
    st.dataframe(E.round(4), width="stretch", hide_index=True)

    st.subheader("Rata-rata hasil percobaan")
    st.dataframe(E.drop(columns=["Percobaan", "Random State"]).mean().to_frame("Rata-rata").round(4), width="stretch")

    rf_win = int((E["RMSE Random Forest"] < E["RMSE Decision Tree"]).sum())
    dt_win = int((E["RMSE Decision Tree"] < E["RMSE Random Forest"]).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest unggul", f"{rf_win} kali")
    c2.metric("Decision Tree unggul", f"{dt_win} kali")
    c3.metric("Hasil seri", f"{5 - rf_win - dt_win} kali")

    fig = plt.figure(figsize=(10, 5))
    plt.plot(E["Percobaan"], E["RMSE Decision Tree"], marker="o", label="Decision Tree", color=ACCENT)
    plt.plot(E["Percobaan"], E["RMSE Random Forest"], marker="s", label="Random Forest", color=BLUE)
    plt.title("Perbandingan RMSE dalam 5 Kali Percobaan")
    plt.xlabel("Percobaan")
    plt.ylabel("RMSE")
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.grid(alpha=0.3)
    show(fig)

    dt_avg, rf_avg = E["RMSE Decision Tree"].mean(), E["RMSE Random Forest"].mean()
    st.subheader("Kesimpulan pengujian konsistensi")
    st.markdown(f"Rata-rata RMSE Decision Tree: **{dt_avg:.4f}**  \nRata-rata RMSE Random Forest: **{rf_avg:.4f}**")
    if rf_avg < dt_avg and rf_win > dt_win:
        st.success("Random Forest menunjukkan performa lebih konsisten. Rata-rata RMSE lebih rendah dan lebih sering unggul.")
    elif rf_avg < dt_avg:
        st.info("Random Forest memiliki rata-rata RMSE lebih rendah. Namun, keunggulannya belum konsisten pada sebagian besar percobaan.")
    elif dt_avg < rf_avg:
        st.info("Decision Tree memiliki rata-rata RMSE lebih rendah. Random Forest belum menunjukkan keunggulan berdasarkan RMSE.")
    else:
        st.info("Kedua algoritma memiliki rata-rata RMSE yang sama.")
    st.caption(
        "RMSE lebih rendah menunjukkan kesalahan prediksi lebih kecil. R2 negatif menunjukkan kemampuan "
        "prediksi masih belum optimal. Hasil ini hanya menggambarkan lima pembagian data yang diuji."
    )

# ---- Coba prediksi --------------------------------------------------------
with tabs[5]:
    st.subheader("Coba prediksi Influence Score")
    st.caption("Isi angka asli (bukan format 3.3k / 475m). Nilai awal diambil dari median data.")
    med = X[NUMERIC_FEATURES].median()
    countries = sorted(X["country"].dropna().unique())
    with st.form("form_prediksi"):
        a, b, c = st.columns(3)
        posts = a.number_input("Jumlah posts", 0.0, value=float(med["posts"]), step=100.0)
        followers = b.number_input("Followers", 0.0, value=float(med["followers"]), step=1e6, format="%.0f")
        avg_likes = c.number_input("Rata-rata likes", 0.0, value=float(med["avg_likes"]), step=1e5, format="%.0f")
        d, e, f = st.columns(3)
        eng = d.number_input("60-day engagement rate (%)", 0.0, value=float(med["60_day_eng_rate"]), step=0.1)
        new_like = e.number_input("Rata-rata likes post baru", 0.0, value=float(med["new_post_avg_like"]), step=1e5, format="%.0f")
        total = f.number_input("Total likes", 0.0, value=float(med["total_likes"]), step=1e8, format="%.0f")
        country = st.selectbox(
            "Negara", countries, index=countries.index("United States") if "United States" in countries else 0
        )
        go_pred = st.form_submit_button("Prediksi")
    if go_pred:
        row = pd.DataFrame(
            [{"posts": posts, "followers": followers, "avg_likes": avg_likes, "60_day_eng_rate": eng,
              "new_post_avg_like": new_like, "total_likes": total, "country": country}]
        )
        m1, m2 = st.columns(2)
        m1.metric("Decision Tree", f"{R['dt'].predict(row)[0]:.1f}")
        m2.metric("Random Forest", f"{R['rf'].predict(row)[0]:.1f}")
