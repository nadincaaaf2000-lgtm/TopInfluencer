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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree

st.set_page_config(page_title="Top Influencer: Decision Tree vs Random Forest", page_icon="🌳", layout="wide")

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
CSV_NAME = "top_insta_influencers_data.csv"
NUMERIC_FEATURES = ["posts", "followers", "avg_likes", "60_day_eng_rate", "new_post_avg_like", "total_likes"]
CATEGORICAL_FEATURES = ["country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "influence_score"
TEST_SIZE, SEED, CV_FOLDS = 0.2, 42, 5
ACCENT, BLUE = "#F26B1D", "#2F7ED8"
MODEL_NAMES = ["Decision Tree (Baseline)", "Decision Tree (Tuned)", "Random Forest (Baseline)", "Random Forest (Tuned)"]
LABELS = {
    "posts": "Jumlah posts", "followers": "Followers", "avg_likes": "Rata-rata likes",
    "60_day_eng_rate": "60-day engagement rate (%)", "new_post_avg_like": "Rata-rata likes post baru",
    "total_likes": "Total likes", "country": "Negara",
}
DT_GRID = {
    "model__max_depth": [3, 5, 8, 12, None],
    "model__min_samples_split": [2, 4, 10],
    "model__min_samples_leaf": [1, 2, 4],
}
RF_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 4],
    "model__min_samples_leaf": [1, 2, 4],
}
    columns=["column", "description", "unit_or_type"],
)


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
        mult = 1
        if value.endswith("k"):
            mult, value = 1_000, value[:-1]
        elif value.endswith("m"):
            mult, value = 1_000_000, value[:-1]
        elif value.endswith("b"):
            mult, value = 1_000_000_000, value[:-1]
        return float(value.replace(",", "")) * mult
    except ValueError:
        return np.nan


@st.cache_data(show_spinner=False)
def load_default():
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
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    valid = df[TARGET].notna()
    df = df.loc[valid].reset_index(drop=True)
    return df, df[FEATURES].copy(), df[TARGET].copy()


def make_pipeline(model):
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocessor", pre), ("model", model)])


def scores(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {"MAE": mean_absolute_error(y_true, pred), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "R2 Score": r2_score(y_true, pred)}


@st.cache_resource(show_spinner="Melatih & tuning model (5-fold CV)...")
def run_analysis(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    specs = {
        "Decision Tree": (DecisionTreeRegressor(random_state=SEED), DT_GRID),
        "Random Forest": (RandomForestRegressor(random_state=SEED), RF_GRID),
    }
    models, tuning = {}, []
    for name, (est, grid) in specs.items():
        models[f"{name} (Baseline)"] = make_pipeline(clone(est)).fit(X_tr, y_tr)
        gs = GridSearchCV(make_pipeline(clone(est)), grid, cv=cv, n_jobs=-1,
                          scoring="neg_root_mean_squared_error").fit(X_tr, y_tr)
        models[f"{name} (Tuned)"] = gs.best_estimator_
        tuning.append({
            "model": name, "candidate_count": len(gs.cv_results_["params"]), "cv_folds": CV_FOLDS,
            "best_cv_rmse": round(-gs.best_score_, 4),
            "best_params": str({k.replace("model__", ""): v for k, v in gs.best_params_.items()}),
        })
    preds, perf = {}, []
    for m in MODEL_NAMES:
        preds[m] = models[m].predict(X_te)
        s = scores(y_te, preds[m])
        s["R2 Train"] = r2_score(y_tr, models[m].predict(X_tr))
        name, stage = m.replace(")", "").split(" (")
        perf.append({"Model": name, "Tahap": stage, **s})
    return {"models": models, "preds": preds, "perf": pd.DataFrame(perf), "tuning": pd.DataFrame(tuning),
            "y_test": y_te, "n_train": len(X_tr), "n_test": len(X_te)}


def importances(pipe):
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    idx = ["country" if n.startswith("cat__") else n.replace("num__", "") for n in names]
    return pd.Series(imp, index=idx).groupby(level=0).sum().sort_values()


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def scatter_fig(y_true, pred, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, pred, alpha=0.7, color=ACCENT)
    lim = [min(y_true.min(), pred.min()), max(y_true.max(), pred.max())]
    ax.plot(lim, lim, "--", color="#555555", label="Prediksi sempurna")
    ax.set_title(title)
    ax.set_xlabel("Nilai Aktual")
    ax.set_ylabel("Nilai Prediksi")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def importance_fig(pipe, title):
    s = importances(pipe)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(s.index, s.values, color=BLUE)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar: konfigurasi
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Konfigurasi Model")
    main_name = st.selectbox("Model Utama untuk Prediksi", MODEL_NAMES, index=3)
    comp_opts = ["Tidak ada"] + [m for m in MODEL_NAMES if m != main_name]
    comp_default = comp_opts.index("Decision Tree (Tuned)") if "Decision Tree (Tuned)" in comp_opts else 0
    comp_name = st.selectbox("Model Pembanding (opsional)", comp_opts, index=comp_default)
    with st.expander("📂 Dataset (opsional)"):
        uploaded = st.file_uploader("Upload CSV sendiri", type="csv")
        st.caption("Kolom wajib: " + ", ".join(f"`{c}`" for c in FEATURES + [TARGET]))

# ---------------------------------------------------------------------------
# Data & model
# ---------------------------------------------------------------------------
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

R = run_analysis(X, y)
perf = R["perf"]
main_pipe = R["models"][main_name]
comp_pipe = None if comp_name == "Tidak ada" else R["models"][comp_name]


def metrics_of(name):
    return perf.iloc[MODEL_NAMES.index(name)]


with st.sidebar:
    st.divider()
    st.subheader("Ringkasan Eksperimen")
    st.markdown(
        f"- **Data mentah:** {len(raw):,} baris\n"
        f"- **Data bersih:** {len(y):,} baris\n"
        f"- **Fitur:** {len(FEATURES)} fitur (6 numerik + negara)\n"
        f"- **Target:** influence_score\n"
        f"- **Split:** 80% train / 20% test\n"
        f"- **CV:** {CV_FOLDS}-fold KFold\n"
        f"- **Metrik utama:** RMSE"
    )

st.title("Prediksi Top Instagram Influencer: Decision Tree vs Random Forest")
st.caption(f"Dashboard interaktif untuk memprediksi Influence Score dan mengevaluasi performa model. Sumber data: {source}.")

tab_pred, tab_eval, tab_eda = st.tabs(["Prediksi Influence Score", "Evaluasi & Perbandingan Model", "EDA & Ringkasan Data"])

# ---------------------------------------------------------------------------
# Tab 1: Prediksi
# ---------------------------------------------------------------------------
with tab_pred:
    st.header("Prediksi Sampel Tunggal")
    med = X[NUMERIC_FEATURES].median()
    countries = sorted(X["country"].dropna().unique())
    names = df["channel_info"].astype(str).tolist() if "channel_info" in df.columns else [f"Baris {i}" for i in range(len(df))]

    c1, c2 = st.columns([1, 2])
    src = c1.radio("Sumber nilai awal fitur", ["Ambil contoh dari dataset", "Nilai median dataset"])
    idx = None
    if src.startswith("Ambil"):
        idx = c2.selectbox("Pilih influencer contoh", range(len(df)), format_func=lambda i: f"{names[i]}  ·  skor {y.iloc[i]:.0f}")
        base = X.iloc[idx][NUMERIC_FEATURES].fillna(med)
        base_country = X.iloc[idx]["country"]
    else:
        base, base_country = med, ("United States" if "United States" in countries else countries[0])
    if base_country not in countries:
        base_country = countries[0]
    sig = f"{src}_{idx}"

    with st.form("form_prediksi"):
        st.write("Sesuaikan nilai **7 fitur** berikut (isi angka asli, bukan format 3.3k / 475m):")
        cols = st.columns(4)
        vals = {}
        for k, feat in enumerate(NUMERIC_FEATURES):
            vals[feat] = cols[k % 4].number_input(LABELS[feat], min_value=0.0, value=float(base[feat]),
                                                  format="%.2f", key=f"in_{feat}_{sig}")
        vals["country"] = cols[2].selectbox(LABELS["country"], countries, index=countries.index(base_country),
                                            key=f"in_country_{sig}")
        go = st.form_submit_button("Prediksi Influence Score", type="primary")

    if go:
        st.session_state["pred"] = {"row": pd.DataFrame([vals]), "actual": float(y.iloc[idx]) if idx is not None else None}

    if "pred" in st.session_state:
        row, actual = st.session_state["pred"]["row"], st.session_state["pred"]["actual"]
        all_pred = {m: float(R["models"][m].predict(row)[0]) for m in MODEL_NAMES}
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Prediksi ({main_name})", f"{all_pred[main_name]:.2f}")
        m2.metric(f"Prediksi ({comp_name})" if comp_pipe is not None else "Model pembanding",
                  f"{all_pred[comp_name]:.2f}" if comp_pipe is not None else "-")
        m3.metric("Skor Asli Sampel", f"{actual:.2f}" if actual is not None else "-")

        left, right = st.columns([2, 1])
        out = pd.DataFrame({"Model": MODEL_NAMES, "Prediksi": [all_pred[m] for m in MODEL_NAMES]})
        if actual is not None:
            out["Selisih dari Aktual"] = (out["Prediksi"] - actual).abs()
        left.bar_chart(out.set_index("Model")["Prediksi"])
        right.dataframe(out.round(2), hide_index=True, width="stretch")
        if actual is not None:
            st.caption("Skor asli hanya ditampilkan jika nilai awal diambil dari dataset. Jika angka fitur diubah, "
                       "perbandingan dengan skor asli tidak lagi setara.")
        st.caption("Model dipilih berdasarkan RMSE pada data uji, bukan dari besar-kecilnya angka prediksi.")

# ---------------------------------------------------------------------------
# Tab 2: Evaluasi
# ---------------------------------------------------------------------------
with tab_eval:
    st.header("Performa Final pada Testing Set")
    mm = metrics_of(main_name)
    cols = st.columns(4)
    cols[0].metric(f"{main_name} RMSE", f"{mm['RMSE']:.4f}")
    cols[1].metric(f"{main_name} R²", f"{mm['R2 Score']:.4f}")
    if comp_pipe is not None:
        mc = metrics_of(comp_name)
        cols[2].metric(f"{comp_name} RMSE", f"{mc['RMSE']:.4f}")
        cols[3].metric(f"{comp_name} R²", f"{mc['R2 Score']:.4f}")
    st.dataframe(perf.round(4), hide_index=True, width="stretch")

    best = perf.sort_values("RMSE").iloc[0]
    st.markdown(f"Model terbaik berdasarkan RMSE: **{best['Model']} ({best['Tahap']})** · RMSE **{best['RMSE']:.4f}** · R² **{best['R2 Score']:.4f}**")
    if best["R2 Score"] >= 0.7:
        st.success("Performa prediksi model terbaik tergolong baik (R² ≥ 0,7).")
    elif best["R2 Score"] >= 0:
        st.warning("Performa prediksi model terbaik masih terbatas (R² antara 0 dan 0,7).")
    else:
        st.warning("Model terbaik menurut RMSE pun belum optimal karena R² negatif.")

    st.subheader("Perbandingan Baseline vs Tuned")
    fig, ax = plt.subplots(figsize=(8, 4))
    perf.pivot(index="Model", columns="Tahap", values="RMSE")[["Baseline", "Tuned"]].plot(kind="bar", ax=ax, color=[BLUE, ACCENT])
    ax.set_ylabel("RMSE")
    ax.set_title("Perbandingan RMSE: Baseline vs Tuned")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    show(fig)

    st.subheader(f"Ringkasan Hyperparameter Tuning ({CV_FOLDS}-Fold CV)")
    st.dataframe(R["tuning"], hide_index=True, width="stretch")

    st.subheader("Visualisasi Evaluasi")
    a, b = st.columns(2)
    with a:
        show(scatter_fig(R["y_test"], R["preds"][main_name], f"Aktual vs Prediksi - {main_name}"))
    with b:
        if comp_pipe is not None:
            show(scatter_fig(R["y_test"], R["preds"][comp_name], f"Aktual vs Prediksi - {comp_name}"))
        else:
            st.info("Pilih model pembanding di sidebar untuk melihat plot kedua.")
    a, b = st.columns(2)
    with a:
        if hasattr(main_pipe.named_steps["model"], "feature_importances_"):
            show(importance_fig(main_pipe, f"Feature Importance - {main_name}"))
    with b:
        if comp_pipe is not None:
            show(importance_fig(comp_pipe, f"Feature Importance - {comp_name}"))

    st.subheader("Struktur Decision Tree (Tuned, kedalaman 3)")
    dt = R["models"]["Decision Tree (Tuned)"]
    fig = plt.figure(figsize=(22, 9))
    plot_tree(dt.named_steps["model"], feature_names=dt.named_steps["preprocessor"].get_feature_names_out(),
              filled=True, rounded=True, max_depth=3, fontsize=8)
    show(fig)

# ---------------------------------------------------------------------------
# Tab 3: EDA
# ---------------------------------------------------------------------------
with tab_eda:
    st.header("Ringkasan & Catatan Eksperimen")
    st.markdown(
        f"- Observasi mentah: {len(raw):,}\n"
        f"- Observasi bersih (influence_score valid): {len(df):,}\n"
        f"- Fitur: {len(FEATURES)} ({len(NUMERIC_FEATURES)} numerik, {len(CATEGORICAL_FEATURES)} kategorik)\n"
        f"- Training/testing: {R['n_train']:,}/{R['n_test']:,} (80:20)\n"
        f"- Cross-validation: {CV_FOLDS}-fold KFold · Metrik tuning: RMSE\n"
        f"- Kandidat Decision Tree: {R['tuning'].iloc[0]['candidate_count']} · Random Forest: {R['tuning'].iloc[1]['candidate_count']}\n"
        f"- Model terbaik berdasarkan RMSE testing: {best['Model']} ({best['Tahap']})"
    )
    with st.expander("Catatan Interpretasi & Batasan Metodologi"):
        st.markdown(
            "- Dataset relatif kecil, sehingga hasil sensitif terhadap pembagian data.\n"
            "- Nilai 3.3k / 475m / 1.2b diubah menjadi angka penuh sebelum pemodelan.\n"
            "- Nilai kosong diisi median (numerik) atau modus (negara).\n"
            "- Hasil hanya berlaku pada dataset, split, random state, dan grid parameter yang diuji."
        )

    st.subheader("Pratinjau Dataset (50 Baris Pertama)")
    st.dataframe(df.head(50), width="stretch", height=380)

    dfn = df[NUMERIC_FEATURES].copy()
    dfn[TARGET] = df[TARGET]
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(df[TARGET], bins=25, color=BLUE, edgecolor="white")
        ax.set_title("Distribusi Influence Score")
        ax.set_xlabel("Influence Score")
        ax.set_ylabel("Jumlah influencer")
        fig.tight_layout()
        show(fig)
    with b:
        corr = dfn.corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, cmap="viridis", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)))
        ax.set_yticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if corr.iloc[i, j] < 0.5 else "black")
        fig.colorbar(im, ax=ax, label="Koefisien korelasi")
        ax.set_title("Matriks Korelasi Pearson")
        fig.tight_layout()
        show(fig)

    st.subheader("Distribusi Fitur Numerik")
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        ax.hist(dfn[col].dropna(), bins=30, color=BLUE)
        ax.set_title(col)
    fig.tight_layout()
    show(fig)

    st.subheader("10 Negara dengan Influencer Terbanyak")
    top = df["country"].value_counts().head(10)
    a, b = st.columns([2, 1])
    a.bar_chart(top)
    b.dataframe(top.rename("jumlah").reset_index(), hide_index=True, width="stretch")

    st.subheader("Statistik Deskriptif Fitur")
    st.dataframe(dfn.describe().T.round(4), width="stretch")
