import base64
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Decision Tree vs Random Forest",
    page_icon="🌳",
    layout="wide",
)

ACCENT = "#F26B1D"

# CSS hanya memakai warna aksen (oranye). Warna latar & teks sengaja TIDAK
# di-hardcode supaya selalu mengikuti tema Streamlit (.streamlit/config.toml),
# jadi teks tidak akan pernah "nabrak" dengan background.
st.markdown(
    f"""
    <style>
    h1 {{ color: {ACCENT} !important; font-weight: 800 !important; }}
    [data-testid="stSidebar"] {{ border-right: 3px solid {ACCENT}; }}
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{ color: {ACCENT} !important; }}
    [data-testid="stMetricValue"] {{ color: {ACCENT}; }}
    [data-testid="stSidebar"] code {{
        color: {ACCENT};
        background: rgba(242, 107, 29, 0.12);
        padding: 1px 6px;
        border-radius: 4px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Definisi kolom
# ---------------------------------------------------------------------------
CSV_NAME = "top_insta_influencers_data.csv"
NUM_COLS = [
    "posts",
    "followers",
    "avg_likes",
    "60_day_eng_rate",
    "new_post_avg_like",
    "total_likes",
]
CAT_COLS = ["country"]
FEATURES = NUM_COLS + CAT_COLS
TARGET = "influence_score"
REQUIRED = FEATURES + [TARGET]
SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}


# ---------------------------------------------------------------------------
# Load & bersihkan data
# ---------------------------------------------------------------------------
def to_number(value):
    """Ubah '3.3k', '475.8m', '29.0b', '1.39%' menjadi angka."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = str(value).strip().lower().replace(",", "")
    if s in ("", "-", "nan", "none"):
        return np.nan
    s = s.rstrip("%")
    mult = 1.0
    if s and s[-1] in SUFFIX:
        mult = SUFFIX[s[-1]]
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Kolom wajib tidak ada: " + ", ".join(missing))

    for col in NUM_COLS + [TARGET]:
        df[col] = df[col].map(to_number)
    df = df.dropna(subset=[TARGET])
    for col in NUM_COLS:
        df[col] = df[col].fillna(df[col].median())
    df["country"] = (
        df["country"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    )
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_default():
    """Dataset bawaan: cari CSV di repo, kalau tidak ada pakai versi tertanam."""
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
            f"{CSV_NAME} tidak ditemukan dan default_data.py juga tidak bisa dibaca "
            "(pastikan kedua file ada di folder yang sama dengan app.py)."
        ) from exc


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def make_pipeline(model):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", model)])


def grouped_importance(pipe):
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    out = {}
    for name, value in zip(names, imp):
        key = "country" if name.startswith("cat__") else name.replace("remainder__", "")
        out[key] = out.get(key, 0.0) + float(value)
    return pd.Series(out).sort_values(ascending=True)


@st.cache_data(show_spinner="Melatih model...")
def train_models(df, test_size, seed, n_estimators, dt_depth, rf_depth, leaf):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    models = {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=dt_depth, min_samples_leaf=leaf, random_state=seed
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=n_estimators, max_depth=rf_depth, min_samples_leaf=leaf,
            random_state=seed, n_jobs=-1,
        ),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    results, preds, pipes = [], {}, {}
    for name, model in models.items():
        pipe = make_pipeline(model).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_te, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                "R² Test": r2_score(y_te, pred),
                "R² Train": r2_score(y_tr, pipe.predict(X_tr)),
                "R² CV (5-fold)": cross_val_score(
                    make_pipeline(model), X, y, cv=cv, scoring="r2"
                ).mean(),
            }
        )
        preds[name] = pred
        pipes[name] = pipe
    return {
        "table": pd.DataFrame(results),
        "y_test": y_te.to_numpy(),
        "preds": preds,
        "pipes": pipes,
        "importance": {n: grouped_importance(p) for n, p in pipes.items()},
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Dataset")
    uploaded = st.file_uploader("Upload dataset CSV kamu sendiri (opsional)", type="csv")
    st.markdown(
        "Kalau tidak upload, web memakai **dataset bawaan** (Top Instagram "
        "Influencers). File upload harus punya kolom: "
        + ", ".join(f"`{c}`" for c in REQUIRED)
        + "."
    )

    st.header("⚙️ Pengaturan Model")
    test_size = st.slider("Porsi data uji", 0.10, 0.50, 0.20, 0.05)
    seed = st.number_input("Random state", 0, 9999, 42, 1)
    dt_depth = st.slider("Decision Tree: max depth", 1, 20, 3)
    n_estimators = st.slider("Random Forest: jumlah pohon", 10, 500, 200, 10)
    rf_depth = st.slider("Random Forest: max depth", 1, 30, 5)
    leaf = st.slider("Min sampel per daun (kedua model)", 1, 20, 5)

# ---------------------------------------------------------------------------
# Ambil data (otomatis)
# ---------------------------------------------------------------------------
try:
    if uploaded is not None:
        raw_df, source = pd.read_csv(uploaded, encoding="utf-8-sig"), f"upload: {uploaded.name}"
    else:
        raw_df, source = load_default()
    data = prepare(raw_df)
    if len(data) < 20:
        raise ValueError("Data terlalu sedikit (minimal 20 baris) untuk dilatih.")
except Exception as exc:  # noqa: BLE001
    st.title("🌳 Perbandingan Decision Tree dan Random Forest")
    st.error(f"Dataset tidak bisa diproses: {exc}")
    st.stop()

results = train_models(
    data, float(test_size), int(seed), int(n_estimators), int(dt_depth), int(rf_depth), int(leaf)
)
table = results["table"]

# ---------------------------------------------------------------------------
# Halaman utama
# ---------------------------------------------------------------------------
st.title("🌳 Perbandingan Decision Tree dan Random Forest")
st.markdown(
    "Prediksi **Influence Score** pada dataset Top Instagram Influencers. "
    f"Sumber data: {source} ({len(data)} baris)."
)

tab_data, tab_model, tab_pred = st.tabs(["📊 Data", "⚖️ Perbandingan Model", "🔮 Coba Prediksi"])

# ---- Tab Data -------------------------------------------------------------
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah baris", f"{len(data):,}")
    c2.metric("Jumlah negara", data["country"].nunique())
    c3.metric("Rata-rata score", f"{data[TARGET].mean():.1f}")
    c4.metric("Rentang score", f"{data[TARGET].min():.0f} – {data[TARGET].max():.0f}")

    st.subheader("Cuplikan data (setelah dibersihkan)")
    st.dataframe(data.head(20), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Distribusi Influence Score")
        fig = px.histogram(data, x=TARGET, nbins=20, color_discrete_sequence=[ACCENT])
        fig.update_layout(yaxis_title="Jumlah", xaxis_title="Influence score")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Korelasi fitur numerik")
        corr = data[NUM_COLS + [TARGET]].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="Oranges", aspect="auto"
        )
        st.plotly_chart(fig, width="stretch")

# ---- Tab Perbandingan -----------------------------------------------------
with tab_model:
    st.caption(
        f"Data latih: {results['n_train']} baris · data uji: {results['n_test']} baris. "
        "MAE dan RMSE makin kecil makin bagus, R² makin besar makin bagus."
    )
    st.dataframe(
        table.set_index("Model").style.format("{:.3f}"),
        width="stretch",
    )

    best = table.loc[table["RMSE"].idxmin(), "Model"]
    other = table.loc[table["Model"] != best, "Model"].iloc[0]
    b = table[table["Model"] == best].iloc[0]
    o = table[table["Model"] == other].iloc[0]
    if abs(b["RMSE"] - o["RMSE"]) / max(o["RMSE"], 1e-9) < 0.02:
        st.info(
            f"RMSE kedua model hampir sama ({b['RMSE']:.2f} vs {o['RMSE']:.2f}). "
            f"Dilihat dari MAE dan R² CV, "
            f"{table.loc[table['R² CV (5-fold)'].idxmax(), 'Model']} sedikit lebih stabil."
        )
    else:
        st.success(
            f"**{best}** punya error paling kecil pada data uji "
            f"(RMSE {b['RMSE']:.2f} vs {o['RMSE']:.2f} milik {other}; "
            f"R² {b['R² Test']:.3f} vs {o['R² Test']:.3f})."
        )
    if table["R² Test"].max() < 0.2:
        st.caption(
            "Catatan: R² rendah berarti Influence Score sulit ditebak dari fitur yang ada "
            "(datanya hanya 200 baris dan ada beberapa outlier). Ini temuan analisis, bukan error."
        )
    for _, row in table.iterrows():
        gap = row["R² Train"] - row["R² Test"]
        if gap > 0.3:
            st.warning(
                f"{row['Model']} terindikasi overfitting: R² latih {row['R² Train']:.2f} "
                f"jauh di atas R² uji {row['R² Test']:.2f}. Coba kecilkan max depth."
            )

    long = table.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metrik")
    left, right = st.columns(2)
    with left:
        st.subheader("Error (lebih kecil = lebih baik)")
        fig = px.bar(
            long, x="Metrik", y="value", color="Model", barmode="group",
            color_discrete_sequence=[ACCENT, "#4DA3FF"],
        )
        fig.update_layout(yaxis_title="Nilai")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("R² (lebih besar = lebih baik)")
        r2 = table.melt(
            id_vars="Model",
            value_vars=["R² Train", "R² Test", "R² CV (5-fold)"],
            var_name="Jenis",
        )
        fig = px.bar(
            r2, x="Jenis", y="value", color="Model", barmode="group",
            color_discrete_sequence=[ACCENT, "#4DA3FF"],
        )
        fig.update_layout(yaxis_title="R²")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Aktual vs prediksi (data uji)")
    cols = st.columns(2)
    lo, hi = float(results["y_test"].min()), float(results["y_test"].max())
    for col, name in zip(cols, results["preds"]):
        with col:
            fig = px.scatter(
                x=results["y_test"], y=results["preds"][name],
                labels={"x": "Aktual", "y": "Prediksi"},
                title=name, color_discrete_sequence=[ACCENT],
            )
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                          line=dict(dash="dash", color="#9AA0A6"))
            st.plotly_chart(fig, width="stretch")

    st.subheader("Fitur yang paling berpengaruh")
    cols = st.columns(2)
    for col, (name, imp) in zip(cols, results["importance"].items()):
        with col:
            fig = px.bar(
                x=imp.values, y=imp.index, orientation="h", title=name,
                labels={"x": "Importance", "y": ""},
                color_discrete_sequence=[ACCENT],
            )
            st.plotly_chart(fig, width="stretch")

# ---- Tab Prediksi ---------------------------------------------------------
with tab_pred:
    st.subheader("Coba prediksi Influence Score")
    st.caption("Isi angka asli (bukan format 3.3k / 475m). Nilai awal diambil dari median data.")
    med = data[NUM_COLS].median()
    countries = sorted(data["country"].unique())
    with st.form("form_prediksi"):
        a, b_, c = st.columns(3)
        posts = a.number_input("Jumlah posts", 0.0, value=float(med["posts"]), step=100.0)
        followers = b_.number_input("Followers", 0.0, value=float(med["followers"]), step=1e6, format="%.0f")
        avg_likes = c.number_input("Rata-rata likes", 0.0, value=float(med["avg_likes"]), step=1e5, format="%.0f")
        d, e, f = st.columns(3)
        eng = d.number_input("60-day engagement rate (%)", 0.0, value=float(med["60_day_eng_rate"]), step=0.1)
        new_like = e.number_input("Rata-rata likes post baru", 0.0, value=float(med["new_post_avg_like"]), step=1e5, format="%.0f")
        total = f.number_input("Total likes", 0.0, value=float(med["total_likes"]), step=1e8, format="%.0f")
        country = st.selectbox("Negara", countries, index=countries.index("United States") if "United States" in countries else 0)
        go_pred = st.form_submit_button("Prediksi")

    if go_pred:
        row = pd.DataFrame([{
            "posts": posts, "followers": followers, "avg_likes": avg_likes,
            "60_day_eng_rate": eng, "new_post_avg_like": new_like,
            "total_likes": total, "country": country,
        }])
        m1, m2 = st.columns(2)
        for col, (name, pipe) in zip((m1, m2), results["pipes"].items()):
            col.metric(name, f"{pipe.predict(row)[0]:.1f}")
