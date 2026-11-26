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


st.set_page_config(
    page_title="Top Influencer: Decision Tree vs Random Forest",
    page_icon="🌳",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
CSV_NAME = "top_insta_influencers_data.csv"

NUMERIC_FEATURES = [
    "posts",
    "followers",
    "avg_likes",
    "60_day_eng_rate",
    "new_post_avg_like",
    "total_likes",
]
CATEGORICAL_FEATURES = ["country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "influence_score"

TEST_SIZE = 0.2
SEED = 42
CV_FOLDS = 5

ACCENT = "#F26B1D"
BLUE = "#2F7ED8"

MODEL_NAMES = [
    "Decision Tree (Baseline)",
    "Decision Tree (Tuned)",
    "Random Forest (Baseline)",
    "Random Forest (Tuned)",
]

LABELS = {
    "posts": "Jumlah posts",
    "followers": "Followers",
    "avg_likes": "Rata-rata likes",
    "60_day_eng_rate": "60-day engagement rate (%)",
    "new_post_avg_like": "Rata-rata likes post baru",
    "total_likes": "Total likes",
    "country": "Negara",
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


# ---------------------------------------------------------------------------
# Load dan preprocessing
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
    here = Path(__file__).resolve().parent

    possible_paths = [
        here / CSV_NAME,
        here / "data" / CSV_NAME,
        Path.cwd() / CSV_NAME,
    ]

    for path in possible_paths:
        if path.exists():
            return (
                pd.read_csv(path, encoding="utf-8-sig"),
                f"file {path.name}",
            )

    try:
        from default_data import CSV_GZ_B64

        raw = gzip.decompress(base64.b64decode(CSV_GZ_B64))
        return (
            pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig"),
            "dataset bawaan",
        )

    except Exception as exc:
        raise FileNotFoundError(
            f"{CSV_NAME} tidak ditemukan dan default_data.py tidak bisa dibaca. "
            "Letakkan salah satunya di folder yang sama dengan app.py."
        ) from exc


def prepare(raw):
    df = raw.copy()

    df.columns = [
        str(column).strip().lower().replace("\ufeff", "")
        for column in df.columns
    ]

    missing = [
        column
        for column in FEATURES + [TARGET]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Kolom wajib tidak ada: " + ", ".join(missing)
        )

    # Ubah fitur numerik, termasuk format seperti 3.3k, 475m, dan 12%.
    for column in NUMERIC_FEATURES:
        df[column] = df[column].apply(convert_value)

        # Mencegah kolom yang seluruh nilainya kosong menyebabkan error
        # saat proses imputasi median.
        if df[column].isna().all():
            df[column] = 0.0

    # Rapikan kolom negara dan samakan tipe datanya.
    df["country"] = df["country"].astype("string").str.strip()
    df["country"] = df["country"].replace("", pd.NA)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    # Baris tanpa target tidak dapat digunakan untuk pelatihan.
    df = df.loc[df[TARGET].notna()].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Tidak ada baris dengan influence_score yang valid."
        )

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return df, X, y


def make_pipeline(model):
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        )
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="Tidak diketahui",
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def scores(y_true, prediction):
    mse = mean_squared_error(y_true, prediction)

    return {
        "MAE": mean_absolute_error(y_true, prediction),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2 Score": r2_score(y_true, prediction),
    }


# ---------------------------------------------------------------------------
# Pelatihan model dan hyperparameter tuning
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Melatih & tuning model (5-fold CV)...")
def run_analysis(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    specs = {
        "Decision Tree": (
            DecisionTreeRegressor(random_state=SEED),
            DT_GRID,
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=SEED),
            RF_GRID,
        ),
    }

    models = {}
    tuning = []

    for name, (estimator, grid) in specs.items():

        # Model baseline
        baseline = make_pipeline(clone(estimator))
        baseline.fit(X_train, y_train)
        models[f"{name} (Baseline)"] = baseline

        # Hyperparameter tuning
        search = GridSearchCV(
            estimator=make_pipeline(clone(estimator)),
            param_grid=grid,
            cv=cv,
            n_jobs=1,
            scoring="neg_root_mean_squared_error",
        )

        search.fit(X_train, y_train)

        models[f"{name} (Tuned)"] = search.best_estimator_

        tuning.append(
            {
                "model": name,
                "candidate_count": len(search.cv_results_["params"]),
                "cv_folds": CV_FOLDS,
                "best_cv_rmse": round(-search.best_score_, 4),
                "best_params": str(
                    {
                        key.replace("model__", ""): value
                        for key, value in search.best_params_.items()
                    }
                ),
            }
        )

    predictions = {}
    performance = []

    for model_name in MODEL_NAMES:
        prediction = models[model_name].predict(X_test)
        predictions[model_name] = prediction

        metric = scores(y_test, prediction)
        metric["R2 Train"] = r2_score(
            y_train,
            models[model_name].predict(X_train),
        )

        name, stage = model_name.replace(")", "").split(" (")

        performance.append(
            {
                "Model": name,
                "Tahap": stage,
                **metric,
            }
        )

    return {
        "models": models,
        "preds": predictions,
        "perf": pd.DataFrame(performance),
        "tuning": pd.DataFrame(tuning),
        "y_test": y_test,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ---------------------------------------------------------------------------
# Visualisasi
# ---------------------------------------------------------------------------
def importances(pipe):
    feature_names = (
        pipe.named_steps["preprocessor"].get_feature_names_out()
    )
    importance_values = (
        pipe.named_steps["model"].feature_importances_
    )

    labels = [
        "country" if name.startswith("cat__")
        else name.replace("num__", "")
        for name in feature_names
    ]

    return (
        pd.Series(importance_values, index=labels)
        .groupby(level=0)
        .sum()
        .sort_values()
    )


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def scatter_fig(y_true, prediction, title):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        y_true,
        prediction,
        alpha=0.7,
        color=ACCENT,
    )

    limits = [
        min(y_true.min(), prediction.min()),
        max(y_true.max(), prediction.max()),
    ]

    ax.plot(
        limits,
        limits,
        "--",
        color="#555555",
        label="Prediksi sempurna",
    )

    ax.set_title(title)
    ax.set_xlabel("Nilai Aktual")
    ax.set_ylabel("Nilai Prediksi")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def importance_fig(pipe, title):
    series = importances(pipe)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.barh(
        series.index,
        series.values,
        color=BLUE,
    )

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

    main_name = st.selectbox(
        "Model Utama untuk Prediksi",
        MODEL_NAMES,
        index=3,
    )

    comparison_options = ["Tidak ada"] + [
        model
        for model in MODEL_NAMES
        if model != main_name
    ]

    if "Decision Tree (Tuned)" in comparison_options:
        comparison_default = comparison_options.index(
            "Decision Tree (Tuned)"
        )
    else:
        comparison_default = 0

    comp_name = st.selectbox(
        "Model Pembanding (opsional)",
        comparison_options,
        index=comparison_default,
    )

    with st.expander("📂 Dataset (opsional)"):
        uploaded = st.file_uploader(
            "Upload CSV sendiri",
            type="csv",
        )

        st.caption(
            "Kolom wajib: "
            + ", ".join(f"`{column}`" for column in FEATURES + [TARGET])
        )


# ---------------------------------------------------------------------------
# Data dan model
# ---------------------------------------------------------------------------
try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded, encoding="utf-8-sig")
        source = f"upload: {uploaded.name}"
    else:
        raw, source = load_default()

    df, X, y = prepare(raw)

    if len(y) < 20:
        raise ValueError(
            "Data terlalu sedikit (minimal 20 baris "
            "bernilai influence_score)."
        )

except Exception as exc:
    st.error(f"Dataset tidak bisa diproses: {exc}")
    st.stop()


try:
    R = run_analysis(X, y)

except Exception as exc:
    st.error(f"Proses pelatihan model gagal: {exc}")
    st.exception(exc)
    st.stop()


perf = R["perf"]
main_pipe = R["models"][main_name]

comp_pipe = (
    None
    if comp_name == "Tidak ada"
    else R["models"][comp_name]
)


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


# ---------------------------------------------------------------------------
# Judul dan tab dashboard
# ---------------------------------------------------------------------------
st.title(
    "Prediksi Top Instagram Influencer: Decision Tree vs Random Forest"
)

st.caption(
    "Dashboard interaktif untuk memprediksi Influence Score "
    f"dan mengevaluasi performa model. Sumber data: {source}."
)

tab_pred, tab_eval, tab_eda = st.tabs(
    [
        "Prediksi Influence Score",
        "Evaluasi & Perbandingan Model",
        "EDA & Ringkasan Data",
    ]
)


# ---------------------------------------------------------------------------
# Tab 1: Prediksi
# ---------------------------------------------------------------------------
with tab_pred:
    st.header("Prediksi Sampel Tunggal")

    median_values = X[NUMERIC_FEATURES].median().fillna(0)

    countries = sorted(
        X["country"].dropna().astype(str).unique().tolist()
    )

    # Jika semua nilai country kosong, tetap sediakan satu pilihan.
    if not countries:
        countries = ["Tidak diketahui"]

    if "channel_info" in df.columns:
        influencer_names = df["channel_info"].astype(str).tolist()
    else:
        influencer_names = [
            f"Baris {i}" for i in range(len(df))
        ]

    col1, col2 = st.columns([1, 2])

    source_choice = col1.radio(
        "Sumber nilai awal fitur",
        [
            "Ambil contoh dari dataset",
            "Nilai median dataset",
        ],
    )

    idx = None

    if source_choice.startswith("Ambil"):
        idx = col2.selectbox(
            "Pilih influencer contoh",
            range(len(df)),
            format_func=lambda i: (
                f"{influencer_names[i]}  ·  skor {y.iloc[i]:.0f}"
            ),
        )

        base = X.iloc[idx][NUMERIC_FEATURES].fillna(median_values)
        base_country = X.iloc[idx]["country"]

    else:
        base = median_values
        base_country = (
            "United States"
            if "United States" in countries
            else countries[0]
        )

    if pd.isna(base_country) or str(base_country) not in countries:
        base_country = countries[0]
    else:
        base_country = str(base_country)

    signature = f"{source_choice}_{idx}"

    with st.form("form_prediksi"):
        st.write(
            "Sesuaikan nilai **7 fitur** berikut "
            "(isi angka asli, bukan format 3.3k / 475m):"
        )

        columns = st.columns(4)
        values = {}

        for k, feature in enumerate(NUMERIC_FEATURES):
            values[feature] = columns[k % 4].number_input(
                LABELS[feature],
                min_value=0.0,
                value=float(base[feature]),
                format="%.2f",
                key=f"in_{feature}_{signature}",
            )

        values["country"] = columns[2].selectbox(
            LABELS["country"],
            countries,
            index=countries.index(base_country),
            key=f"in_country_{signature}",
        )

        go = st.form_submit_button(
            "Prediksi Influence Score",
            type="primary",
        )

    if go:
        actual_value = (
            float(y.iloc[idx])
            if idx is not None
            else None
        )

        st.session_state["pred"] = {
            "row": pd.DataFrame([values]),
            "actual": actual_value,
        }

    if "pred" in st.session_state:
        row = st.session_state["pred"]["row"]
        actual = st.session_state["pred"]["actual"]

        all_predictions = {
            model: float(R["models"][model].predict(row)[0])
            for model in MODEL_NAMES
        }

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            f"Prediksi ({main_name})",
            f"{all_predictions[main_name]:.2f}",
        )

        metric2.metric(
            f"Prediksi ({comp_name})"
            if comp_pipe is not None
            else "Model pembanding",
            f"{all_predictions[comp_name]:.2f}"
            if comp_pipe is not None
            else "-",
        )

        metric3.metric(
            "Skor Asli Sampel",
            f"{actual:.2f}" if actual is not None else "-",
        )

        left, right = st.columns([2, 1])

        output = pd.DataFrame(
            {
                "Model": MODEL_NAMES,
                "Prediksi": [
                    all_predictions[model]
                    for model in MODEL_NAMES
                ],
            }
        )

        if actual is not None:
            output["Selisih dari Aktual"] = (
                output["Prediksi"] - actual
            ).abs()

        left.bar_chart(
            output.set_index("Model")["Prediksi"]
        )

        right.dataframe(
            output.round(2),
            hide_index=True,
            width="stretch",
        )

        if actual is not None:
            st.caption(
                "Skor asli hanya ditampilkan jika nilai awal diambil "
                "dari dataset. Jika angka fitur diubah, perbandingan "
                "dengan skor asli tidak lagi setara."
            )

        st.caption(
            "Model dipilih berdasarkan RMSE pada data uji, "
            "bukan dari besar-kecilnya angka prediksi."
        )


# ---------------------------------------------------------------------------
# Tab 2: Evaluasi
# ---------------------------------------------------------------------------
with tab_eval:
    st.header("Performa Final pada Testing Set")

    main_metrics = metrics_of(main_name)
    metric_columns = st.columns(4)

    metric_columns[0].metric(
        f"{main_name} RMSE",
        f"{main_metrics['RMSE']:.4f}",
    )

    metric_columns[1].metric(
        f"{main_name} R²",
        f"{main_metrics['R2 Score']:.4f}",
    )

    if comp_pipe is not None:
        comparison_metrics = metrics_of(comp_name)

        metric_columns[2].metric(
            f"{comp_name} RMSE",
            f"{comparison_metrics['RMSE']:.4f}",
        )

        metric_columns[3].metric(
            f"{comp_name} R²",
            f"{comparison_metrics['R2 Score']:.4f}",
        )

    st.dataframe(
        perf.round(4),
        hide_index=True,
        width="stretch",
    )

    best = perf.sort_values("RMSE").iloc[0]

    st.markdown(
        f"Model terbaik berdasarkan RMSE: "
        f"**{best['Model']} ({best['Tahap']})** · "
        f"RMSE **{best['RMSE']:.4f}** · "
        f"R² **{best['R2 Score']:.4f}**"
    )

    if best["R2 Score"] >= 0.7:
        st.success(
            "Performa prediksi model terbaik tergolong baik (R² ≥ 0,7)."
        )
    elif best["R2 Score"] >= 0:
        st.warning(
            "Performa prediksi model terbaik masih terbatas "
            "(R² antara 0 dan 0,7)."
        )
    else:
        st.warning(
            "Model terbaik menurut RMSE pun belum optimal "
            "karena R² negatif."
        )

    st.subheader("Perbandingan Baseline vs Tuned")

    fig, ax = plt.subplots(figsize=(8, 4))

    (
        perf.pivot(
            index="Model",
            columns="Tahap",
            values="RMSE",
        )[["Baseline", "Tuned"]]
        .plot(
            kind="bar",
            ax=ax,
            color=[BLUE, ACCENT],
        )
    )

    ax.set_ylabel("RMSE")
    ax.set_title("Perbandingan RMSE: Baseline vs Tuned")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    show(fig)

    st.subheader(
        f"Ringkasan Hyperparameter Tuning ({CV_FOLDS}-Fold CV)"
    )

    st.dataframe(
        R["tuning"],
        hide_index=True,
        width="stretch",
    )

    st.subheader("Visualisasi Evaluasi")

    col_a, col_b = st.columns(2)

    with col_a:
        show(
            scatter_fig(
                R["y_test"],
                R["preds"][main_name],
                f"Aktual vs Prediksi - {main_name}",
            )
        )

    with col_b:
        if comp_pipe is not None:
            show(
                scatter_fig(
                    R["y_test"],
                    R["preds"][comp_name],
                    f"Aktual vs Prediksi - {comp_name}",
                )
            )
        else:
            st.info(
                "Pilih model pembanding di sidebar "
                "untuk melihat plot kedua."
            )

    col_a, col_b = st.columns(2)

    with col_a:
        if hasattr(
            main_pipe.named_steps["model"],
            "feature_importances_",
        ):
            show(
                importance_fig(
                    main_pipe,
                    f"Feature Importance - {main_name}",
                )
            )

    with col_b:
        if comp_pipe is not None:
            show(
                importance_fig(
                    comp_pipe,
                    f"Feature Importance - {comp_name}",
                )
            )

    st.subheader("Struktur Decision Tree (Tuned, kedalaman 3)")

    decision_tree = R["models"]["Decision Tree (Tuned)"]

    fig = plt.figure(figsize=(22, 9))

    plot_tree(
        decision_tree.named_steps["model"],
        feature_names=(
            decision_tree.named_steps[
                "preprocessor"
            ].get_feature_names_out()
        ),
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
    )

    show(fig)


# ---------------------------------------------------------------------------
# Tab 3: EDA
# ---------------------------------------------------------------------------
with tab_eda:
    st.header("Ringkasan & Catatan Eksperimen")

    st.markdown(
        f"- Observasi mentah: {len(raw):,}\n"
        f"- Observasi bersih (influence_score valid): {len(df):,}\n"
        f"- Fitur: {len(FEATURES)} "
        f"({len(NUMERIC_FEATURES)} numerik, "
        f"{len(CATEGORICAL_FEATURES)} kategorik)\n"
        f"- Training/testing: "
        f"{R['n_train']:,}/{R['n_test']:,} (80:20)\n"
        f"- Cross-validation: {CV_FOLDS}-fold KFold · "
        f"Metrik tuning: RMSE\n"
        f"- Kandidat Decision Tree: "
        f"{R['tuning'].iloc[0]['candidate_count']} · "
        f"Random Forest: "
        f"{R['tuning'].iloc[1]['candidate_count']}\n"
        f"- Model terbaik berdasarkan RMSE testing: "
        f"{best['Model']} ({best['Tahap']})"
    )

    with st.expander("Catatan Interpretasi & Batasan Metodologi"):
        st.markdown(
            "- Dataset relatif kecil, sehingga hasil sensitif "
            "terhadap pembagian data.\n"
            "- Nilai 3.3k / 475m / 1.2b diubah menjadi angka penuh "
            "sebelum pemodelan.\n"
            "- Nilai kosong diisi median (numerik) atau nilai "
            "'Tidak diketahui' (negara).\n"
            "- Hasil hanya berlaku pada dataset, split, random state, "
            "dan grid parameter yang diuji."
        )

    st.subheader("Pratinjau Dataset (50 Baris Pertama)")

    st.dataframe(
        df.head(50),
        width="stretch",
        height=380,
    )

    numeric_data = df[NUMERIC_FEATURES].copy()
    numeric_data[TARGET] = df[TARGET]

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.hist(
            df[TARGET],
            bins=25,
            color=BLUE,
            edgecolor="white",
        )

        ax.set_title("Distribusi Influence Score")
        ax.set_xlabel("Influence Score")
        ax.set_ylabel("Jumlah influencer")

        fig.tight_layout()
        show(fig)

    with col_b:
        correlation = numeric_data.corr()

        fig, ax = plt.subplots(figsize=(6, 5))

        image = ax.imshow(
            correlation,
            cmap="viridis",
            vmin=-1,
            vmax=1,
        )

        ax.set_xticks(range(len(correlation)))
        ax.set_yticks(range(len(correlation)))

        ax.set_xticklabels(
            correlation.columns,
            rotation=60,
            ha="right",
            fontsize=8,
        )

        ax.set_yticklabels(
            correlation.columns,
            fontsize=8,
        )

        for i in range(len(correlation)):
            for j in range(len(correlation)):
                value = correlation.iloc[i, j]

                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < 0.5 else "black",
                )

        fig.colorbar(
            image,
            ax=ax,
            label="Koefisien korelasi",
        )

        ax.set_title("Matriks Korelasi Pearson")

        fig.tight_layout()
        show(fig)

    st.subheader("Distribusi Fitur Numerik")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 7),
    )

    for ax, column in zip(
        axes.ravel(),
        NUMERIC_FEATURES,
    ):
        ax.hist(
            numeric_data[column].dropna(),
            bins=30,
            color=BLUE,
        )
        ax.set_title(column)

    fig.tight_layout()
    show(fig)

    st.subheader("10 Negara dengan Influencer Terbanyak")

    top_countries = (
        df["country"]
        .fillna("Tidak diketahui")
        .value_counts()
        .head(10)
    )

    col_a, col_b = st.columns([2, 1])

    col_a.bar_chart(top_countries)

    col_b.dataframe(
        top_countries.rename("jumlah").reset_index(),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Statistik Deskriptif Fitur")

    st.dataframe(
        numeric_data.describe().T.round(4),
        width="stretch",
    )
