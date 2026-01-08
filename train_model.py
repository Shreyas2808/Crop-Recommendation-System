# train_model.py
"""
Train a classifier to predict crop Type from soil/fertilizer features.

Outputs:
- artifacts/pipeline.pkl
- artifacts/preprocessor.pkl
- artifacts/xgb_model.pkl
- artifacts/meta.json
- reports/accuracy_summary.csv
- reports/accuracy_recall_f1_graph.png
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# Try XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

SEED = 42
np.random.seed(SEED)

NUM_CANDIDATES = [
    "soil_ph", "organic_carbon_pct", "sand_pct", "silt_pct", "clay_pct",
    "N_req_kg_ha", "P2O5_req_kg_ha", "K2O_req_kg_ha",
    "NPK_total", "N_to_P", "N_to_K", "P_to_K"
]

CAT_CANDIDATES = [
    "soil_type", "season", "irrigation", "fertilizer_type", "city"
]

TARGET = "Type"


def detect_features(df):
    num_features = [c for c in NUM_CANDIDATES if c in df.columns]
    cat_features = [c for c in CAT_CANDIDATES if c in df.columns]
    return num_features, cat_features


def build_preprocessor(num_features, cat_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]) if num_features else "drop"

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]) if cat_features else "drop"

    transformers = []
    if num_features:
        transformers.append(("num", numeric_pipe, num_features))
    if cat_features:
        transformers.append(("cat", cat_pipe, cat_features))

    return ColumnTransformer(transformers)


def build_models():
    models = {}

    if XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.85,
            random_state=SEED,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1
        )

    models["hgb"] = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.08,
        random_state=SEED
    )

    models["rf"] = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1
    )

    return models


def plot_from_accuracy_csv(csv_path, save_path):
    df = pd.read_csv(csv_path)

    models = df["model"]
    accuracy = df["acc_mean"]
    recall = df["recall_mean"]
    f1 = df["f1_mean"]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(9, 5))

    bars1 = plt.bar(x - width, accuracy, width, label="Accuracy")
    bars2 = plt.bar(x, recall, width, label="Recall")
    bars3 = plt.bar(x + width, f1, width, label="F1-Score")

    plt.xticks(x, models)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Accuracy, Recall and F1-Score Comparison")
    plt.legend()
    plt.ylim(0, 1.05)

    # show values on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args):
    df = pd.read_csv(args.data)
    assert TARGET in df.columns, "Target column not found"

    for col in NUM_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df[TARGET].notna()].copy()

    num_features, cat_features = detect_features(df)
    X = df[num_features + cat_features]

    le = LabelEncoder()
    y = le.fit_transform(df[TARGET].astype(str))

    pre = build_preprocessor(num_features, cat_features)
    models = build_models()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []

    for name, model in models.items():
        pipe = make_pipeline(pre, model)

        acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy").mean()
        rec = cross_val_score(pipe, X, y, cv=skf, scoring="recall_macro").mean()
        f1m = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro").mean()

        cv_rows.append({
            "model": name,
            "acc_mean": acc,
            "recall_mean": rec,
            "f1_mean": f1m
        })

    cv_df = pd.DataFrame(cv_rows).sort_values("acc_mean", ascending=False)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    csv_path = reports_dir / "accuracy_summary.csv"
    cv_df.to_csv(csv_path, index=False)

    plot_from_accuracy_csv(
        csv_path,
        reports_dir / "accuracy_recall_f1_graph.png"
    )

    best_model_name = cv_df.iloc[0]["model"]
    chosen_model = models[best_model_name]

    full_pipe = make_pipeline(pre, chosen_model)
    full_pipe.fit(X, y)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    joblib.dump(full_pipe, artifacts_dir / "pipeline.pkl")
    joblib.dump(pre, artifacts_dir / "preprocessor.pkl")
    joblib.dump(chosen_model, artifacts_dir / "xgb_model.pkl")

    meta = {
        "target": TARGET,
        "classes": le.classes_.tolist(),
        "num_features": num_features,
        "cat_features": cat_features,
        "chosen_model": best_model_name
    }

    (artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    args = parser.parse_args()
    main(args)