"""
Offline training script.

This script:
1. Loads the Kaggle Telco Customer Churn CSV.
2. Cleans TotalCharges.
3. Creates the project-defined eco_preference proxy.
4. Builds RF, SVM and ANN classifiers with preprocessing.
5. Evaluates each model with and without eco_preference.
6. Selects the best model by F1-score, then ROC-AUC.
7. Trains K-Means segmentation on tenure, MonthlyCharges,
   TotalCharges and eco_preference.
8. Saves model artifacts for main.py.

Expected CSV:
backend/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler as ClusterScaler
from kmeans_predictor import ScaledKMeansPredictor

from preprocessing import add_eco_preference

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

TARGET = "Churn"

BASE_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
BASE_CATEGORICAL = [
    "gender", "Partner", "Dependents", "Contract",
    "PaperlessBilling", "PaymentMethod", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
]

MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        probability=True,
        class_weight="balanced",
        random_state=42,
    ),
    "ANN": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        max_iter=500,
        early_stopping=True,
        random_state=42,
    ),
}


def make_pipeline(use_esg: bool, estimator):
    numeric = BASE_NUMERIC + (["eco_preference"] if use_esg else [])
    categorical = BASE_CATEGORICAL

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def evaluate(pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    return {
        "model": pipe.named_steps["model"].__class__.__name__,
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "pipeline": pipe,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download the blastchar Telco Customer Churn CSV and place it there."
        )

    df = pd.read_csv(DATA_PATH)

    # TotalCharges has blanks for zero-tenure customers in the source dataset.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df = add_eco_preference(df)
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})

    # Preserve churn proportion in the 80:20 split.
    X = df.drop(columns=[TARGET, "customerID"], errors="ignore")
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    results = []

    for model_name, estimator in MODELS.items():
        for use_esg in [False, True]:
            pipe = make_pipeline(use_esg, estimator)
            result = evaluate(
                pipe, X_train, X_test, y_train, y_test
            )
            result["model_name"] = model_name
            result["esg_feature"] = use_esg
            results.append(result)

            print(
                f"{model_name:15s} | ESG={use_esg!s:5s} | "
                f"Precision={result['precision']:.4f} | "
                f"Recall={result['recall']:.4f} | "
                f"F1={result['f1']:.4f} | "
                f"ROC-AUC={result['roc_auc']:.4f}"
            )

    # Primary selection criterion: F1; tie-breaker: ROC-AUC.
    best = max(results, key=lambda r: (r["f1"], r["roc_auc"]))

    print("\nSelected model:")
    print(best["model_name"], "with ESG =", best["esg_feature"])
    print("F1 =", best["f1"], "ROC-AUC =", best["roc_auc"])

    # Save the complete fitted pipeline. This includes the encoder/scaler,
    # so prediction-time transformations are identical to training.
    joblib.dump(best["pipeline"], ARTIFACT_DIR / "model.joblib")

    # K-Means segmentation.
    segment_features = df[
        ["tenure", "MonthlyCharges", "TotalCharges", "eco_preference"]
    ].astype(float)

    cluster_scaler = ClusterScaler()
    segment_scaled = cluster_scaler.fit_transform(segment_features)

    candidates = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(segment_scaled)
        candidates[k] = silhouette_score(segment_scaled, labels)

    best_k = max(candidates, key=candidates.get)
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(segment_scaled)

    # IMPORTANT: API receives raw four variables, so save a wrapper-like
    # dictionary containing both scaler and KMeans.
    joblib.dump(
        {
            "scaler": cluster_scaler,
            "kmeans": kmeans,
        },
        ARTIFACT_DIR / "kmeans.joblib",
    )

    # Update API-compatible KMeans artifact to accept raw values by wrapping
    # prediction in a small sklearn-compatible object.
    joblib.dump(
        ScaledKMeansPredictor(cluster_scaler, kmeans),
        ARTIFACT_DIR / "kmeans.joblib",
    )

    # Basic automatic segment labels from cluster means.
    profile = df.assign(cluster=labels).groupby("cluster").agg(
        tenure=("tenure", "mean"),
        MonthlyCharges=("MonthlyCharges", "mean"),
        TotalCharges=("TotalCharges", "mean"),
        eco_preference=("eco_preference", "mean"),
        churn_rate=(TARGET, "mean"),
    )

    segment_names = {}
    for cluster_id, row in profile.iterrows():
        if row["churn_rate"] >= profile["churn_rate"].quantile(0.75):
            name = "at-risk"
        elif row["eco_preference"] >= profile["eco_preference"].quantile(0.75):
            name = "eco-conscious"
        elif row["MonthlyCharges"] >= profile["MonthlyCharges"].quantile(0.75):
            name = "high-value"
        else:
            name = "loyal-low-engagement"
        segment_names[str(int(cluster_id))] = f"{name}_segment_{int(cluster_id)}"

    metadata = {
        "model_type": best["model_name"],
        "with_esg_feature": bool(best["esg_feature"]),
        "selection_metric": "F1-score, ROC-AUC tie-breaker",
        "metrics": {
            f"{r['model_name']}_ESG_{r['esg_feature']}": {
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "roc_auc": r["roc_auc"],
                "confusion_matrix": r["confusion_matrix"],
            }
            for r in results
        },
        "k": int(best_k),
        "silhouette_scores": {str(k): v for k, v in candidates.items()},
        "segment_names": segment_names,
    }

    joblib.dump(metadata, ARTIFACT_DIR / "metadata.joblib")

    print("\nSaved:")
    print(ARTIFACT_DIR / "model.joblib")
    print(ARTIFACT_DIR / "kmeans.joblib")
    print(ARTIFACT_DIR / "metadata.joblib")


if __name__ == "__main__":
    main()
