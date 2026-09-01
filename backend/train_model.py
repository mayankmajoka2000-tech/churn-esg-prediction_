"""
Offline training script for Customer Churn + ESG/Eco Preference prediction.

This script:
1. Loads the Customer Churn CSV.
2. Cleans TotalCharges.
3. Creates the project-defined eco_preference feature.
4. Builds Random Forest, SVM and ANN classifiers.
5. Evaluates each model with and without eco_preference.
6. Selects the best model using F1-score, then ROC-AUC.
7. Trains K-Means customer segmentation.
8. Automatically selects the best K using silhouette score.
9. Saves trained model artifacts for the API.

Expected dataset:
backend/Customer_Churning.csv
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from kmeans_predictor import ScaledKMeansPredictor
from preprocessing import add_eco_preference


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Your actual uploaded dataset
DATA_PATH = BASE_DIR / "Customer_Churning.csv"

# Folder where trained models will be saved
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


# ============================================================
# TARGET VARIABLE
# ============================================================

TARGET = "Churn"


# ============================================================
# FEATURES
# ============================================================

BASE_NUMERIC = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

BASE_CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


# ============================================================
# MACHINE LEARNING MODELS
# ============================================================

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


# ============================================================
# CREATE MODEL PIPELINE
# ============================================================

def make_pipeline(use_esg, estimator):

    numeric_features = BASE_NUMERIC.copy()

    if use_esg:
        numeric_features.append("eco_preference")

    categorical_features = BASE_CATEGORICAL.copy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    return pipeline


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate(
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
):

    # Train
    pipeline.fit(X_train, y_train)

    # Predictions
    predictions = pipeline.predict(X_test)

    # Probability of churn
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    ).tolist()

    return {
        "model": pipeline.named_steps["model"].__class__.__name__,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "pipeline": pipeline,
    }


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("CUSTOMER CHURN + ESG MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"\nDataset not found at:\n{DATA_PATH}\n\n"
            "Make sure Customer_Churning.csv is inside the backend folder."
        )

    print("\nDataset found:")
    print(DATA_PATH)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print("\nDataset shape:")
    print(df.shape)

    print("\nDataset columns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # Clean TotalCharges
    # --------------------------------------------------------

    if "TotalCharges" not in df.columns:

        raise ValueError(
            "The dataset does not contain the required "
            "'TotalCharges' column."
        )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # --------------------------------------------------------
    # Create eco preference feature
    # --------------------------------------------------------

    print("\nCreating eco_preference feature...")

    df = add_eco_preference(df)

    if "eco_preference" not in df.columns:

        raise ValueError(
            "eco_preference was not created by add_eco_preference()."
        )

    # --------------------------------------------------------
    # Convert target
    # --------------------------------------------------------

    if TARGET not in df.columns:

        raise ValueError(
            f"The dataset does not contain the target column '{TARGET}'."
        )

    df[TARGET] = df[TARGET].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    # Remove rows where target could not be converted
    df = df.dropna(subset=[TARGET])

    df[TARGET] = df[TARGET].astype(int)

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X = df.drop(
        columns=[
            TARGET,
            "customerID",
        ],
        errors="ignore",
    )

    y = df[TARGET]

    print("\nTarget distribution:")
    print(y.value_counts())

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results = []

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    for model_name, estimator in MODELS.items():

        for use_esg in [False, True]:

            print(
                f"\nTraining {model_name} | "
                f"ESG feature = {use_esg}"
            )

            pipeline = make_pipeline(
                use_esg,
                estimator,
            )

            result = evaluate(
                pipeline,
                X_train,
                X_test,
                y_train,
                y_test,
            )

            result["model_name"] = model_name
            result["esg_feature"] = use_esg

            results.append(result)

            print(
                f"Precision : {result['precision']:.4f}"
            )

            print(
                f"Recall    : {result['recall']:.4f}"
            )

            print(
                f"F1-score  : {result['f1']:.4f}"
            )

            print(
                f"ROC-AUC   : {result['roc_auc']:.4f}"
            )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best = max(
        results,
        key=lambda result: (
            result["f1"],
            result["roc_auc"],
        ),
    )

    print("\n")
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        "Model:",
        best["model_name"],
    )

    print(
        "ESG feature:",
        best["esg_feature"],
    )

    print(
        "F1-score:",
        round(best["f1"], 4),
    )

    print(
        "ROC-AUC:",
        round(best["roc_auc"], 4),
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    model_path = ARTIFACT_DIR / "model.joblib"

    joblib.dump(
        best["pipeline"],
        model_path,
    )

    print("\nSaved model:")
    print(model_path)

    # ========================================================
    # K-MEANS SEGMENTATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("K-MEANS CUSTOMER SEGMENTATION")
    print("=" * 70)

    segment_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "eco_preference",
    ]

    segment_features = df[
        segment_columns
    ].astype(float)

    # --------------------------------------------------------
    # Scale clustering features
    # --------------------------------------------------------

    cluster_scaler = StandardScaler()

    segment_scaled = cluster_scaler.fit_transform(
        segment_features
    )

    # --------------------------------------------------------
    # Find best K using silhouette score
    # --------------------------------------------------------

    candidates = {}

    for k in range(2, 9):

        kmeans_test = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )

        labels_test = kmeans_test.fit_predict(
            segment_scaled
        )

        score = silhouette_score(
            segment_scaled,
            labels_test,
        )

        candidates[k] = score

        print(
            f"K={k} | "
            f"Silhouette Score={score:.4f}"
        )

    best_k = max(
        candidates,
        key=candidates.get,
    )

    print(
        "\nSelected K:",
        best_k,
    )

    # --------------------------------------------------------
    # Train final K-Means
    # --------------------------------------------------------

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20,
    )

    labels = kmeans.fit_predict(
        segment_scaled
    )

    # --------------------------------------------------------
    # Save K-Means predictor
    # --------------------------------------------------------

    kmeans_predictor = ScaledKMeansPredictor(
        cluster_scaler,
        kmeans,
    )

    kmeans_path = ARTIFACT_DIR / "kmeans.joblib"

    joblib.dump(
        kmeans_predictor,
        kmeans_path,
    )

    print("\nSaved K-Means model:")
    print(kmeans_path)

    # ========================================================
    # CUSTOMER SEGMENT PROFILES
    # ========================================================

    profile = (
        df.assign(cluster=labels)
        .groupby("cluster")
        .agg(
            tenure=("tenure", "mean"),
            MonthlyCharges=("MonthlyCharges", "mean"),
            TotalCharges=("TotalCharges", "mean"),
            eco_preference=("eco_preference", "mean"),
            churn_rate=(TARGET, "mean"),
        )
    )

    print("\n")
    print("=" * 70)
    print("CUSTOMER SEGMENT PROFILES")
    print("=" * 70)

    print(profile)

    # ========================================================
    # AUTOMATIC SEGMENT NAMES
    # ========================================================

    segment_names = {}

    churn_threshold = profile[
        "churn_rate"
    ].quantile(0.75)

    eco_threshold = profile[
        "eco_preference"
    ].quantile(0.75)

    charges_threshold = profile[
        "MonthlyCharges"
    ].quantile(0.75)

    for cluster_id, row in profile.iterrows():

        if row["churn_rate"] >= churn_threshold:

            name = "at-risk"

        elif row["eco_preference"] >= eco_threshold:

            name = "eco-conscious"

        elif row["MonthlyCharges"] >= charges_threshold:

            name = "high-value"

        else:

            name = "loyal-low-engagement"

        segment_names[
            str(int(cluster_id))
        ] = (
            f"{name}_segment_{int(cluster_id)}"
        )

    print("\nSegment names:")

    for cluster, name in segment_names.items():

        print(
            f"Cluster {cluster}: {name}"
        )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "model_type": best["model_name"],

        "with_esg_feature": bool(
            best["esg_feature"]
        ),

        "selection_metric": (
            "F1-score, ROC-AUC tie-breaker"
        ),

        "dataset": "Customer_Churning.csv",

        "metrics": {
            f"{result['model_name']}_ESG_{result['esg_feature']}": {
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "roc_auc": result["roc_auc"],
                "confusion_matrix": result[
                    "confusion_matrix"
                ],
            }

            for result in results
        },

        "k": int(best_k),

        "silhouette_scores": {
            str(k): float(score)

            for k, score in candidates.items()
        },

        "segment_names": segment_names,
    }

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata_path = ARTIFACT_DIR / "metadata.joblib"

    joblib.dump(
        metadata,
        metadata_path,
    )

    print("\nSaved metadata:")
    print(metadata_path)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        "1.",
        model_path,
    )

    print(
        "2.",
        kmeans_path,
    )

    print(
        "3.",
        metadata_path,
    )

    print("\nBest model:")
    print(best["model_name"])

    print(
        "ESG feature used:",
        best["esg_feature"],
    )

    print(
        "F1-score:",
        round(best["f1"], 4),
    )

    print(
        "ROC-AUC:",
        round(best["roc_auc"], 4),
    )

    print(
        "Number of customer segments:",
        best_k,
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
