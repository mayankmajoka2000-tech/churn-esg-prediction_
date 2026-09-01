from pathlib import Path
import logging

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schema import CustomerProfile, PredictionResponse
from preprocessing import customer_to_dataframe
from kmeans_predictor import ScaledKMeansPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-esg-api")

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

app = FastAPI(
    title="Churn Prediction with ESG-Conscious Customers API",
    description=(
        "Decision-support API for telecom customer churn prediction and "
        "ESG/eco-preference segmentation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to the frontend origin in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
kmeans = None
model_metadata = None


def load_artifacts() -> None:
    global model, kmeans, model_metadata

    required = {
        "model": ARTIFACT_DIR / "model.joblib",
        "kmeans": ARTIFACT_DIR / "kmeans.joblib",
        "metadata": ARTIFACT_DIR / "metadata.joblib",
    }

    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: "
            + ", ".join(missing)
            + ". Run train_model.py before starting the API."
        )

    model = joblib.load(required["model"])
    kmeans = joblib.load(required["kmeans"])
    model_metadata = joblib.load(required["metadata"])

    logger.info("Loaded model=%s", model_metadata.get("model_type", "unknown"))


@app.on_event("startup")
def startup_event():
    try:
        load_artifacts()
    except Exception as exc:
        logger.exception("Artifact loading failed: %s", exc)
        # Keep the application available so GET / can report the problem.
        # /predict will return a clear 503 until artifacts are available.


@app.get("/")
def health_check():
    loaded = model is not None and kmeans is not None
    return {
        "status": "ok" if loaded else "degraded",
        "model_type": (
            model_metadata.get("model_type", "unknown")
            if model_metadata else None
        ),
        "artifacts_loaded": loaded,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerProfile):
    if model is None or kmeans is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not loaded. Run train_model.py and restart the API.",
        )

    try:
        customer_dict = customer.model_dump()
        df = customer_to_dataframe(customer_dict)

        # The saved classifier is expected to be a complete sklearn Pipeline
        # containing the same preprocessing used during training.
        probability = float(model.predict_proba(df)[0, 1])
        label = "Yes" if probability >= 0.5 else "No"

        eco_preference = int(df.loc[0, "eco_preference"])

        # K-Means is trained on the four segmentation variables.
        segment_input = df[[
            "tenure", "MonthlyCharges", "TotalCharges", "eco_preference"
        ]].astype(float)

        cluster_id = int(kmeans.predict(segment_input)[0])

        segment_names = (
            model_metadata.get("segment_names", {})
            if model_metadata else {}
        )
        segment = segment_names.get(str(cluster_id), f"segment_{cluster_id}")

        return PredictionResponse(
            churn_probability=round(probability, 6),
            churn_label=label,
            segment=segment,
            eco_preference=eco_preference,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )
