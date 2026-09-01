from pathlib import Path
import logging

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schema import CustomerProfile, PredictionResponse
from preprocessing import customer_to_dataframe


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-esg-api")


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Churn Prediction with ESG-Conscious Customers API",
    description=(
        "API for telecom customer churn prediction and "
        "ESG/eco-preference customer segmentation."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Model variables
# ---------------------------------------------------------

model = None
kmeans = None
model_metadata = None


# ---------------------------------------------------------
# Load trained artifacts
# ---------------------------------------------------------

def load_artifacts():
    global model
    global kmeans
    global model_metadata

    model_path = ARTIFACT_DIR / "model.joblib"
    kmeans_path = ARTIFACT_DIR / "kmeans.joblib"
    metadata_path = ARTIFACT_DIR / "metadata.joblib"

    missing = []

    if not model_path.exists():
        missing.append("model.joblib")

    if not kmeans_path.exists():
        missing.append("kmeans.joblib")

    if not metadata_path.exists():
        missing.append("metadata.joblib")

    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: "
            + ", ".join(missing)
            + ". Run train_model.py first."
        )

    model = joblib.load(model_path)
    kmeans = joblib.load(kmeans_path)
    model_metadata = joblib.load(metadata_path)

    logger.info(
        "Model loaded successfully: %s",
        model_metadata.get("model_type", "unknown"),
    )


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

@app.on_event("startup")
def startup_event():
    try:
        load_artifacts()
        logger.info("All model artifacts loaded successfully.")

    except Exception as exc:
        logger.exception(
            "Artifact loading failed: %s",
            exc,
        )


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/")
def health_check():

    loaded = (
        model is not None
        and kmeans is not None
        and model_metadata is not None
    )

    return {
        "status": "ok" if loaded else "degraded",
        "service": "Churn ESG Prediction API",
        "model_type": (
            model_metadata.get("model_type", "unknown")
            if model_metadata
            else None
        ),
        "artifacts_loaded": loaded,
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(customer: CustomerProfile):

    if model is None or kmeans is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifacts are not loaded. "
                "Run train_model.py and restart the API."
            ),
        )

    try:

        # Convert incoming customer data into DataFrame
        customer_dict = customer.model_dump()

        df = customer_to_dataframe(customer_dict)

        # -------------------------------------------------
        # Churn prediction
        # -------------------------------------------------

        probability = float(
            model.predict_proba(df)[0, 1]
        )

        churn_label = (
            "Yes"
            if probability >= 0.5
            else "No"
        )

        # -------------------------------------------------
        # ESG / Eco preference
        # -------------------------------------------------

        eco_preference = int(
            df.loc[0, "eco_preference"]
        )

        # -------------------------------------------------
        # Customer segmentation
        # -------------------------------------------------

        segment_input = df[
            [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "eco_preference",
            ]
        ].astype(float)

        cluster_id = int(
            kmeans.predict(segment_input)[0]
        )

        # -------------------------------------------------
        # Segment name
        # -------------------------------------------------

        segment_names = (
            model_metadata.get("segment_names", {})
            if model_metadata
            else {}
        )

        segment = segment_names.get(
            str(cluster_id),
            f"segment_{cluster_id}",
        )

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return PredictionResponse(
            churn_probability=round(
                probability,
                6,
            ),
            churn_label=churn_label,
            segment=segment,
            eco_preference=eco_preference,
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Prediction failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )
