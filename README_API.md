# Churn + ESG FastAPI Backend

This backend implements the API requirements in the project specification:
- `GET /` health check
- `POST /predict` prediction endpoint
- `GET /docs` Swagger UI
- 19 customer input fields
- server-side `eco_preference` calculation
- saved model + K-Means artifacts loaded once at startup
- Pydantic categorical validation
- CORS
- JSON error responses

## 1. Dataset

Download the blastchar Telco Customer Churn dataset from Kaggle and place:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

inside this `backend/` folder.

## 2. Install

```bash
python -m venv venv
source venv/bin/activate
# Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

## 3. Train artifacts

```bash
python train_model.py
```

This creates:
- `artifacts/model.joblib`
- `artifacts/kmeans.joblib`
- `artifacts/metadata.joblib`

## 4. Start API

```bash
uvicorn main:app --reload --port 8000
```

Open Swagger:

`http://localhost:8000/docs`

## 5. Endpoints

### GET /

Returns API health and loaded model type.

### POST /predict

Input: exactly the 19 customer fields defined in `schema.py`.

The client must NOT send `eco_preference`; the API calculates it from:
- `PaperlessBilling == "Yes"` → +1
- `StreamingTV == "Yes"` OR `StreamingMovies == "Yes"` → +1

Response:

```json
{
  "churn_probability": 0.73,
  "churn_label": "Yes",
  "segment": "at-risk_segment_2",
  "eco_preference": 1
}
```

## Important project limitation

The public Telco dataset has no actual ESG variable. `eco_preference` is therefore a project-defined behavioural proxy, not a measured ESG score. Results involving this feature must be interpreted as associations rather than causal ESG effects.
