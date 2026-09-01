import pandas as pd

# Exact 19 client/API fields required by the project specification.
API_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "Contract",
    "PaperlessBilling", "PaymentMethod", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
]


def compute_eco_preference(row: pd.Series) -> int:
    """Project-defined ESG/eco-preference proxy, scored from 0 to 2."""
    score = 0
    if row["PaperlessBilling"] == "Yes":
        score += 1
    if row["StreamingTV"] == "Yes" or row["StreamingMovies"] == "Yes":
        score += 1
    return score


def add_eco_preference(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["eco_preference"] = out.apply(compute_eco_preference, axis=1)
    return out


def customer_to_dataframe(customer: dict) -> pd.DataFrame:
    df = pd.DataFrame([customer], columns=API_FEATURES)
    return add_eco_preference(df)
