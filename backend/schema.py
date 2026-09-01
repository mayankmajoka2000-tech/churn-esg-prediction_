from typing import Literal
from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]
Gender = Literal["Male", "Female"]
Contract = Literal["Month-to-month", "One year", "Two year"]
PaymentMethod = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
MultipleLines = Literal["Yes", "No", "No phone service"]
InternetService = Literal["DSL", "Fiber optic", "No"]
InternetDependentService = Literal["Yes", "No", "No internet service"]


class CustomerProfile(BaseModel):
    gender: Gender
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo

    tenure: int = Field(ge=0, le=100)
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    PhoneService: YesNo
    MultipleLines: MultipleLines
    InternetService: InternetService
    OnlineSecurity: InternetDependentService
    OnlineBackup: InternetDependentService
    DeviceProtection: InternetDependentService
    TechSupport: InternetDependentService
    StreamingTV: InternetDependentService
    StreamingMovies: InternetDependentService


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn_label: Literal["Yes", "No"]
    segment: str
    eco_preference: int = Field(ge=0, le=2)
