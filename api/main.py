from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features.feature_engineering import create_features


MODEL_PATH = Path("models/versions/v1/model.joblib")
MODEL_VERSION = "v1"


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production API for customer churn prediction",
    version=MODEL_VERSION,
)


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION
    }


@app.get("/model-info")
def model_info():
    classifier = model.named_steps["classifier"]

    return {
        "model_version": MODEL_VERSION,
        "model_type": type(classifier).__name__,
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        input_data = customer.model_dump()

        df = create_features(
            __import__("pandas").DataFrame([input_data])
        )

        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4),
            "model_version": MODEL_VERSION
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerData]):
    try:
        import pandas as pd

        data = [customer.model_dump() for customer in customers]
        df = pd.DataFrame(data)

        df = create_features(df)

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        results = []

        for prediction, probability in zip(predictions, probabilities):
            results.append({
                "prediction": "Yes" if int(prediction) == 1 else "No",
                "churn_probability": round(float(probability), 4),
                "model_version": MODEL_VERSION
            })

        return {
            "count": len(results),
            "predictions": results
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )