
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.database import get_connection
from src.features.feature_engineering import create_features
from fastapi.middleware.cors import CORSMiddleware




MODEL_PATH = Path("models/versions/v1/model.joblib")
MODEL_VERSION = "v1"

app = FastAPI(
    title="Customer Churn Prediction API",
    version=MODEL_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


def save_prediction(prediction: str, probability: float):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO predictions
                    (prediction, churn_probability, model_version)
                VALUES (%s, %s, %s)
                """,
                (prediction, probability, MODEL_VERSION),
            )

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
    }


@app.get("/model-info")
def model_info():
    classifier = model.named_steps["classifier"]

    return {
        "model_version": MODEL_VERSION,
        "model_type": type(classifier).__name__,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        df = pd.DataFrame([customer.model_dump()])
        df = create_features(df)

        prediction_value = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])
        prediction = "Yes" if prediction_value == 1 else "No"

        save_prediction(prediction, probability)

        return {
            "prediction": prediction,
            "churn_probability": round(probability, 4),
            "model_version": MODEL_VERSION,
            "logged_to_database": True,
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerData]):
    if not customers:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one customer.",
        )

    try:
        df = pd.DataFrame(
            [customer.model_dump() for customer in customers]
        )
        df = create_features(df)

        prediction_values = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        results = []

        # Save all batch predictions in one database transaction.
        connection = get_connection()

        try:
            with connection.cursor() as cursor:
                for value, probability in zip(
                    prediction_values, probabilities
                ):
                    prediction = "Yes" if int(value) == 1 else "No"
                    probability = float(probability)

                    cursor.execute(
                        """
                        INSERT INTO predictions
                            (prediction, churn_probability, model_version)
                        VALUES (%s, %s, %s)
                        """,
                        (prediction, probability, MODEL_VERSION),
                    )

                    results.append({
                        "prediction": prediction,
                        "churn_probability": round(probability, 4),
                        "model_version": MODEL_VERSION,
                    })

            connection.commit()

        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        return {
            "count": len(results),
            "predictions": results,
            "logged_to_database": True,
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
