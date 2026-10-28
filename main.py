
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Load the trained model
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Dry Bean Random Forest API")


# Input: 16 features from the user
class BeanInput(BaseModel):
    Area: float = Field(gt=0)
    Perimeter: float = Field(gt=0)
    MajorAxisLength: float = Field(gt=0)
    MinorAxisLength: float = Field(gt=0)
    AspectRation: float = Field(gt=0)
    Eccentricity: float = Field(gt=0, le=1)
    ConvexArea: float = Field(gt=0)
    EquivDiameter: float = Field(gt=0)
    Extent: float = Field(gt=0, le=1)
    Solidity: float = Field(gt=0, le=1)
    roundness: float = Field(gt=0, le=1)
    Compactness: float = Field(gt=0, le=1)
    ShapeFactor1: float = Field(gt=0)
    ShapeFactor2: float = Field(gt=0)
    ShapeFactor3: float = Field(gt=0, le=1)
    ShapeFactor4: float = Field(gt=0, le=1)


@app.get("/")
def ready_to_serve():
    return {"status": "Ready to Serve"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Endpoint for making predictions
@app.post("/predict")
def predict(data: BeanInput):
    features = pd.DataFrame([data.model_dump()])

    prediction = model.predict(features)[0]

    return {
        "prediction": str(prediction)
    }
