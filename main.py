from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

app = FastAPI(
    title="ML Inference API",
    description="A demo REST API that serves an Iris flower classifier.",
    version="1.0.0",
)

class PredictRequest(BaseModel):
    # Iris dataset uses four numeric measurements, in centimeters.
    features: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        examples=[[5.1, 3.5, 1.4, 0.2]],
        description="Four values: sepal length, sepal width, petal length, petal width.",
    )

def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "model.joblib is missing. Run `python train_model.py` before starting the API."
        )
    return joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {
        "message": "ML Inference API is running",
        "docs": "/docs",
        "predict_endpoint": "POST /predict",
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_exists": MODEL_PATH.exists()}

@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        model = load_model()
        values = np.array(payload.features, dtype=float).reshape(1, -1)
        prediction = int(model.predict(values)[0])
        probabilities = model.predict_proba(values)[0].tolist()
        class_names = ["setosa", "versicolor", "virginica"]
        return {
            "prediction": prediction,
            "class_name": class_names[prediction],
            "probabilities": {
                name: round(float(prob), 6)
                for name, prob in zip(class_names, probabilities)
            },
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="Invalid feature values.") from exc
