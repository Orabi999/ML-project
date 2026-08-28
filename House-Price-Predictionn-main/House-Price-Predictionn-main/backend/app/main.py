import json
import joblib
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.preprocessing import build_input_dataframe

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "models/house_price.pkl")
    locations_path = os.getenv("LOCATIONS_PATH", "locations.json")
    
    if os.path.exists(model_path):
        ml_models["pipeline"] = joblib.load(model_path)
    else:
        ml_models["pipeline"] = None
        
    if os.path.exists(locations_path):
        with open(locations_path, "r") as f:
            ml_models["locations"] = json.load(f)
    else:
        ml_models["locations"] = ["other"]
        
    yield
    ml_models.clear()

app = FastAPI(title="House Price Prediction API", lifespan=lifespan)

cors_origin = os.getenv("CORS_ORIGIN", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": ml_models.get("pipeline") is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not ml_models.get("pipeline"):
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    
    input_df = build_input_dataframe(request, ml_models["locations"])
    prediction = ml_models["pipeline"].predict(input_df)[0]
    return PredictionResponse(predicted_price=float(prediction))
