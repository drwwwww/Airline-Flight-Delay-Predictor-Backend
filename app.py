import json, joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

HERE = Path(__file__).parent  
MODEL_PATH = (HERE / "model.pkl").resolve() 
FEATURES_PATH = (HERE / "features.json").resolve()

class FlightFeatures(BaseModel):
    depHour: int = Field(..., ge=0, le=23, description="Local hour 0–23 at origin")
    origin: str = Field(..., min_length=3, max_length=3, description="IATA code, e.g., ATL")
    dest: str = Field(..., min_length=3, max_length=3, description="IATA code, e.g., JFK")

class FlightOut(BaseModel):
    delayProb: float = Field(..., ge=0.0, le=1.0, description="Probability of a delay")
    delayed: bool = Field(..., description="Will the flight be delayed")

app = FastAPI(title="Flight Delay Predictor", version="0.1")

model = None
featureMeta = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prod: your real domain(s)
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["Content-Type","Authorization"],
)

def ensure_file(path, label):
    if not Path(path).exists() or not Path(path).is_file():
        raise RuntimeError(f"{label} not found at {path}")

@app.on_event("startup")
def loadArtifacts():
    global model, featureMeta
    ensure_file(MODEL_PATH, "model.pkl")
    ensure_file(FEATURES_PATH, "features.json")
    feats = FEATURES_PATH.read_text(encoding="utf-8")
    fj = json.loads(feats)
    if "numeric" not in fj or "categorical" not in fj:
        raise RuntimeError("features.json needs 'numeric' and 'categorical'")


    featureMeta = fj

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model lacks predict_proba")
    
@app.get("/health")
def health():
    return {"ok": True, "model_loaded": model is not None}

def normalizeIata(s):
    return s.strip().upper()

def getColumns():
    return featureMeta["numeric"] + featureMeta["categorical"]

@app.post("/predict")
def Predictionout():
    delayProb = 0.0
    delayed = False
