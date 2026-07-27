"""
FastAPI app that serves installation duration predictions from a trained XGBoost model.

Run:  uvicorn app:app --reload --port 8000
"""

import json
import os

import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Previsão de Duração de Instalações", version="1.0.0")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")
META_PATH = os.path.join(os.path.dirname(__file__), "model_meta.json")

model = None
tecnico_map = {}
feature_names = []


def load_model():
    global model, tecnico_map, feature_names
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "model.json not found. Run `python train_model.py` first."
        )
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    with open(META_PATH) as f:
        meta = json.load(f)
    tecnico_map = meta["tecnico_map"]
    feature_names = meta["feature_names"]


@app.on_event("startup")
def startup():
    load_model()


class PrevisaoRequest(BaseModel):
    m_cabo: int
    tecnico: str


class PrevisaoResponse(BaseModel):
    tempo_previsto: float
    m_cabo: int
    tecnico: str


@app.post("/prever", response_model=PrevisaoResponse)
def prever(req: PrevisaoRequest):
    tecnico_enc = tecnico_map.get(req.tecnico, 0)
    X = [[req.m_cabo, tecnico_enc]]
    pred = float(model.predict(X)[0])
    return PrevisaoResponse(
        tempo_previsto=round(pred, 1),
        m_cabo=req.m_cabo,
        tecnico=req.tecnico,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
