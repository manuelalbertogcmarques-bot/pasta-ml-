"""
FastAPI app that serves installation duration predictions from a trained XGBoost model.

Run:  uvicorn app:app --reload --port 8000
"""

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from supabase import create_client

load_dotenv()

app = FastAPI(title="Previsão de Duração de Instalações", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")
META_PATH = os.path.join(os.path.dirname(__file__), "model_meta.json")

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")

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
    try:
        load_model()
    except FileNotFoundError:
        tecnico_map.update({"tecnico1": 0, "tecnico2": 1})
        feature_names.extend(["m_cabo", "tecnico_enc"])


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


def fetch_training_data() -> pd.DataFrame:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL e SUPABASE_ANON_KEY não configurados.")
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = sb.table("instalacoes").select("m_cabo, tecnico, tempo_instalacao_real").execute()
    rows = resp.data
    if not rows:
        raise RuntimeError("Sem dados na tabela instalacoes.")
    return pd.DataFrame(rows)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["tecnico_enc"] = df["tecnico"].map(tecnico_map).fillna(0).astype(int)
    X = df[["m_cabo", "tecnico_enc"]]
    y = df["tempo_instalacao_real"]
    return X, y


def retrain_model() -> dict:
    global model, tecnico_map, feature_names

    if not tecnico_map:
        tecnico_map = {"tecnico1": 0, "tecnico2": 1}
    if not feature_names:
        feature_names = ["m_cabo", "tecnico_enc"]

    df = fetch_training_data()
    n_registos = len(df)

    X, y = prepare_features(df)

    if n_registos < 4:
        model = xgb.XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        model.fit(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    model.save_model(MODEL_PATH)

    meta = {
        "tecnico_map": tecnico_map,
        "feature_names": feature_names,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    result = {"sucesso": True, "registos": n_registos}
    if n_registos >= 4:
        result["mae"] = round(mae, 2)
        result["rmse"] = round(rmse, 2)
    return result


@app.post("/retreinar")
def retreinar():
    try:
        result = retrain_model()
        return result
    except Exception as e:
        return {"sucesso": False, "erro": str(e)}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
