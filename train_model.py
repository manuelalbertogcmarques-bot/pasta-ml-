"""
Train an XGBoost regression model to predict installation duration.

Reads data from the Supabase `instalacoes` table, trains the model,
evaluates it, and exports the trained model to JSON for use by the FastAPI app.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
    sys.exit(1)


def fetch_data() -> pd.DataFrame:
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = sb.table("instalacoes").select("m_cabo, tecnico, tempo_instalacao_real").execute()
    rows = resp.data
    if not rows:
        print("ERROR: No data found in instalacoes table.")
        sys.exit(1)
    df = pd.DataFrame(rows)
    print(f"Fetched {len(df)} records from Supabase.")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    tecnico_map = {"tecnico1": 0, "tecnico2": 1}
    df["tecnico_enc"] = df["tecnico"].map(tecnico_map).fillna(0).astype(int)
    X = df[["m_cabo", "tecnico_enc"]]
    y = df["tempo_instalacao_real"]
    return X, y


def train_and_evaluate():
    df = fetch_data()
    X, y = prepare_features(df)

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
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n=== Model Evaluation ===")
    print(f"MAE  (Mean Absolute Error): {mae:.2f} minutes")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} minutes")
    print(f"Test samples: {len(y_test)}")
    print(f"Train samples: {len(y_train)}")

    # Save tecnico mapping alongside model
    model_data = {
        "tecnico_map": {"tecnico1": 0, "tecnico2": 1},
        "feature_names": ["m_cabo", "tecnico_enc"],
    }

    model_path = os.path.join(os.path.dirname(__file__), "model.json")
    model.save_model(model_path)

    meta_path = os.path.join(os.path.dirname(__file__), "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(model_data, f, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {meta_path}")

    # Show a few sample predictions
    print("\n=== Sample Predictions ===")
    samples = X_test.head(5)
    actuals = y_test.head(5).values
    preds = model.predict(samples)
    for i, (_, row) in enumerate(samples.iterrows()):
        tecnico_name = "tecnico1" if row["tecnico_enc"] == 0 else "tecnico2"
        print(
            f"  m_cabo={row['m_cabo']:.0f}, tecnico={tecnico_name} -> "
            f"predicted={preds[i]:.1f} min, actual={actuals[i]} min"
        )

    return model


if __name__ == "__main__":
    train_and_evaluate()
