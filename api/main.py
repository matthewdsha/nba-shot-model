from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

app = FastAPI(title="NBA Shot Quality API")
model = joblib.load("models/shot_model.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

class ShotRequest(BaseModel):
  shot_distance: float
  loc_x: float
  loc_y: float
  period: int
  clock_seconds: float

@app.post("/predict")
def predict(req: ShotRequest):
  row = {col: 0 for col in feature_cols}
  row["SHOT_DISTANCE"] = req.shot_distance
  row["LOC_X"] = req.loc_x
  row["LOC_Y"] = req.loc_y
  row["PERIOD"] = req.period
  row["CLOCK_SECONDS"] = req.clock_seconds
  df = pd.DataFrame([row])
  prob = model.predict_proba(df)[0][1]
  return {"xFG_pct": round(float(prob), 4)}