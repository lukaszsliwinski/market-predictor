from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "lgbm_model.pkl"
DATA_PATH = BASE_DIR / "data" / "predict.csv"

def predict():
  model = joblib.load(MODEL_PATH)
  data = pd.read_csv(DATA_PATH)

  X = data.drop(
    columns=["Datetime", "Date_NY", "Session_weekday", "Open", "Close", "High", "Low", "Dir", "Day_dir_till_hour", "Target"],
    errors="ignore"
  )

  data["Pred"] = model.predict(X)
  return data[["Date_NY", "Hour_NY", "Day_dir_till_hour", "Pred"]]
