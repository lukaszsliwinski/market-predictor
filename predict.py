import joblib
import pandas as pd

model = joblib.load("models/lgbm_model.pkl")

data = pd.read_csv("data/predict.csv")

X = data.drop(
  columns=["Datetime", "Date_NY", "Session_weekday", "Open", "Close", "High", "Low", "Dir", "Day_dir_till_hour", "Target"],
  errors="ignore"
)

data["Pred"] = model.predict(X)
print(data[["Date_NY", "Hour_NY", "Pred"]])