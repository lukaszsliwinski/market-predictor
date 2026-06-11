import joblib
import pandas as pd

from constants import FEATURES

def predict():
  model = joblib.load("models/lgbm_model.pkl")
  data = pd.read_csv("data/data.csv")

  data = data.iloc[-1:]

  X = data[FEATURES]

  data["Pred"] = model.predict(X)
  return data[["Date_NY", "Hour_NY", "Day_dir_till_hour", "Pred"]]

prediction = predict()
print(prediction)