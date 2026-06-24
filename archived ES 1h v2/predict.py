import joblib
import pandas as pd
import numpy as np

from constants import FEATURES

def predict():
  model = joblib.load("models/lgbm_model.pkl")
  data = pd.read_csv("data/data.csv")

  data = data.iloc[-1:]

  X = data[FEATURES]

  data["Pred"] = model.predict(X)

  data["Pred"] = np.where(
    (data["Pred"] == 0),
    -1.0,
    1.0
  )
  data["Pred_dir"] = np.where(
    data["Day_dir_till_hour"] == data["Pred"],
    1.0,
    -1.0
  ).astype(np.float32)

  return data[["Date_NY", "Hour_NY", "Day_dir_till_hour", "Pred", "Pred_dir"]]

prediction = predict()
print(prediction)