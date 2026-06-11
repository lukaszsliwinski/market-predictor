
import numpy as np
import pandas as pd
from datetime import datetime, date
from lightgbm import LGBMClassifier
import joblib

from constants import FEATURES


def train(start_train="2024-05-01", end_train=date.today(), report=False):
  # =========================
  # DATA
  # =========================

  df = pd.read_csv("data/data.csv")
  df["Datetime"] = pd.to_datetime(df["Datetime"])

  start_date = pd.to_datetime(start_train).date()
  end_date   = pd.to_datetime(end_train).date()

  df = df[
    (df["Datetime"].dt.date >= start_date) &
    (df["Datetime"].dt.date <= end_date)
  ]

  # =========================
  # FEATURES / TARGET
  # =========================

  X = df[FEATURES]
  y = df["Target"].astype(int)

  # =========================
  # TIME SPLIT
  # =========================

  split = int(len(df) * 0.8)

  X_train, X_test = X.iloc[:split], X.iloc[split:]
  y_train, y_test = y.iloc[:split], y.iloc[split:]

  df_test = df.iloc[split:].copy()

  # =========================
  # MODEL
  # =========================

  model = LGBMClassifier(
    n_estimators=135,
    num_leaves=63,
    learning_rate=0.089,
    max_depth=6,
    random_state=42,
    verbose=-1
  )



  # model = LGBMClassifier(
  #   n_estimators=153,
  #   num_leaves=63,
  #   learning_rate=0.087,
  #   max_depth=6,
  #   random_state=42,
  #   verbose=-1
  # )

  # model = LGBMClassifier(
  #   n_estimators=72,
  #   num_leaves=31,
  #   learning_rate=0.05,
  #   max_depth=11,
  #   random_state=42,
  #   verbose=-1
  # )

  model.fit(X_train, y_train)

  # Rewrite current model
  joblib.dump(model, "models/lgbm_model.pkl")

  # =========================
  # PREDICTIONS
  # =========================

  y_pred = model.predict(X_test)
  proba = model.predict_proba(X_test)[:, 1]

  df_test["pred"] = y_pred
  df_test["proba"] = proba


  if report:
    # =========================
    # EVALUATION
    # =========================

    # 1. Hour-based performance
    hour_stats = df_test.groupby("Hour_NY").apply(
      lambda x: (x["pred"] == x["Target"]).mean()
    )

    print("\nAccuracy by Hour:")
    print(hour_stats.sort_values(ascending=False))


    # 2. Detailed results for selected hours
    selected_hours = [11]
    filtered = df_test[df_test["Hour_NY"].isin(selected_hours)].copy()

    print("\nDetailed predictions:\n")

    for _, row in filtered.iterrows():
      pred = float(row["pred"])
      real = float(row["Target"])

      # Monday=1 ... Friday=5
      weekday_num = row["Session_weekday"]

      status = "correct" if pred == real else "incorrect"

      print(
        f'{row["Date_NY"]} {row["Hour_NY"]}:00 ({weekday_num}) - '
        f'pred: {pred:.1f} / real: {real:.1f} ({status})'
      )


    # 3. Accuracy report
    df = df_test[df_test["Hour_NY"].isin(selected_hours)].copy()
    results = []

    # Statistics
    for wd in range(1, 6):  # MON-FRI
      for hour in selected_hours:

        subset = df[
          (df["Session_weekday"] == wd) &
          (df["Hour_NY"] == hour)
        ]

        for pred_class in [0.0, 1.0]:

          pred_subset = subset[
            subset["pred"] == pred_class
          ]

          if len(pred_subset) == 0:
            acc = np.nan
            count = 0
          else:
            acc = (
              pred_subset["pred"] ==
              pred_subset["Target"]
            ).mean()

            count = len(pred_subset)

          results.append({
            "weekday": wd,
            "hour": hour,
            "pred_class": pred_class,
            "accuracy": acc,
            "count": count
          })

    summary = pd.DataFrame(results)

    # Pivot
    table = summary.pivot_table(
      index=["weekday", "hour"],
      columns="pred_class",
      values="accuracy"
    )

    count_table = summary.pivot_table(
      index=["weekday", "hour"],
      columns="pred_class",
      values="count"
    )

    # Format
    weekday_map = {
      1: "MON",
      2: "TUE",
      3: "WED",
      4: "THU",
      5: "FRI",
    }

    def format_hour(h):
      return f"{h}:00"

    def format_acc(x):
      if pd.isna(x):
        return "X"

      if x >= 0.8:
        return f"{x:.2f}"

      return "X"

    # Accuracy table
    formatted_table = table.copy()

    for col in formatted_table.columns:
      formatted_table[col] = formatted_table[col].apply(format_acc)

    formatted_table = formatted_table.reset_index()

    formatted_table["weekday"] = (
      formatted_table["weekday"]
      .map(weekday_map)
    )

    formatted_table["hour"] = (
      formatted_table["hour"]
      .apply(format_hour)
    )

    formatted_table = formatted_table.rename(columns={
      "weekday": "Weekday",
      "hour": "Hour",
      0.0: "Pred 0",
      1.0: "Pred 1"
    })

    formatted_table = formatted_table.set_index(
      ["Weekday", "Hour"]
    )

    # Count table
    formatted_counts = count_table.copy()

    formatted_counts = formatted_counts.reset_index()

    formatted_counts["weekday"] = (
      formatted_counts["weekday"]
      .map(weekday_map)
    )

    formatted_counts["hour"] = (
      formatted_counts["hour"]
      .apply(format_hour)
    )

    formatted_counts = formatted_counts.rename(columns={
      "weekday": "Weekday",
      "hour": "Hour",
      0.0: "Count 0",
      1.0: "Count 1"
    })

    formatted_counts = formatted_counts.set_index(
      ["Weekday", "Hour"]
    )

    # =========================
    # OUTPUT
    # =========================

    print("\nACCURACY (>= 0.80):\n")
    print(formatted_table)

    print("\nSAMPLE SIZE:\n")
    print(formatted_counts)
