import itertools
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# 1. Load data
df = pd.read_csv("es_features.csv")

# Ensure Datetime is not a feature for the model
if 'Datetime' in df.columns:
  df = df.drop(columns=['Datetime'])

# 2. Split into X and y
X = df.drop(columns=['target'])
y = df['target']

# 3. Chronological split (very important in finance!)
# Do not use shuffle=True, the model must not know the future
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.85)

X_train, X_val, X_test = X.iloc[:train_size], X.iloc[train_size:val_size], X.iloc[val_size:]
y_train, y_val, y_test = y.iloc[:train_size], y.iloc[train_size:val_size], y.iloc[val_size:]

print(f"Training size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

# 4. Initialize and train model
model = xgb.XGBClassifier(
  n_estimators=500,
  max_depth=4,
  learning_rate=0.01,
  subsample=0.8,
  colsample_bytree=0.8,
  use_label_encoder=False,
  eval_metric='logloss',
  early_stopping_rounds=50
)

# Train with validation set (last 20% of training data used for monitoring)
model.fit(
  X_train, y_train,
  eval_set=[(X_val, y_val)],
  verbose=False
)

# 5. Evaluation
y_pred = model.predict(X_test)
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

# === PERFORMANCE ANALYSIS BY HOUR (from sin/cos) ===
sin_vals = X_test["sin_hour"].values
cos_vals = X_test["cos_hour"].values

angles = np.arctan2(sin_vals, cos_vals)
hours = ((np.degrees(angles) % 360) / 15).astype(int)

results_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred,
    "hour": hours
})

hour_stats = results_df.groupby("hour").apply(
    lambda x: accuracy_score(x["y_true"], x["y_pred"])
).reset_index()

hour_stats.columns = ["hour", "accuracy"]

print("\n--- ACCURACY PER HOUR ---")
for _, row in hour_stats.sort_values("hour").iterrows():
    print(f"{int(row['hour'])}: {row['accuracy']:.4f}")

# 6. Feature importance (console)
importance = model.get_booster().get_score(importance_type='gain')

importance_df = pd.DataFrame(
    importance.items(),
    columns=['feature', 'importance']
).sort_values(by='importance', ascending=False)

print("\n--- FEATURE IMPORTANCE (TOP 15) ---")
print(importance_df.head(15).to_string(index=False))
