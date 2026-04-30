import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load data from CSV
data = pd.read_csv("data.csv")

# Create features and target
feature_cols = [
  "Open", "High", "Low", "Close", "Volume",
  "intraday_range_pct", "gap_pct", "real_body", "candle_dir",
  "ma_5", "ma_10", "ma_20",
  "momentum_5", "momentum_10",
  "volatility_5", "volatility_10", "volatility_20",
  "vol_ratio_5", "vol_ratio_10",
  "rsi_14", "macd"
]
X = data[feature_cols].values
y = data["Target"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create sequences for LSTM (days)
SEQ_LEN = 50
def create_sequences(X, y, seq_len):
  xs, ys = [], []
  for i in range(len(X) - seq_len):
    xs.append(X[i:i+seq_len])
    ys.append(y[i+seq_len])
  return np.array(xs), np.array(ys)
X_seq, y_seq = create_sequences(X, y, SEQ_LEN)

# Test and train data
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# -----------------------------
# 3️⃣ Dataset i DataLoader
# -----------------------------
class TimeSeriesDataset(Dataset):
  def __init__(self, X, y):
    self.X = torch.tensor(X, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.long)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# CNN + LSTM model
class CNN_LSTM(nn.Module):
  def __init__(self, n_features, hidden_dim=64, lstm_layers=1, output_dim=2):
    super(CNN_LSTM, self).__init__()
    self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1)
    self.relu = nn.ReLU()
    self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    # x: [batch, seq_len, features]
    x = x.permute(0, 2, 1)      # [batch, features, seq_len] for Conv1d
    x = self.conv1(x)
    x = self.relu(x)
    x = x.permute(0, 2, 1)      # [batch, seq_len, channels] for LSTM
    lstm_out, _ = self.lstm(x)
    out = self.fc(lstm_out[:, -1, :])  # take the last sequence step
    return out

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_LSTM(n_features=X_train.shape[2]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 150

for epoch in range(EPOCHS):
  model.train()
  train_loss = 0
  for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    out = model(xb)
    loss = criterion(out, yb)
    loss.backward()
    optimizer.step()
    train_loss += loss.item() * xb.size(0)
  train_loss /= len(train_loader.dataset)
  print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}")

# Model evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
  for xb, yb in test_loader:
    xb, yb = xb.to(device), yb.to(device)
    out = model(xb)
    preds = torch.argmax(out, dim=1)
    correct += (preds == yb).sum().item()
    total += yb.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")