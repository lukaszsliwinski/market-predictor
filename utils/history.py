from pathlib import Path
import pandas as pd

def history():
  BASE_DIR = Path(__file__).resolve().parent.parent
  DATA_PATH = BASE_DIR / "data" / "data.csv"

  data = pd.read_csv(DATA_PATH)

  return data