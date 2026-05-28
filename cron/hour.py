import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.features import create_features

# for prediction
create_features(predict=True)

# for history
create_features()
