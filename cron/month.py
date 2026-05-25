import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.features import create_features
from utils.model import train

create_features(save_cache=True)
train()
