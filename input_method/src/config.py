"""
超参数配置文件
"""
import pandas as pd
from pathlib import Path

# FILE_PATH = Path.cwd()
ROOT_DIR = Path(__file__).parent.parent
# print(ROOT_DIR)
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data"/ "processing"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
print(RAW_DATA_DIR)