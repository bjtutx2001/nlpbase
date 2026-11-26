from pathlib import Path

from nbconvert.exporters.templateexporter import ROOT

#当前脚本完整路径
ROOT_DIR = Path(__file__).parent.parent

#工作目录
ROOT_WORK=Path.cwd()

RAW_DIR = ROOT_DIR / "data"/ "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

SEQ_LEN = 128
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
# LEARNING_RATE = 0.001
EPOCHS = 10