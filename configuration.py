from dataclasses import dataclass
import os.path as osp
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

@dataclass
class TrainConfig:
    mode: str        = 'train'
    
    data_dir: Path   = Path("/home/dev/DATASET/die_casting")
    model_dir: Path  = BASE_DIR / "experiments" / "models" 
    log_dir: Path    = BASE_DIR / "experiments" / "logs"
    
    device: str      = 'cuda'
    num_epochs: int  = 20
    model_name: str  = 'resnet50'
    lr: float        = 1e-4