from dataclasses import dataclass
from datetime import datetime
import os.path as osp
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

now = datetime.now()

@dataclass
class TrainConfig:
    mode: str        = 'train'
    model_name: str  = 'resnet50'
    
    date_name        = now.strftime("%Y%m%d_%H%M%S")
    data_dir: Path   = Path("/content/drive/MyDrive/SEMINAR/DATASET/알루미늄 주조 공정 데이터")
    model_dir: Path  = BASE_DIR / "experiments" / "models" / f"{model_name}_{date_name}"
    log_dir: Path    = BASE_DIR / "experiments" / "logs" / f"{model_name}_{date_name}"
    
    num_epochs: int  = 100
    batch_size: int  = 16
    workers: int     = 8
    lr: float        = 1e-4
    num_classes: int = 3
    device: str      = 'cuda'
    

@dataclass
class TestConfig:
    test_model_name: str  = 'resnet50_20250914_100846'
    test_model_dir : str  =  BASE_DIR / "experiments" / "models" / f"{test_model_name}"
    test_results_dir: str = BASE_DIR / "experiments" / "results" / f"{test_model_name}"
    
    threshold: float      = 0.5