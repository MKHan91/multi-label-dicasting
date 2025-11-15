import os
from dataclasses import dataclass, field
from datetime import datetime
import os.path as osp
from pathlib import Path


label_map = {
    "Normal": [0, 0, 0],
    "P"     : [1, 0, 0],
    "S"     : [0, 1, 0],
    "PS"    : [1, 1, 0],
    "IMC"   : [0, 0, 1],
}

BASE_DIR = Path(__file__).resolve().parent
now = datetime.now()


@dataclass
class BaseConfig:
    mode: str        = 'train'
    device: str      = 'cuda'
    

@dataclass
class DataConfig:
    label_csv_name: str     = 'diecasting_w_imc'
    label_list_w_imc: list[str]   = field(default_factory=['P', 'S', 'IMC'])
    label_list_wo_imc: list[str]   = field(default_factory=['P', 'S'])
    data_dir: Path   = Path(osp.join(osp.dirname(os.getcwd()), "dataset"))
    
    
@dataclass
class TrainConfig:
    train_model_name: str  = 'v1115_v1'
    
    model_dir: Path  = BASE_DIR / "experiments" / "models" / f"{train_model_name}"
    log_dir: Path    = BASE_DIR / "experiments" / "logs" / f"{train_model_name}"
    
    num_epochs: int  = 100
    batch_size: int  = 16
    workers: int     = 8
    lr: float        = 1e-4
    
    num_classes: int = 3
    

@dataclass
class TestConfig:
    test_model_name: str  = 'resnet50_20250914_100846'
    model_dir : str  =  BASE_DIR / "experiments" / "models" / f"{test_model_name}"
    results_dir: str = BASE_DIR / "experiments" / "results" / f"{test_model_name}"
    
    threshold: float      = 0.5