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
    mode: bool  = 'train'


@dataclass
class DataConfig:
    label_csv_name: str     = 'diecasting_w_imc'
    label_list: list[str]   = field(
        default_factory=lambda: [
            'P', 
            'S',
            "IMC"
        ])
    data_dir: Path   = Path(osp.join(os.getcwd(), "dataset"))
    
    
@dataclass
class TrainConfig:
    arch_name: str         = "resnet50"
    train_model_name: str  = 'v1115_v1'
    
    model_dir: Path  = BASE_DIR / "experiments" / "models" / f"{train_model_name}"
    log_dir: Path    = BASE_DIR / "experiments" / "logs" / f"{train_model_name}"
    
    num_epochs: int  = 100
    batch_size: int  = 16
    workers: int     = 8
    lr: float        = 1e-4
    
    num_classes: int = 3
    train_thld: float = 0.5
    

@dataclass
class TestConfig:
    do_infer: bool = False
    
    test_model_name: str  = 'v1115_v1'
    model_dir : str  =  BASE_DIR / "experiments" / "models" / f"{test_model_name}"
    results_dir: str = BASE_DIR / "experiments" / "results" / f"{test_model_name}"
    
    threshold: float      = 0.5
    save_dir: str   = Path("")