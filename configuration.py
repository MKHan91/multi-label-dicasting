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
    arch_name: str         = "resnet101"
    train_model_name: str  = f'v2_{arch_name}_v1.1.0'
    
    model_dir: Path  = BASE_DIR / "experiments" / "models" / f"{train_model_name}"
    log_dir: Path    = BASE_DIR / "experiments" / "logs" / f"{train_model_name}"
    code_dir: Path   = BASE_DIR / "experiments" / "codes" / f"{train_model_name}"
    
    num_epochs: int  = 100
    batch_size: int  = 16
    workers: int     = 8
    lr: float        = 1e-4
    
    num_classes: int = 3
    train_thld: float = 0.5
    

@dataclass
class TestConfig:
    model_name: str  = 'v2_resnet50_v1.0.0'
    
    threshold: float = 0.5
    batch_size: int = 16
    workers: int = 8