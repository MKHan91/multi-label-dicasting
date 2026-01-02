import os
from dataclasses import dataclass, field
from datetime import datetime
import os.path as osp
from pathlib import Path

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
    bit_labels: dict[str, str] = field(
        default_factory=lambda: {
        "0": "Normal",
        "4": "P",
        "2": "S",
        "6": "PS",
        "1": "IMC",
        "3": "S_IMC",
        "5": "P_IMC",
        "7": "PS_IMC",
        })
    
    data_dir: Path   = Path(osp.join(os.getcwd(), "dataset"))
    
@dataclass
class TrainConfig:
    arch_name: str   = "resnet50"
    model_name: str  = f'v1_ad_{arch_name}_v1.1.7'
    
    model_dir: Path  = BASE_DIR / "experiments" / "models" / f"{model_name}"
    log_dir: Path    = BASE_DIR / "experiments" / "logs" / f"{model_name}"
    code_dir: Path   = BASE_DIR / "experiments" / "codes" / f"{model_name}"
    
    num_epochs: int  = 1600
    batch_size: int  = 8
    workers: int     = 8
    lr: float        = 3e-4
    
    num_classes: int = 3
    train_thld: float = 0.5
    
    @dataclass
    class LossConfig:
        ssim_weight: float = 0.45
        l1_weight: float  = 0.55
    

@dataclass
class TestConfig:
    arch_name: str   = "resnet50"
    model_name: str  = f'v1_ad_{arch_name}_v1.0.5'
    
    model_dir: Path  = BASE_DIR / "experiments" / "models" / f"{model_name}"
    log_dir: Path    = BASE_DIR / "experiments" / "logs" / f"{model_name}"
    
    batch_size: int = 16
    epoch: int = 790
    threshold: float = 0.5
    workers: int = 8
    
    mispred_detail: str = 'Normal'