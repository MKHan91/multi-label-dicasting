import torch
import numpy as np
import pandas as pd


miscount = {
    "Normal": {"Normal":0, "P": 0, "S": 0, "PS": 0, "IMC": 0},
    "P": {"Normal":0, "P": 0, "S": 0, "PS": 0, "IMC": 0},
    "S": {"Normal":0, "P": 0, "S": 0, "PS": 0, "IMC": 0},
    "PS": {"Normal":0, "P": 0, "S": 0, "PS": 0, "IMC": 0},
    "IMC": {"Normal":0, "P": 0, "S": 0, "PS": 0, "IMC": 0}
}

def label2str(cfg, src):
    bits = 2 ** torch.arange(src.size(1)-1, -1, -1, device=src.device)
    decimals = (src * bits).sum(dim=1).to(torch.float32)
    decimals = decimals.cpu().numpy().astype(np.uint8)
    decimals = decimals.astype(str)
    names = [cfg.bit_labels[item] for item in decimals]
    
    return names


def get_mismatch(labels, preds):
    comp: bool = torch.all(labels == preds, axis=1)
    mislabel = labels[~comp]
    mispred = preds[~comp]
    
    return mislabel, mispred


def count_mismatch(label_names, pred_names):
    for label_name, pred_name in zip(label_names, pred_names):
        miscount[label_name][pred_name] += 1
    
    df = pd.DataFrame(miscount)
    
    return df