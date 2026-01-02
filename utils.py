import torch
import numpy as np
import pandas as pd
import os
import os.path as osp


miscount = {
    "Normal": {"Normal":0, "P": 0, "S": 0, "PS": 0, "IMC": 0, "S_IMC": 0, "P_IMC": 0, "PS_IMC": 0,},
    "P": {"Normal":0, "P": 0, "S": 0, "PS": 0, "IMC": 0, "S_IMC": 0, "P_IMC": 0, "PS_IMC": 0,},
    "S": {"Normal":0, "P": 0, "S": 0, "PS": 0, "IMC": 0, "S_IMC": 0, "P_IMC": 0, "PS_IMC": 0,},
    "PS": {"Normal":0, "P": 0, "S": 0, "PS": 0, "IMC": 0, "S_IMC": 0, "P_IMC": 0, "PS_IMC": 0,},
    "IMC": {"Normal":0, "P": 0, "S": 0, "PS": 0, "IMC": 0, "S_IMC": 0, "P_IMC": 0, "PS_IMC": 0,},
    
}

def label2str(cfg, src):
    bits = 2 ** torch.arange(src.size(1)-1, -1, -1, device=src.device)
    decimals = (src * bits).sum(dim=1).to(torch.uint8)
    decimals = decimals.cpu().numpy()
    decimals = decimals.astype(str)
    names = [cfg.bit_labels[item] for item in decimals]
    
    # for idx, item in enumerate(decimals):
    #     try:
    #         cfg.bit_labels[item]
    #     except KeyError:
    #         print(idx)
    
    return names


def get_mismatch(labels, preds):
    comp: bool = torch.all(labels == preds, axis=1)
    mislabel = labels[~comp]
    mispred = preds[~comp]
    
    return mislabel, mispred


def get_confusion_matrix(label_names, pred_names):
    for label_name, pred_name in zip(label_names, pred_names):
        miscount[label_name][pred_name] += 1
    
    df = pd.DataFrame(miscount)
    
    return df

def backup_codes(code_dir):
    import shutil

    for item in os.listdir(os.getcwd()):
        if osp.isdir(osp.join(os.getcwd(), item)):
            if item == 'experiments' or item == 'dataset': continue
            
            for item2 in os.listdir(osp.join(os.getcwd(), item)):
                if item2.endswith('.py'):
                    shutil.copy(osp.join(os.getcwd(), item, item2), code_dir)
        else:
            if item.endswith('.py'):
                shutil.copy(osp.join(os.getcwd(), item), code_dir)