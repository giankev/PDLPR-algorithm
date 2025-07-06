import torch
import torch.nn as nn
from pdlpr import PDLPR 
import torch.nn.functional as F
from trainer import train, set_seed

# Global
seed = 42
num_epochs = 1
lr = 1e-3
lr_decay_factor = 0.9
lr_decay_epochs = 20
save_checkpoint_path = "model"
load_checkpoint_path = None

set_seed(seed)

# --- Definizione delle liste CCPD originali ---
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣",
             "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
             'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
       'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# --- Charset unificato ---
charset = sorted(set(provinces + alphabets + ads))
charset.append('-')  # blank symbol for CTC
num_classes = len(charset)
print("Num classes: ", num_classes)
char2idx = {c: i for i, c in enumerate(charset)}
idx2char = {i: c for i, c in enumerate(charset)}


## Train loader


model = PDLPR(num_classes=num_classes)




train(train_loader, model, char2idx, 'cuda', num_epochs,
      lr, load_checkpoint_path, save_checkpoint_path, lr_decay_factor, lr_decay_epochs)