import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import math
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model: nn.Module, data_loader, optimizer, device: str, epoch: int = 0):
    model.train()
    criterion = nn.SmoothL1Loss()
    running = 0.0

    pbar = tqdm(data_loader, desc=f" Epoch {epoch}", leave=True)
    for batch_idx, (imgs, targets) in enumerate(pbar):
        imgs  = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        running += batch_loss * imgs.size(0)
        pbar.set_postfix(loss=f"{batch_loss:.6f}")  

    return running / len(data_loader.dataset)

def train(model: nn.Module,
          dl_train,
          dl_val,
          epochs: int,
          lr: float,
          device: str,
          ckpt_path: str | Path = "best_lpdet.pt"):
    
    model.to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.8
    )
    best_val = math.inf
    
    for epoch in range(1, epochs + 1):
        print(f"\n Starting epoch {epoch}/{epochs}")
        loss_tr = train_one_epoch(model, dl_train, opt, device, epoch)
        msg = f" Epoch {epoch:02d}/{epochs} | train_loss: {loss_tr:.6f}"
        if dl_val is not None:
            loss_val = evaluate(model, dl_val, device)
            msg += f" | val_loss: {loss_val:.6f}"
        print(msg)
        scheduler.step()

        if loss_val < best_val:
            best_val = loss_val
            torch.save(model.state_dict(), ckpt_path)
            print(f"saved new best model to {ckpt_path}")

    return model

def evaluate(model: nn.Module, data_loader, device: str):
    model.eval()
    criterion = nn.SmoothL1Loss(reduction="sum")
    loss_sum = 0.0
    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss_sum += criterion(preds, targets).item()
    return loss_sum / len(data_loader.dataset)