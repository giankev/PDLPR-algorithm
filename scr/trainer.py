import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(train_loader,
              model,
              char2idx,
              device='cuda',
              num_epochs=10,
              lr=1e-3,
              load_checkpoint_path=None,
              save_checkpoint_path=None,
              lr_decay_factor=0.9,
              lr_decay_epochs=20):

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=char2idx['-'], zero_infinity=True)

    start_epoch = 0
    best_loss = float('inf')
    last_decay_epoch = 0

    if load_checkpoint_path is not None and os.path.isfile(load_checkpoint_path):
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['weights'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', best_loss)
        last_decay_epoch = checkpoint.get('last_decay_epoch', 0)
        print(f"Checkpoint caricato da {load_checkpoint_path}, ripartendo dall'epoca {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, label_strings in progress_bar:
            images = images.to(device)

            # Encode targets
            targets = [torch.tensor([char2idx[c] for c in label], dtype=torch.long) for label in label_strings]
            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
            targets_concat = torch.cat(targets).to(device)

            # Forward
            logits = model(images)  # (B, T, C)
            log_probs = logits.log_softmax(2)
            log_probs = log_probs.permute(1, 0, 2)  # (T, B, C)

            input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.long)

            loss = ctc_loss(log_probs, targets_concat, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss medio: {avg_loss:.6f}")

        # Decay learning rate se necessario
        if (epoch + 1) % lr_decay_epochs == 0:
            if avg_loss >= best_loss:
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = old_lr * lr_decay_factor
                    param_group['lr'] = new_lr
                print(f"Learning rate ridotto a {new_lr:.2e} all'epoca {epoch+1} (loss non migliorata)")
                last_decay_epoch = epoch + 1
            else:
                best_loss = avg_loss

        # Salvataggio checkpoint
        if save_checkpoint_path is not None:
            checkpoint = {
                'epoch': epoch + 1,
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
                'last_decay_epoch': last_decay_epoch
            }
            os.makedirs(os.path.dirname(save_checkpoint_path), exist_ok=True)
            torch.save(checkpoint, save_checkpoint_path)
            print(f"Checkpoint salvato in {save_checkpoint_path}")
