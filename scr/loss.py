import torch
import torch.nn as nn
from pdlpr import PDLPR


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']



def decode_ccpd_label(label_str, provinces, alphabets, ads):
    # Split the string in 7 indices
    indices = list(map(int, label_str.strip().split('_')))
    
    if len(indices) != 7:
        raise ValueError("Label must contain 7 indices")
    
    # Mappa i singoli indici ai caratteri usando le 3 liste
    province = provinces[indices[0]]
    alphabet = alphabets[indices[1]]
    ad_chars = [ads[i] for i in indices[2:]]
    
    # Joints all the charachters
    license_plate = province + alphabet + ''.join(ad_chars)
    return license_plate



def compute_ctc_loss(logits, targets, target_lengths:int = 7):
    """
    logits: [B, T, C] - output from the model
    targets: [sum_target_lengths] - tensor with all target sequences concatenated
    target_lengths: [B] - lengths of each target sequence
    """
    log_probs = logits.permute(1, 0, 2).log_softmax(2)  # [T, B, C]
    input_lengths = torch.full(size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    return loss


def predict(model, images, idx2char):
    model.eval()
    with torch.no_grad():
        logits = model(images)  # [B, T, C]
        log_probs = logits.log_softmax(2)
        preds = torch.argmax(log_probs, dim=2)  # [B, T]

        # Rimuovi blank e caratteri ripetuti (greedy decoding)
        results = []
        for pred in preds:
            seq = []
            prev = -1
            for p in pred:
                p = p.item()
                if p != prev and p != 0:  # skip blank and repeated
                    seq.append(idx2char[p])
                prev = p
            results.append("".join(seq))
    return results



print(len(provinces))
print(len(ads)+len(alphabets))

label = "0_0_22_27_27_33_16"
decoded = decode_ccpd_label(label, provinces, alphabets, ads)
print(decoded)  # → 皖AXYYO7G

# Dummy input
images = torch.randn(4, 3, 48, 144)  # B = 4
targets = torch.tensor([1, 2, 3, 4, 5, 6, 7] * 4, dtype=torch.long)  # tutti con lunghezza 7
target_lengths = torch.full((4,), 7, dtype=torch.long)

# Model
model = PDLPR()
logits = model(images)
pred_logits = logits[:, :7, :]           # (B, 7, 37)
pred_indices = pred_logits.argmax(dim=-1)  # (B, 7)



# Compute loss
loss = compute_ctc_loss(logits, targets, target_lengths)
print("CTC Loss:", loss.item())

