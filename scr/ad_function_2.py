import torch
import torch.nn as nn
import torch.nn.functional as F
from pdlpr import PDLPR 

# --- Definizione delle liste CCPD originali ---
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣",
             "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
             'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
       'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

total_vocab = len(alphabets) + len(ads) + len(provinces)  # es: 36 + 31 + 26 = 93
num_classes = total_vocab + 1 


def decode_ccpd_label(label_str, provinces=provinces, alphabets=alphabets, ads=ads):
    """Decodifica stringa del tipo '0_0_22_27_27_33_16' in targa es. '皖AWWX6G' """
    indices = list(map(int, label_str.strip().split('_')))
    if len(indices) != 7:
        raise ValueError("Label must contain 7 indices")

    province = provinces[indices[0]]
    alphabet = alphabets[indices[1]]
    ad_chars = [ads[i] for i in indices[2:]]

    return province + alphabet + ''.join(ad_chars)


def encode_ccpd_label(label_str):
    indices = list(map(int, label_str.strip().split('_')))
    return torch.tensor(indices, dtype=torch.long)


# CTC decoding: rimuovi duplicati e blank (index 36)
def ctc_decode(pred, blank=36):
    decoded = []
    previous = blank
    for p in pred:
        if p != blank and p != previous:
            decoded.append(p)
        previous = p
    return decoded



ctc_loss_fn = nn.CTCLoss(blank=36, zero_infinity=True)  # Assumiamo 'O' (ultimo in ogni lista) sia blank

images = torch.randn(1, 3, 48, 144)
y_true = "0_0_22_27_27_33_16"

model = PDLPR(num_classes=37)

logits = model(images)
log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)

targets = encode_ccpd_label(y_true)
input_lengths = torch.tensor([logits.shape[1]])
target_lengths = torch.tensor([len(targets)])

loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
print("CTC Loss:", loss.item())

preds = torch.argmax(log_probs, dim=2).squeeze(1).cpu().numpy().tolist()
decoded_indices = ctc_decode(preds)
print(f"Decoded indices: {decoded_indices}")
decoded_str = decode_ccpd_label('_'.join(map(str, decoded_indices)))
print("Predicted plate string:", decoded_str)


