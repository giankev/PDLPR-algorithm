import torch
import torch.nn as nn
from pdlpr import PDLPR 
import torch.nn.functional as F

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
print(charset)

# --- Mappature per codifica/decodifica ---
char2idx = {c: i for i, c in enumerate(charset)}
idx2char = {i: c for i, c in enumerate(charset)}


def decode_ccpd_label(label_str, provinces=provinces, alphabets=alphabets, ads=ads):
    """Decodifica stringa del tipo '0_0_22_27_27_33_16' in targa es. '皖AWWX6G' """
    indices = list(map(int, label_str.strip().split('_')))
    if len(indices) != 7:
        raise ValueError("Label must contain 7 indices")

    province = provinces[indices[0]]
    alphabet = alphabets[indices[1]]
    ad_chars = [ads[i] for i in indices[2:]]

    return province + alphabet + ''.join(ad_chars)


def encode_plate(plate_str, char2idx):
    """Converte la stringa '皖AWWX6G' in lista di indici [3, 12, 30, 30, ...]"""
    return [char2idx[c] for c in plate_str]

def greedy_decode(logits, blank_index):
    # logits: (B, T, C) - softmax is not needed here if using argmax
    preds = logits.argmax(dim=2)  # (B, T)
    decoded_batch = []
    for pred in preds:
        chars = []
        prev = None
        for p in pred:
            p = p.item()
            if p != blank_index and p != prev:
                chars.append(idx2char[p])
            prev = p
        decoded_batch.append(''.join(chars))
    return decoded_batch


label_str = "0_0_22_27_27_33_16"

# Decode CCPD
plate_str = decode_ccpd_label(label_str, provinces, alphabets, ads)
print("Decoded plate:", plate_str)

# Encode to indices
encoded = encode_plate(plate_str, char2idx)
print("Encoded indices:", encoded)

# Decode back (test)
reconstructed = ''.join([idx2char[i] for i in encoded])
print("Reconstructed:", reconstructed)



model = PDLPR(num_classes=num_classes) 

batch_size = 1
images = torch.randn(batch_size, 3, 48, 144)  # batch di immagini

logits = model(images)  # (B, T, C)

print("Logits shape:", logits.shape)  # Es. (1, 108, 37)

# Decodifica con greedy
blank_idx = char2idx['-']
decoded = greedy_decode(logits, blank_idx)
print("Decoded plates:", decoded)

y_true = ["0_0_22_27_27_33_16"]

plates = [decode_ccpd_label(y) for y in y_true]

# Encoding targhe come tensori di indici
targets = [torch.tensor(encode_plate(p, char2idx), dtype=torch.long) for p in plates]
target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

# Concatenazione dei target (necessaria per CTC)
targets_concat = torch.cat(targets)



log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)  # (T, B, C)

batch_size = logits.size(0)
input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.long)

ctc_loss_fn = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

loss = ctc_loss_fn(log_probs, targets_concat, input_lengths, target_lengths)

print("Original plates:", plates)
print("Targets concatenati:", targets_concat)
print("Predicted plate string:", decoded[0])
print("Lunghezza predizione:", len(decoded[0]))
print("CTC Loss:", loss.item())