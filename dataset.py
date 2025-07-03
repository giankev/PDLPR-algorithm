class CCPDVocabulary:
    def __init__(self):
        self.provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽",
                          "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
                          "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
                          "云", "藏", "陕", "甘", "青", "宁", "新", "警",
                          "学", "O"]
        
        self.alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J',
                          'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
                          'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
        
        self.ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                    'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', 
                    '6', '7', '8', '9', 'O']
        
        unique_chars = sorted(list(set(self.provinces + self.alphabets + self.ads)))
        self.chars = ['<SOS>', '<EOS>', '<PAD>'] + unique_chars

        self.char_to_id = {char: i for i, char in enumerate(self.chars)}
        self.id_to_char = {i: char for i, char in enumerate(self.chars)}

    def get_vocab_size(self):
        return len(self.chars)
    
    def token_to_id(self, token):
        return self.char_to_id.get(token)

    def encode(self, text):
        return [self.token_to_id(char) for char in text]

    def decode_plate_from_filename(self, filename_part):
        parts = filename_part.split('_')
        plate_chars = [
            self.provinces[int(parts[0])],
            self.alphabets[int(parts[1])]
        ]
        plate_chars.extend([self.ads[int(p)] for p in parts[2:7]])
        return "".join(plate_chars)



import torch
from torch.utils.data import Dataset
import os

class CCPDTextProcessor(Dataset):
    def __init__(self,
                 root_dir,
                 vocab,
                 seq_len: int = 7):
        """
        Questa classe processa solo la parte testuale (le targhe) del dataset CCPD.
        Restituisce gli input per il decoder, le label e le maschere, insieme al nome del file
        dell'immagine corrispondente per un abbinamento successivo.

        Args:
            root_dir (str): La cartella che contiene le immagini del CCPD (es. 'ccpd_base').
            vocab (CCPDVocabulary): L'oggetto vocabolario specifico per CCPD.
            seq_len (int): Lunghezza massima della sequenza di output.
        """
        super().__init__()
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)
        self.vocab = CCPDVocabulary()
        self.seq_len = seq_len

        self.sos_token = torch.tensor([self.vocab.token_to_id("<SOS>")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.vocab.token_to_id("<EOS>")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.vocab.token_to_id("<PAD>")], dtype=torch.int64)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        
        try:
            # Estrae la stringa della targa dal nome del file
            tgt_text = self.vocab.decode_plate_from_filename(filename.split('-')[4])
        except (IndexError, ValueError):
            # Se il nome del file è malformato, ne prende un altro a caso
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())
            
        # Tokenizza la stringa della targa
        dec_input_tokens = self.vocab.encode(tgt_text)
        
        # Calcola il numero di token di padding necessari
        num_padding = self.seq_len - len(dec_input_tokens) - 1 # -1 per il <SOS>
        if num_padding < 0:
            # Se la sequenza è troppo lunga, ne prende un'altra
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())

        # Crea l'input per il decoder: <SOS> + targa + <PAD>...
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.pad_token.repeat(num_padding)
        ])

        # Crea la label: targa + <EOS> + <PAD>...
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            self.pad_token.repeat(num_padding)
        ])
        
        # Crea la maschera per il decoder
        # 1. Maschera per il padding
        padding_mask = (decoder_input != self.pad_token).unsqueeze(0) # (1, seq_len)
        # 2. Maschera causale per impedire di "guardare avanti"
        look_ahead_mask = causal_mask(decoder_input.size(0)) # (seq_len, seq_len)
        # 3. Combina le due maschere
        decoder_mask = padding_mask & look_ahead_mask # (1, seq_len, seq_len)

        return {
            "filename": filename,               # Nome del file per caricare l'immagine dopo
            "decoder_input": decoder_input,     # Tensore (seq_len)
            "decoder_mask": decoder_mask,       # Tensore (1, seq_len, seq_len)
            "label": label,                     # Tensore (seq_len)
            "tgt_text": tgt_text,               # Stringa per debug
        }

def causal_mask(size):
    # Genera una maschera triangolare inferiore per l'attenzione causale
    # (seq_len, seq_len)
    return torch.triu(torch.ones(size, size), diagonal=1).eq(0)





