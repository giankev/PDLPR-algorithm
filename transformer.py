
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len) 
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 1D tensor -> (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) #map same numbers to vector, look up table [vocab_size, d_model]. 

    def forward(self, x): # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper "Attention is all you need"
        return self.embedding(x) * math.sqrt(self.d_model) # (batch, seq_len, d_model)


class CNNBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, alfa = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.LReLU = nn.LeakyReLU(alfa)

    def forward(self, x):
        return self.LReLU(self.bn(self.conv(x)))


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer, z):
            return self.norm(x + self.dropout(sublayer(z))) #add&norm


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int = 1024, h: int = 8, dropout: float= 0.1) -> None:
        super().__init__()
        self.d_model = d_model #Embedding vector size
        self.h = h #number of heads
        
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # query, key, value sono i tensori Q, K, V che arrivano dalla fase di Multi-Head Split
        # le loro dimensioni sono: (batch, h, seq_len, d_k)
        d_k = query.shape[-1]
        # Questo recupera la dimensione dell'ultimo asse del tensore 'query'.
        # Nel nostro caso, query ha dimensioni (batch, h, seq_len, d_k)
        # Quindi, query.shape[-1] restituirà il valore di d_k (la dimensione delle Keys/Queries per ogni head).
        # Esempio: se query è (32, 8, 100, 64), allora d_k sarà 64.

    # 1. Calcolo dei Punteggi di Rilevanza (Scaled Dot-Product)
    # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) --> (batch, h, seq_len, seq_len)
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        #dim=-1: È un modo convenzionale per riferirsi all'ultima dimensione del tensore.
        #"Per ogni vettore lungo l'ultima dimensione (dim=-1), esegui l'operazione softmax indipendentemente. 
        # I valori all'interno di quel vettore devono sommarsi a 1."
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value) #(batch, h, seq_len, d_k)

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        
        # (splittiamo l embedding per ogni head)
        # query.view(...): Questo è un'operazione di reshape. Prendiamo la dimensione d_model e la "rompiamo" 
        #Esempio: (B, S, 512) diventa (B, S, 8, 64).
        # .transpose(1, 2): Scambia la dimensione della seq_len (1) con la dimensione delle h teste (2).
        # Esempio: (B, S, H, Dk) diventa (B, H, S, Dk).
        # in due nuove dimensioni: h (numero di teste) e d_k (dimensione per ogni testa).
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_modelL'ordine (Batch, Sequence Length, Heads, Head Dimension) è più intuitivo per la fase successiva di "combinazione" delle teste.
        
        #  Vogliamo che per ogni token della sequenza, tutti i risultati delle sue H teste siano affiancati. Prima erano H gruppi di S token, ora sono S gruppi di H teste.
       #tensor.view(dim1, dim2, ..., -1, ..., dimN), stai dicendo a PyTorch: "Voglio che questo tensore venga riorganizzato con le dimensioni dim1, dim2, ecc.,
       #  e per la dimensione dove ho messo -1, calcola tu il valore corretto in modo che il numero totale di elementi nel tensore rimanga lo stesso."
        #dim(x)=(batch, h, seq_len, d_k)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        #dim(x)=(batch, seq_len, d_model)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class EncoderBlock(nn.Module):
    
    def __init__(self, MHA_block: MultiHeadAttentionBlock, CNN1_block: CNNBlock, CNN2_block: CNNBlock, dropout):
        super().__init__()
        
        self.CNNBLOCK1 = CNN1_block
        self.MHA = MHA_block
        self.CNNBLOCK2 = CNN2_block
        self.residual_connection =  ResidualConnection(512, dropout)

    def forward(self, x, src_mask):
        z = self.CNNBLOCK1(x) # (B, C=1024, H=6, W=18)
        B, C, H, W = z.shape
        z = z.view(B, C, H * W).permute(0, 2, 1)  # (B, seq_len=108, d_model=1024)
        z = self.MHA(z, z, z, src_mask) # (batch, seq_len, d_model)
        z = z.permute(0, 2, 1).view(B, C, H, W)  # (B, 1024, 6, 18)
        x = self.residual_connection(x,self.CNNBLOCK2, z)
        return x


class Encoder(nn.Module):
    
    def __init__(self, features: int, layers:  nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))
        


class CNNBlock3(nn.Module):
    def __init__(self):
        super().__init__()
        # Come da paper: stride=(3,1), kernel=(2,1), padding=(1,0)
        self.conv = nn.Conv2d(512, 512, kernel_size=(2,1), stride=(3,1), padding=(1,0))
    def forward(self, x):
        return self.conv(x)

class CNNBlock4(nn.Module):
    def __init__(self):
        super().__init__()
        # Come da paper: stride=(3,1), kernel=(1,1), padding=(0,0)
        # Il padding (0,1) nel paper è probabile un typo o un'interpretazione
        # diversa. Con kernel=1, padding=(0,0) è più standard per mantenere la larghezza.
        self.conv = nn.Conv2d(512, 512, kernel_size=(1,1), stride=(3,1), padding=(0,0))
    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    
    def __init__(self, features: int, MHA_mask: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.MHA_mask = MHA_mask
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.MHA_mask(x, x, x, tgt_mask), x)
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask), x)
        x = self.feed_forward_block(x)
        x = self.norm(x)
        
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 tgt_embed: InputEmbeddings, 
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer,
                 src_embed) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(tgt_vocab_size: int,
                      src_seq_len: int = 9,
                      tgt_seq_len: int = 9,
                      src_embed,
                      d_model: int=1024,
                      N: int=3,
                      h: int=8,
                      dropout: float=0.1,
                      d_ff: int=2048) -> Transformer:
    
    # Create the embedding layers
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_cnn_block_1 = CNNBlock(512, 1024)
        encoder_cnn_block_2 = CNNBlock(1024, 512)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, encoder_cnn_block_1, encoder_cnn_block_2, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, tgt_embed, src_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


