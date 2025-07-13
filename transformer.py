
# Transformer implementation 

import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    # (batch, seq_len) -> (batch, seq_len, d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)
    # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class MultiHeadAttention(nn.Module):
    #input: (batch, seq_len, d_model)
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==True, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, _ = self.attention(q, k, v, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, feed_forward, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residuals = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residuals[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residuals[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, feed_forward, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residuals = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.residuals[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residuals[1](x, lambda x: self.cross_attention(x, enc_output, enc_output, src_mask))
        x = self.residuals[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, linear):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.linear = linear

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, tgt_mask, encoder_output, src_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def projection(self, x):
        return self.linear(x)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        # Embedding + position encoding
        src = self.src_embed(src)
        src = self.src_pos(src)

        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)

        # Encoder
        memory = self.encoder(src, src_mask)

        # Decoder
        out = self.decoder(tgt, memory, src_mask, tgt_mask)

        # Project to vocab logits
        return self.linear(out)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len : int, tgt_seq_len : int, d_model : int= 512, N : int = 6, h : int = 8, dropout : float = 0.1, d_ff : int = 2048)-> Transformer:
    #create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    #create positional embedding layers
    src_pos = PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model,tgt_seq_len,dropout)
    #create encoder block
    encoder_list=[]
    for i in range(N):
        encoder_self_attention = MultiHeadAttention(d_model,h,dropout)
        encoder_feed_forward = FeedForward(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention,encoder_feed_forward,dropout)
        encoder_list.append(encoder_block)
    #create decoder block
    decoder_list=[]
    for i in range(N):
        decoder_self_attention = MultiHeadAttention(d_model,h,dropout)
        decoder_cross_attention= MultiHeadAttention(d_model,h,dropout)
        decoder_feed_forward=FeedForward(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention,decoder_cross_attention,decoder_feed_forward,dropout)
        decoder_list.append(decoder_block)
        
    encoder=Encoder(encoder_list)
    decoder=Decoder(decoder_list)
    
    projection_layer= ProjectionLayer(d_model,tgt_vocab_size)

    #create transformer
    transformer= Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer