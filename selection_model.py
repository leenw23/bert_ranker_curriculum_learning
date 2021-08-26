from numpy import float32
import torch
from torch import nn
from transformers import BertModel
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

# Response Ranking Model
class BertSelect(nn.Module):
    def __init__(self, bert: BertModel):
        super(BertSelect, self).__init__()
        self.bert = bert
        self.linear = torch.nn.Linear(768, 1, bias=False)

    def forward(self, ids, mask):
        output, _ = self.bert(ids, mask, return_dict=False)
        cls_ = output[:, 0]
        return self.linear(cls_)

    def get_attention(self, ids, mask):
        output = self.bert(ids, mask, return_dict=True, output_attentions=True)
        prediction = self.linear(output["last_hidden_state"][:, 0])
        return prediction, output["attentions"]


# Transformer-based Ranker for Curriculum Learning
# Paper: Dialogue Response Selection with Hierarchical Curriculum Learning
class TransformerRanker(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, nlayers: int):
        super(TransformerRanker, self).__init__()
        self.c_encoder = TransformerModel(ntoken, d_model, nhead, nlayers)
        self.r_encoder = TransformerModel(ntoken, d_model, nhead, nlayers)

    def forward(self, c_ids, r_ids):
        c_output = self.c_encoder(c_ids)
        r_output = self.r_encoder(r_ids)

        c_output = c_output[:, 0]
        r_output = r_output[:, 0]

        score = torch.matmul(c_output, torch.transpose(r_output, 0, 1))

        return score

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model=256, nhead=8, nlayers=3, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)