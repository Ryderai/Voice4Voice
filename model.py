import math
from typing import Tuple

import torch
from torch import nn, Tensor, tensor_split
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import Dataset


class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.model_type = 'Transformer'
        
        
        self.embedding = nn.Embedding(ntoken, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=nlayers,
            num_encoder_layers=nlayers,
            dropout=dropout
        )

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        # src = self.embedding(src) * math.sqrt(self.d_model)
        # tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        
        return self.transformer(src, tgt)