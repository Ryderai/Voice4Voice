import math
from typing import Optional

import torch
from torch import nn, Tensor, tensor_split  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer  # type: ignore
from torch.utils.data import Dataset  # type: ignore


# class PositionalEncoding(nn.Module):
#     def __init__(self, max_len, dim_model):
#         super().__init__()

#         division_term = torch.exp(
#             torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
#         )  # 1000^(2i/dim_model)

#         self.pos_encoding = torch.zeros(max_len, dim_model)
#         positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
#         # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
#         self.pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

#         # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
#         self.pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

#     def forward(self) -> Tensor:
#         # Residual connection + pos encoding
#         return self.pos_encoding


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len, device):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        self.pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(
            -1, 1
        )  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        self.pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        self.pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # .expand(3, max_len, dim_model)??
        self.pos_encoding = self.pos_encoding.to(device)

    def forward(self, token_embedding: Tensor) -> Tensor:
        # Residual connection + pos encoding
        # print(token_embedding.shape, self.pos_encoding.shape)
        return self.dropout(token_embedding + self.pos_encoding)


class TransformerModel(nn.Module):
    def __init__(
        self,
        device,
        ntoken: int,
        d_model: int = 512,
        nhead: int = 8,
        nlayers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.ntoken = ntoken
        self.model_type = "Transformer"

        self.decoder_prenet = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.ReLU()
        )

        self.pos_embedding = nn.Embedding(ntoken, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=nlayers,
            num_encoder_layers=nlayers,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder_postnet = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # self.fixed_pos_embedding = PositionalEncoding(ntoken, d_model)
        self.pos_emb_residual = PositionalEncoding(d_model, dropout, ntoken, device)

    def forward(self, src: Tensor, tgt: Tensor, tgt_mask: Tensor) -> Tensor:
        src = self.decoder_prenet(src)
        tgt = self.decoder_prenet(tgt)
        src = self.pos_emb_residual(src)
        tgt = self.pos_emb_residual(tgt)

        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.decoder_postnet(out)
