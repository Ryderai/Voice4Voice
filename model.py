import math
from typing import Optional

import torch
from torch import device, nn, Tensor, tensor_split  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer  # type: ignore
import pytorch_lightning as pl
from torch.utils.data import Dataset


class PositionalEncoding(pl.LightningModule):
    dropout: nn.Dropout
    pos_encoding: Tensor

    def __init__(self, dim_model: int, dropout_p: float, max_len: int):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dim_model = dim_model
        self.dev = device
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
        self.pos_encoding = self.pos_encoding

    def forward(self, token_embedding: Tensor) -> Tensor:
        return self.dropout(
            token_embedding * math.sqrt(self.dim_model) + self.pos_encoding
        )


class TransformerModel(pl.LightningModule):
    def __init__(
        self,
        config,
        ntoken: int,
        d_model: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.ntoken = ntoken
        self.tgt_mask = self.get_tgt_mask(ntoken)
        self.model_type = "Transformer"

        self.lr = config["lr"]
        dropout = config["dropout"]
        nhead = config["nhead"]
        nlayers = config["nlayers"]

        self.prenet = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
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
        self.pos_emb_residual = PositionalEncoding(d_model, dropout, ntoken)

    def forward(self, src: Tensor, tgt: Tensor, tgt_mask: Tensor) -> Tensor:
        # src = self.prenet(src)
        # tgt = self.prenet(tgt)
        src = self.pos_emb_residual(src)
        tgt = self.pos_emb_residual(tgt)

        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.decoder_postnet(out)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss_fn(self, input: Tensor, target: Tensor):
        return F.mse_loss(input, target)

    def get_tgt_mask(self, size) -> torch.Tensor:
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, 0.0)

        return mask

    def training_step(self, batch, batch_idx):
        input_tensors, output_tensors = batch
        pred = self(input_tensors[:, :-1], output_tensors[:, :-1], self.tgt_mask)
        return self.loss_fn(pred, output_tensors[:, 1:])
