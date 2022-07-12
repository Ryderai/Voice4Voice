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
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(
            -1, 1
        )  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        self.register_buffer("pos_encoding",pos_encoding) # Makes positional encoding apart of model state_dict

    def forward(self, token_embedding: Tensor) -> Tensor:
        return self.dropout(
            token_embedding * math.sqrt(self.dim_model) + self.pos_encoding
        )


class TransformerModel(pl.LightningModule):
    """
    An object containing the architecture for a "Spec2Spec" transformer model. Inherits pl.LightningModule.
    
    Attributes
    ----------
    d_model: int
        The dimensionality of the model.
    ntoken: int
        The number of tokens the model ingests.
    tgt_mask: Torch DERPECITATED MOVED AROUDN REWRTIE YES
        Masks out the future things fron the multiheaded masked attention in the decoder. It's a triangular matrix.
    model_type: str
        
    """
    
    d_model: int
    ntoken: int
    model_type: str = "Transformer"
    lr: float
    prenet: nn.Sequential
    pos_embedding: nn.Embedding
    transformer: nn.Transformer
    pos_emb_residual: PositionalEncoding
    
    def __init__(
        self,
        config: dict,
        ntoken: int,
        d_model: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.ntoken = ntoken

        self.lr = config["lr"]
        dropout = config["dropout"]
        nhead = config["nhead"]
        nlayers = config["nlayers"]
        
        self.start_token_embedding = nn.Linear(1, d_model)

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

        self.mel_linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        
        self.stop_linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # self.fixed_pos_embedding = PositionalEncoding(ntoken, d_model)
        self.pos_emb_residual = PositionalEncoding(d_model, dropout, ntoken)

    def forward(self, src: Tensor, tgt: Tensor, tgt_mask: Tensor) -> Tensor:
        
        src = self.prenet(src).type_as(src)
        tgt = self.prenet(tgt).type_as(tgt)

        tgt = torch.concat(self.start_token_embedding(1), tgt)

        src = self.pos_emb_residual(src).type_as(src)
        tgt = self.pos_emb_residual(tgt).type_as(tgt)

        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        
        out = self.decoder_postnet(out)

        return self.mel_linear(out), self.stop_linear(out)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_tgt_mask(self, size) -> torch.Tensor: # For now its fine having self attention be applied to padding, though should fix in future. Meaning this method should take in the (model sequence length) and the (audio clip sequence length) as paremeters
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, 0.0)

        return mask

    def training_step(self, batch, batch_idx): # Find better naming convention (output or encoder or target input)
        input_tensors, output_tensors, target_stops, lengths = batch
        predicted_specs, predicted_stops = self(input_tensors, output_tensors, self.get_tgt_mask(input_tensors.shape[1]))
        predicted_specs, predicted_stops = [spec[:lengths[i]] for i,spec in enumerate(predicted_specs)], [stop[:lengths[i]] for i,stop in enumerate(predicted_stops)] #Cuts down output from model to corresponding audio clip length (makes sure loss is not caculated across frames where no audio exists)
        return F.mse_loss(predicted_specs, output_tensors) + F.binary_cross_entropy(predicted_stops, target_stops) #https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
