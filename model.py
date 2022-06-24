import math

from torch import nn, Tensor, tensor_split  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer  # type: ignore
from torch.utils.data import Dataset  # type: ignore


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int = 512,
        nhead: int = 8,
        nlayers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.model_type = "Transformer"

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=nlayers,
            num_encoder_layers=nlayers,
            dropout=dropout,
            batch_first=True,
        )

        self.enoder_prenet = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        self.decoder_prenet = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        self.postnet = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )

    def forward(self, src: Tensor, tgt: Tensor, tgt_mask: Tensor) -> Tensor:
        # tgt = self.enoder_prenet(tgt)
        # src = self.decoder_prenet(src)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)

        # out = self.postnet(out)
        return out
