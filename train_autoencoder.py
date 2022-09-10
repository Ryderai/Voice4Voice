import numpy as np
import os
import torch
from pathlib import Path
from torch import tensor
import torch.nn as nn
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint  # type: ignore
from pytorch_lightning.loggers import WandbLogger
from einops.einops import rearrange

from torch.utils.data import Dataset, DataLoader
from autoencoder import AutoEncoder, Encoder, Decoder
from rich.progress import track
from rich import print
from utils import audio_to_spectrogram, spectrogram_to_image, spectrogram_to_audio

TOKEN_SEQUENCE_LENGTH = 8
FREQUENCY_COUNT = 256


class VoiceData(Dataset):
    def __init__(self):
        audio_folder = "flickr_audio/wavs"
        self.audio_tensors = []
        for audio in track(
            os.listdir(audio_folder)[:8000],
            description="Processing data autoencoder...",
        ):
            self.audio_tensors.extend(
                torch.split(
                    torch.Tensor(
                        audio_to_spectrogram(
                            f"{audio_folder}/{audio}", None, FREQUENCY_COUNT
                        )
                    ),
                    TOKEN_SEQUENCE_LENGTH,
                )[:-1]
            )

    def __getitem__(self, index):
        return self.audio_tensors[index]

    def __len__(self):
        return len(self.audio_tensors)


def main() -> None:
    data_path = f"autoencoder_data_{FREQUENCY_COUNT}_{TOKEN_SEQUENCE_LENGTH}.pkl"
    if not os.path.exists(data_path):
        # print("processing data autoencoder...")
        data = VoiceData()
        torch.save(data, data_path)
    else:
        print("LOADING PREVIOUSLY PROCESSED AUTOENCODER DATA")
        data = torch.load(data_path)

    train_size = int(len(data) * 0.9)
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])
    train_dataloader = DataLoader(
        train_data, batch_size=50, shuffle=True, num_workers=0
    )
    val_dataloader = DataLoader(val_data, batch_size=25, shuffle=False, num_workers=0)

    model = AutoEncoder(3, 125)
    # torch.cuda.empty_cache()
    wandb_logger = WandbLogger("VoiceAutoEncoder", project="Voice4Voice")
    wandb_logger.watch(model)
    trainer = pl.Trainer(
        callbacks=[RichProgressBar()], gpus=1, logger=wandb_logger, max_epochs=1
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), "autoencoder.pt")


if __name__ == "__main__":
    main()
