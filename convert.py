from turtle import forward
import numpy as np
import os
import torch
from pathlib import Path
from torch import Tensor
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint  # type: ignore
from pytorch_lightning.loggers import WandbLogger
from autoencoder import AutoEncoder
import math


# from torch.profiler.profiler import profile
# from torch.profiler.profiler import profile, ProfilerActivity
# from torch.autograd.profiler import record_function

# import ray.tune as tune
# import ray
# from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torch.utils.data import Dataset, DataLoader
from model import TransformerModel
from rich.progress import track
from rich import print
from utils import audio_to_spectrogram, spectrogram_to_image, spectrogram_to_audio
import utils
import torchaudio

# from torchsummary import summary

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()  # type: ignore
    else "cpu"
)

# 1078 is max audio length (in mel frames) from dataset
FREQUENCY_COUNT = 256
PATCH_LENGTH = 8
MAX_SEQUENCE_LENGTH = math.ceil(1078 / PATCH_LENGTH)  # 1078/8 = 134.75
EMBED_SIZE = 125
SAMPLES = 50


class VoiceData(Dataset):
    # REVIEW CODE FOR EFFICIENCY!!! (Should some of these be on gpu? Like length and stops?) ---> https://discuss.pytorch.org/t/best-way-to-convert-a-list-to-a-tensor/59949/3
    def __init__(self):
        female1path = "cmu_arctic/female1"
        female2path = "cmu_arctic/female2"
        ckpt_path = "autoencoder.pt"
        auto_encoder = AutoEncoder(3, EMBED_SIZE)
        auto_encoder.load(ckpt_path)
        encoder = auto_encoder.encoder
        decoder = auto_encoder.decoder

        # covert every wav file in the folder to sequenece of mel spectrograms (ie tokens) then encode each token
        load_and_preprocess = torch.nn.Sequential(
            utils.FolderToPaths(),
            utils.PathsToSpecs(),
            utils.SpecsToTokens(PATCH_LENGTH),
            utils.EncodeTokens(encoder),
            utils.PadToMax(MAX_SEQUENCE_LENGTH),
        )

        self.input_tensors = load_and_preprocess(female1path)
        self.output_tensors = load_and_preprocess(female2path)

    @staticmethod
    def collate_fn(batch):
        # Get rid of PadToMax and add padding only for current batch || DataLoader(dataset,collate_fn=VoiceData.collate_fn)
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

    def __getitem__(self, index):
        return self.input_tensors[index], self.output_tensors[index]

    def __len__(self):
        return len(self.input_tensors)


def predict(model: TransformerModel, input_tensor: Tensor, sequence_length, model_dim):
    model.eval()
    sequence = np.zeros((sequence_length - 1, model_dim))

    with torch.no_grad():
        for i in track(range(sequence_length - 1)):
            output_tensor = torch.tensor(sequence).unsqueeze(0).to(model.device)
            spec, stop = model(
                input_tensor.float(), output_tensor.float(), tgt_mask=None
            )
            spec = spec.detach().cpu().squeeze().numpy()
            stop = stop.detach().cpu().squeeze().numpy()
            # print(f"Stop: {stop[i]}")
            sequence[i] = np.expand_dims(spec[i], 0)
    return sequence


def train(config, train_data, val_data, gpus=1):  # train_data, val_data, gpus):

    train_dataloader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        # collate_fn=VoiceData.collate_fn,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        # collate_fn=VoiceData.collate_fn,
    )
    # with torch.profiler.profiler.profile(
    #     schedule=torch.profiler.profiler.schedule(wait=1, warmup=1, active=5),
    #     on_trace_ready=torch.profiler.profiler.tensorboard_trace_handler(
    #         "./logs/transformerYES"
    #     ),
    #     record_shapes=True,
    # ) as prof:
    model = TransformerModel(config, MAX_SEQUENCE_LENGTH, EMBED_SIZE)

    # name = "".join(f"{k}-{v}_" for k, v in config.items())

    # if os.path.exists(f"logs/{name}"):
    #     print("ALREADY EXISTS!!!")
    #     return
    # else:
    #     logger = TensorBoardLogger("logs", name)
    logger = WandbLogger("Transformer", project="Voice4Voice")
    logger.watch(model)

    trainer = pl.Trainer(
        callbacks=RichProgressBar(),
        gpus=gpus,
        logger=logger,
        log_every_n_steps=11,
        max_epochs=8,
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    return model


def main() -> None:

    data_path = f"data_{FREQUENCY_COUNT}.pt"
    if not os.path.exists(data_path):
        print("processing data...")
        data = VoiceData()
        torch.save(data, data_path)
    else:
        print("LOADING PREVIOUSLY PROCESSED DATA")
        data = torch.load(data_path)

    train_size = int(len(data) * 0.9)
    val_size = len(data) - train_size

    for _ in range(1):
        train_data, val_data = random_split(data, [train_size, val_size])
        print(len(train_data), len(val_data))
        # config = {
        #     "lr": float(np.random.uniform(1e-6, 1e-4)),
        #     "dropout": float(np.random.choice([0.0, 0.1, 0.3, 0.5, 0.7])),
        #     "nhead": int(np.random.choice([1, 2, 4, 8])),
        #     "nlayers": int(np.random.randint(1, 6)),
        #     "batch_size": int(np.random.choice([2, 3, 4])),
        #     "leaky": float(np.random.choice([0.0, 0.01])),
        # }
        config = {
            "lr": 1e-4,
            "dropout": 0.3,
            "nhead": 5,
            "nlayers": 5,
            "batch_size": 4,
            "leaky": 0.01,
            "patch_size": PATCH_LENGTH,
        }

        train(config, train_data, val_data)


if __name__ == "__main__":
    main()
