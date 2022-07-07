import contextlib
import librosa
import librosa.display
from PIL import Image
import numpy as np
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint  # type: ignore
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from scipy.io.wavfile import write as waveWrite
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import TransformerModel
import matplotlib.pyplot as plt
from rich.progress import track
import itertools

# from torchsummary import summary

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()  # type: ignore
    else "cpu"
)

FREQUENCY_COUNT = 256
# START_TOKEN = np.fill(-2, (FREQUENCY_COUNT))
# STOP_TOKEN = np.fill(5, (FREQUENCY_COUNT))
SEQUENCE_LENGTH = 173
BATCH_SIZE = 4


class VoiceData(Dataset):
    def __init__(self):
        input_audio_files = os.listdir("SoundReader/Artin")
        _input = [
            torch.Tensor(audio_to_spectrogram(f"SoundReader/Artin/{voice}"))
            for voice in input_audio_files[:-1]
        ]
        self.input_tensors = torch.stack(_input, dim=0)

        output_audio_files = os.listdir("SoundReader/Ryder")
        output = [
            torch.Tensor(audio_to_spectrogram(f"SoundReader/Ryder/{voice}"))
            for voice in output_audio_files[:-1]
        ]
        self.output_tensors = torch.stack(output, dim=0)

    def __getitem__(self, index):
        return self.input_tensors[index], self.output_tensors[index]

    def __len__(self):
        return self.input_tensors.shape[0]


def audio_to_spectrogram(name: str) -> np.ndarray:
    y, _ = librosa.load(name)
    stft = librosa.core.stft(y=y, n_fft=512, hop_length=128)
    stft = stft.real
    stft = np.swapaxes(stft, 0, 1)

    # stft -= stft.mean()
    # stft /= stft.std()
    # print(stft.min(), stft.max(), stft.mean())
    # stft = np.tanh(stft)
    # print(stft.shape)
    stft = stft[:SEQUENCE_LENGTH, :FREQUENCY_COUNT]

    return stft


def spectrogram_to_image(transform: np.ndarray, name: str) -> None:
    img = transform.copy()
    # img -= img.min()
    # img *= 255 / (img.mean() * 3)
    img *= 100
    img = np.where(img > 254, 0, img)
    img = np.swapaxes(img, 0, 1)
    img = np.flip(img, 0)
    Image.fromarray(img).convert("RGB").save(f"{name}.png")


def spectrogram_to_audio(arr: np.ndarray, name: str, hop_length: int, sr: int) -> None:
    # print(arr.min())
    # arr -= arr.min()
    audio = librosa.core.istft(
        np.swapaxes(arr, 0, 1).astype(np.int32), n_fft=512, hop_length=hop_length
    )
    waveWrite(name, sr, audio)


def predict(model: TransformerModel, input_tensor, sequence_length, model_dim):
    model.eval()
    sequence = np.zeros((1, model_dim))
    with torch.no_grad():
        for i in range(1, sequence_length):
            padding = np.zeros((sequence_length - i, model_dim))
            output = np.concatenate((sequence, padding))
            output_tensor = torch.Tensor(output).to(DEVICE).unsqueeze(0)
            result = (
                model(input_tensor, output_tensor, tgt_mask=None)
                .detach()
                .cpu()
                .squeeze()
                .numpy()
            )
            sequence = np.concatenate((sequence, np.expand_dims(result[i - 1], 0)))
    model.train()
    return sequence


def get_grad_norm(model_params):
    total_norm = 0
    parameters = [p for p in model_params if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def main() -> None:

    # inf = input_audio_files[-1]
    # inf_numpy = audio_to_spectrogram(f"SoundReader/Artin/{inf}")
    # inf_tensor = torch.Tensor(inf_numpy).unsqueeze(0).to(DEVICE)

    # inf_out = output_audio_files[-1]
    # inf_out_numpy = audio_to_spectrogram(f"SoundReader/Ryder/{inf_out}")
    # inf_out_tensor = torch.Tensor(inf_out_numpy).unsqueeze(0).to(DEVICE)

    # spectrogram_to_image(inf_numpy, "inf_spec")
    # spectrogram_to_image(inf_out_numpy, "inf_out_spec")

    data = VoiceData()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    model = TransformerModel(SEQUENCE_LENGTH - 1, FREQUENCY_COUNT, 1, dropout=0.3)

    trainer = pl.Trainer(callbacks=[RichProgressBar()], gpus=1, log_every_n_steps=11)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
