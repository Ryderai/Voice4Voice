import contextlib
import librosa
import librosa.display
from PIL import Image
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy.io.wavfile import write as waveWrite
import torch.nn as nn
from model import TransformerModel
import matplotlib.pyplot as plt
import itertools
from torchsummary import summary

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()  # type: ignore
    else "cpu"
)

FREQUENCY_COUNT = 64
# START_TOKEN = np.fill(-2, (64))
# STOP_TOKEN = np.fill(5, (64))
SEQUENCE_LENGTH = 30


def audio_to_spectrogram(name: str) -> np.ndarray:
    y, _ = librosa.load(name)
    stft = librosa.core.stft(y=y, n_fft=512, hop_length=128)
    stft = stft.real
    stft = np.swapaxes(stft, 0, 1)

    # stft -= stft.mean()
    # stft /= stft.std()
    print(stft.min(), stft.max(), stft.mean())
    # stft = np.tanh(stft)
    print(stft.min(), stft.max())
    stft = stft[:SEQUENCE_LENGTH, :FREQUENCY_COUNT]

    return stft


def get_tgt_mask(size) -> torch.Tensor:
    mask = torch.tril(torch.ones(size, size) == 1)
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, 0.0)

    return mask


def spectrogram_to_image(transform: np.ndarray, name) -> None:
    img = transform.copy()
    img -= img.min()
    img *= 255 / (img.mean() * 3)
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
    input_hi = audio_to_spectrogram("HiArtinStart.wav")
    output_hi = audio_to_spectrogram("HiRyderStart.wav")

    input_hey = audio_to_spectrogram("ArtinHeyStart.wav")
    output_hey = audio_to_spectrogram("RyderHeyStart.wav")

    input_he = audio_to_spectrogram("ArtinHeStart.wav")
    output_he = audio_to_spectrogram("RyderHeStart.wav")

    inference = audio_to_spectrogram("ArtinTheyStart.wav")
    spectrogram_to_audio(input_hey, "inputReconstructed.wav", 128, 22050)
    spectrogram_to_audio(output_hey, "outputReconstructed.wav", 128, 22050)
    spectrogram_to_image(input_hey, "in")
    spectrogram_to_image(output_hey, "out")
    spectrogram_to_image(inference, "inferneceIN")

    input_tensors = torch.stack(
        (torch.Tensor(input_hi), torch.Tensor(input_hey), torch.Tensor(input_he)), dim=0
    ).to(DEVICE)

    output_tensors = torch.stack(
        (torch.Tensor(output_hi), torch.Tensor(output_hey), torch.Tensor(output_he)),
        dim=0,
    ).to(DEVICE)

    inf_tensor = torch.Tensor(inference).to(DEVICE).unsqueeze(0)

    tgt_mask = get_tgt_mask(input_tensors.size(1) - 1).to(DEVICE)

    model = TransformerModel(
        DEVICE, SEQUENCE_LENGTH - 1, FREQUENCY_COUNT, 1, dropout=0.3
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    writer = SummaryWriter()

    model.train()

    for epoch in range(100000):

        pred = model(input_tensors[:, :-1], output_tensors[:, :-1], tgt_mask)
        # pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, output_tensors[:, 1:])
        writer.add_scalar("Loss/train", loss, epoch)

        if epoch % 100 == 0:
            grad_norm = get_grad_norm(model.parameters())
            writer.add_scalar("Grad/train", grad_norm, epoch)
            print(loss.item(), grad_norm)
        if epoch % 500 == 0:
            spec = predict(
                model, inf_tensor[:, :-1], SEQUENCE_LENGTH - 1, FREQUENCY_COUNT
            )

            spectrogram_to_image(spec, "img1")
            spectrogram_to_audio(spec, "TRANSFORMED.wav", 128, 22050)

            torch.save(model.state_dict(), "model")

        opt.zero_grad()
        loss.backward()
        opt.step()


if __name__ == "__main__":
    main()
