import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from scipy.io.wavfile import write as waveWrite
import torch
import torchaudio


def spectrogram_to_image(transform: np.ndarray, name: str) -> None:
    transform = 2 * np.abs(transform) / np.sum(np.hanning(512))
    transform = np.swapaxes(transform, 0, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(
        librosa.amplitude_to_db(transform),
        ax=ax,
        y_axis="linear",
        x_axis="time",
    )
    fig.savefig(f"{name}.png")
    plt.close()


def audio_to_spectrogram(
    name: str,
    token_length,
    max_frequency_length,
) -> torch.Tensor  # Get spectrogram and clips to model input size if needed
    data, sr = torchaudio.load(name)
    stft = torch.stft(input=data, n_fft=512, hop_length=128,return_complex=True)
    stft = stft.real
    stft = torch.swapaxes(stft, 0, 1)

    return stft


def spectrogram_to_audio(arr: np.ndarray, name: str, hop_length: int, sr: int) -> None:
    # print(arr.min())
    # arr -= arr.min()
    audio = librosa.core.istft(
        np.swapaxes(arr, 0, 1).astype(np.int32), n_fft=512, hop_length=hop_length
    )
    waveWrite(name, sr, audio)


def get_grad_norm(model_params):
    total_norm = 0
    parameters = [p for p in model_params if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm
