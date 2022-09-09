import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from scipy.io.wavfile import write as waveWrite
import torch
import torchaudio
import os


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

def ForwardClassWrapper(func):
    class ClassWrapper(torch.nn.Sequential):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def forward(self, x):
            return func(x, *self.args, **self.kwargs)

        def __call__(self,x,*args,**kwargs):
            return self.forward(x,*args,**kwargs)

    return ClassWrapper

@ForwardClassWrapper        
def FolderToPaths(folder):
    return [f"{folder}/{file}" for file in os.listdir(folder)]

@ForwardClassWrapper
def PathsToSpecs(paths):
    specs = [audio_to_spectrogram(path) for path in paths]
    return specs

@ForwardClassWrapper
def SpecsToTokens(specs, patch_length):
    return [torch.stack(torch.split(spec, patch_length,dim=0)[:-1],dim=0) for spec in specs]

@ForwardClassWrapper
def EncodeTokens(specs, encoder):
    return [encoder(spec) for spec in specs]

@ForwardClassWrapper
def PadToMax(specs, max_length):
    return [torch.cat(
        (spec, torch.zeros(max_length - spec.shape[-3], spec.shape[-2], spec.shape[-1]))
    ) for spec in specs]


def audio_to_spectrogram(name: str) -> torch.Tensor:
    data, sr = torchaudio.load(name)
    data = data.squeeze(0)
    stft = torch.stft(input=data, n_fft=512, hop_length=128,return_complex=True)
    stft = stft.real
    stft = torch.swapaxes(stft,0,1)
    return stft

def split_stack(data,split_length):
    data = torch.split(data, split_length)
    data = torch.stack(data)


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
