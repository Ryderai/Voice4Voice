import librosa
import librosa.display
from PIL import Image
import numpy as np
import torch
from scipy.io.wavfile import write as waveWrite
import torch.nn as nn
from model import TransformerModel
import matplotlib.pyplot as plt

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def audio_to_spectrogram(name: str) -> np.ndarray:
    y, _ = librosa.load(name)
    stft = librosa.core.spectrum.stft(y, n_fft=512)
    stft = stft.real
    stft = np.swapaxes(stft, 0, 1)
    # stft = librosa.util.normalize(stft, axis=1)
    stft -= stft.mean()
    stft /= stft.std()
    stft = stft[:109, :256]

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
    audio = librosa.core.istft(np.swapaxes(arr, 0, 1), hop_length=hop_length)
    waveWrite(name, sr, audio)


def predict(model: TransformerModel, input_tensor, sequence_length, model_dim):
    model.eval()
    sequence = np.zeros((1, model_dim))
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


def main() -> None:
    _input = audio_to_spectrogram("HiArtin.wav")
    output = audio_to_spectrogram("HiRyder.wav")

    spectrogram_to_image(_input, "in")
    spectrogram_to_image(output, "out")

    input_tensor = torch.Tensor(_input).to(DEVICE).unsqueeze(0)
    output_tensor = torch.Tensor(output).to(DEVICE).unsqueeze(0)
    tgt_mask = get_tgt_mask(output_tensor.size(1)).to(DEVICE)

    model = TransformerModel(109, 256, 1, dropout=0).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    model.train()

    for i in range(100000):
        pred = model(input_tensor, output_tensor, tgt_mask)
        # pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, output_tensor)

        if i % 100 == 0:
            print(loss.item())
            print(
                torch.autograd.grad(
                    loss, model.parameters(), allow_unused=True, retain_graph=True
                )
            )

        if i % 500 == 0:
            spec = predict(model, input_tensor, 109, 256)
            # spectrogram_to_image(pred.detach().cpu().squeeze().numpy(), "img1")
            # spectrogram_to_audio(
            #     pred.detach().cpu().squeeze().numpy(), "TRANSFORMED.wav", 128, 22050
            # )
            spectrogram_to_image(spec, "img1")
            spectrogram_to_audio(spec, "TRANSFORMED.wav", 128, 22050)
            torch.save(model.state_dict(), "model")

        opt.zero_grad()
        loss.backward()
        opt.step()


if __name__ == "__main__":
    main()
