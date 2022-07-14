import contextlib
import librosa
import librosa.display
from PIL import Image
import numpy as np
import os
import torch
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint  # type: ignore
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger
import ray.tune as tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
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

FREQUENCY_COUNT = 64
SEQUENCE_LENGTH = 173  # 173


class VoiceData(
    Dataset
):  # REVIEW CODE FOR EFFICIENCY!!! (Should some of these be on gpu? Like length and stops?) ---> https://discuss.pytorch.org/t/best-way-to-convert-a-list-to-a-tensor/59949/3
    def __init__(self):
        input_audio_files = os.listdir("SoundReader/Artin")
        _in = [  # Maybe refactor to keep consistent with output? (in = ..., then on a later line do self.input_tensors = in)
            torch.Tensor(audio_to_spectrogram(f"SoundReader/Artin/{voice}"))
            for voice in input_audio_files
        ]

        self.input_tensors = torch.Tensor(
            [
                np.concatenate(
                    (a, np.zeros((SEQUENCE_LENGTH - len(a), FREQUENCY_COUNT)))
                )
                for a in _in
            ]
        )
        print([a.shape for a in self.input_tensors])

        output_audio_files = os.listdir("SoundReader/Ryder")
        out = [
            torch.tensor(
                audio_to_spectrogram(f"SoundReader/Ryder/{voice}", True)
            )  # Should be max(length, Sequence_length-1) clip it at length on less than the sequence length for modfmeol
            for voice in output_audio_files
        ]

        # Initialized list of
        # print(self.lengths)
        self.stops = torch.stack(
            [
                torch.cat(
                    (
                        torch.zeros(len(o)),
                        torch.Tensor([1]),
                        torch.zeros(SEQUENCE_LENGTH - (len(o) + 1)),
                    )
                )
                for o in out
            ]
        )
        self.spec_clipping_masks = torch.stack(
            [
                torch.cat(
                    (
                        torch.zeros(len(o), FREQUENCY_COUNT, dtype=torch.bool),
                        torch.ones(
                            SEQUENCE_LENGTH - len(o),
                            FREQUENCY_COUNT,
                            dtype=torch.bool,
                        ),
                    )
                )
                for o in out
            ]
        )

        self.stops_clipping_masks = torch.stack(
            [
                torch.cat(
                    (
                        torch.zeros(len(o) + 1, dtype=torch.bool),
                        torch.ones(
                            SEQUENCE_LENGTH - (len(o) + 1),
                            dtype=torch.bool,
                        ),
                    )
                )
                for o in out
            ]
        )
        self.output_tensors = torch.stack(
            [
                torch.cat((o, torch.zeros((SEQUENCE_LENGTH - len(o), FREQUENCY_COUNT))))
                for o in out
            ]
        )
        print(self.output_tensors.dtype, self.input_tensors.dtype)
        # print([o.shape for o in self.output_tensors])
        # Padding frames

    def __getitem__(self, index):
        return (
            self.input_tensors[index],
            self.output_tensors[index],
            self.stops[index],
            self.spec_clipping_masks[index],
            self.stops_clipping_masks[index],
        )

    def __len__(self):
        return len(self.input_tensors)


def audio_to_spectrogram(
    name: str,
    out: bool = False,
) -> np.ndarray:  # Get spectrogram and clips to model input size if needed
    y, _ = librosa.load(name)
    stft = librosa.core.stft(y=y, n_fft=512, hop_length=128)
    stft = stft.real
    stft = np.swapaxes(stft, 0, 1)

    # stft -= stft.mean()
    # stft /= stft.std()
    # print(stft.min(), stft.max(), stft.mean())
    # stft = np.tanh(stft)
    # print(stft.shape)
    stft = stft[
        : SEQUENCE_LENGTH - int(out), :FREQUENCY_COUNT
    ]  # Clips to a sequence length of 1 less than the model to allow for concatenation of start token
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


def predict(model: TransformerModel, input_tensor: Tensor, sequence_length, model_dim):
    model.eval()
    sequence = np.zeros((sequence_length - 1, model_dim))

    with torch.no_grad():
        for i in range(sequence_length - 1):
            output_tensor = torch.tensor(sequence).unsqueeze(0).to(model.device)
            spec, stop = model(
                input_tensor.float(), output_tensor.float(), tgt_mask=None
            )
            spec = spec.detach().cpu().squeeze().numpy()
            stop = stop.detach().cpu().squeeze().numpy()
            print(stop[i])
            sequence[i] = np.expand_dims(spec[i], 0)
    return sequence


def get_grad_norm(model_params):
    total_norm = 0
    parameters = [p for p in model_params if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def train(config, gpus=1):
    data = VoiceData()
    dataloader = DataLoader(
        data, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    model = TransformerModel(config, SEQUENCE_LENGTH, FREQUENCY_COUNT)

    # metrics = {"loss": "ptl/train_loss"}  # , "acc": "ptl/val_accuracy"}
    # callbacks = [RichProgressBar(), TuneReportCallback(metrics, on="batch_end")]
    logger = TensorBoardLogger("logs")
    trainer = pl.Trainer(
        callbacks=RichProgressBar(),
        gpus=1,
        logger=logger,
        log_every_n_steps=11,
        max_epochs=500,
    )
    trainer.fit(model, dataloader)

    return model


def main() -> None:
    config = {
        "lr": 3e-4,  # tune.loguniform(1e-4, 1e-1),
        "dropout": 0.0,  # tune.choice([0.1, 0.3, 0.5, 0.7, 0.9]),  # 0.3
        "nhead": 4,  # 1
        "nlayers": 6,  # tune.randint(1, 10),  # 6
        "batch_size": 4,  # tune.choice([2, 4, 6, 8]),  # 4
    }
    # analysis = tune.run(
    #     tune.with_parameters(train, gpus=0),
    #     config=config,
    #     metric="loss",
    #     num_samples=10,
    # )
    # print(analysis.best_config)
    model = train(config)

    input_audio_files = os.listdir("SoundReader/Artin")
    inf = input_audio_files[-1]
    inf_numpy = audio_to_spectrogram(f"SoundReader/Artin/{inf}")
    inf_tensor = torch.tensor(inf_numpy)
    inf_tensor = (
        torch.cat(
            (
                inf_tensor,
                torch.zeros((SEQUENCE_LENGTH - len(inf_tensor), FREQUENCY_COUNT)),
            )
        )
        .unsqueeze(0)
        .to(model.device)
    )
    pred = predict(model, inf_tensor, SEQUENCE_LENGTH, FREQUENCY_COUNT)
    spectrogram_to_image(pred, "inference_image")
    spectrogram_to_audio(pred, "inference_audio.wav", 128, 44100)


if __name__ == "__main__":
    main()
