import numpy as np
import os
import torch
from pathlib import Path
from torch import Tensor
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint  # type: ignore
from pytorch_lightning.loggers import TensorBoardLogger
import ray.tune as tune
import ray
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torch.utils.data import Dataset, DataLoader
from model import TransformerModel
from rich.progress import track
from rich import print
from utils import audio_to_spectrogram, spectrogram_to_image, spectrogram_to_audio

# from torchsummary import summary

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()  # type: ignore
    else "cpu"
)

FREQUENCY_COUNT = 64
SEQUENCE_LENGTH = 1078  # 173


class VoiceData(
    Dataset
):  # REVIEW CODE FOR EFFICIENCY!!! (Should some of these be on gpu? Like length and stops?) ---> https://discuss.pytorch.org/t/best-way-to-convert-a-list-to-a-tensor/59949/3
    def __init__(self):
        female1path = "E:/Code/Python/Voice4Voice/cmu_arctic/female1"
        female2path = "E:/Code/Python/Voice4Voice/cmu_arctic/female2"
        # input_audio_files = os.listdir("SoundReader/Artin")
        input_audio_files = os.listdir(female1path)
        _in = [  # Maybe refactor to keep consistent with output? (in = ..., then on a later line do self.input_tensors = in)
            torch.Tensor(
                audio_to_spectrogram(
                    f"{female1path}/{voice}", SEQUENCE_LENGTH, FREQUENCY_COUNT
                )
            )
            for voice in input_audio_files[:-1]
        ]

        self.input_tensors = torch.Tensor(
            np.array(
                [
                    np.concatenate(
                        (a, np.zeros((SEQUENCE_LENGTH - len(a), FREQUENCY_COUNT)))
                    )
                    for a in _in
                ]
            )
        )
        # print([a.shape for a in self.input_tensors])

        # output_audio_files = os.listdir("SoundReader/Ryder")
        output_audio_files = os.listdir(female2path)
        out = [
            torch.tensor(
                audio_to_spectrogram(
                    f"{female2path}/{voice}", SEQUENCE_LENGTH - 1, FREQUENCY_COUNT
                )  # Clips to a sequence length of 1 less than the model to allow for concatenation of start token
            )
            for voice in output_audio_files[:-1]
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
        # print(self.output_tensors.dtype, self.input_tensors.dtype)
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
            print(f"Stop: {stop[i]}")
            sequence[i] = np.expand_dims(spec[i], 0)
    return sequence


def train(config, gpus=1):  # train_data, val_data, gpus):

    data = VoiceData()
    train_size = int(len(data) * 0.8)
    val_size = len(data) - train_size

    train_data, val_data = random_split(data, [train_size, val_size])

    train_dataloader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_dataloader = DataLoader(
        val_data, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    model = TransformerModel(config, SEQUENCE_LENGTH, FREQUENCY_COUNT)

    metrics = {"val_loss": "val_loss"}
    callbacks = [RichProgressBar(), TuneReportCallback(metrics, on="batch_end")]
    logger = TensorBoardLogger("logs")
    trainer = pl.Trainer(
        callbacks=callbacks,
        gpus=gpus,
        logger=logger,
        log_every_n_steps=11,
        max_epochs=15,
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    return model


def main() -> None:
    # config = {
    #     "lr": 3e-4,
    #     "dropout": 0.0,
    #     "nhead": 4,
    #     "nlayers": 6,
    #     "batch_size": 8
    # }
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "dropout": tune.choice([0.1, 0.3, 0.5, 0.7, 0.9]),  # 0.3
        "nhead": tune.choice([1, 2, 4, 8]),
        "nlayers": tune.randint(1, 10),  # 6
        "batch_size": tune.choice([2, 4, 6, 8]),  # 4
    }

    # data_path = f"data_{FREQUENCY_COUNT}.pt"
    # if not os.path.exists(data_path):
    #     print("processing data...")
    #     data = VoiceData()
    #     torch.save(data, data_path)
    # else:
    #     print("LOADING PREVIOUSLY PROCESSED DATA")
    #     data = torch.load(data_path)

    # train_size = int(len(data) * 0.8)
    # val_size = len(data) - train_size

    # train_data, val_data = random_split(data, [train_size, val_size])
    ray.init(local_mode=True)
    analysis = tune.run(
        tune.with_parameters(
            train, gpus=1
        ),  # train_data=train_data, val_data=val_data, gpus=1),
        config=config,
        metric="loss",
        num_samples=10,
        resources_per_trial={"gpu": 1},
    )
    print(analysis.best_config)
    # model = train(config)

    # input_audio_files = os.listdir("cmu_arctic/female1")
    # inf = input_audio_files[-2]
    # inf_numpy = audio_to_spectrogram(
    #     f"cmu_arctic/female1/{inf}", SEQUENCE_LENGTH, FREQUENCY_COUNT
    # )
    # spectrogram_to_image(inf_numpy, "inference_input_image")
    # spectrogram_to_audio(inf_numpy, "inference_input_audio.wav", 128, 22050)

    # inf_tensor = torch.tensor(inf_numpy)
    # inf_tensor = (
    #     torch.cat(
    #         (
    #             inf_tensor,
    #             torch.zeros((SEQUENCE_LENGTH - len(inf_tensor), FREQUENCY_COUNT)),
    #         )
    #     )
    #     .unsqueeze(0)
    #     .to(model.device)
    # )

    # pred = predict(model, inf_tensor, SEQUENCE_LENGTH, FREQUENCY_COUNT)
    # spectrogram_to_image(pred, "inference_predict_image")
    # spectrogram_to_audio(pred, "inference_predict_audio.wav", 128, 22050)

    # output_audio_files = os.listdir("cmu_arctic/female2")
    # inf = output_audio_files[-2]
    # inf_numpy = audio_to_spectrogram(
    #     f"cmu_arctic/female2/{inf}", SEQUENCE_LENGTH, FREQUENCY_COUNT
    # )
    # spectrogram_to_image(inf_numpy, "inference_target_image")
    # spectrogram_to_audio(inf_numpy, "inference_target_audio.wav", 128, 22050)


if __name__ == "__main__":
    main()
