import os
import numpy as np
import librosa


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
    # stft = stft[
    #     : SEQUENCE_LENGTH - int(out), :FREQUENCY_COUNT
    # ]  # Clips to a sequence length of 1 less than the model to allow for concatenation of start token
    return stft


female = os.listdir("cmu_arctic/female1")

lengths = []
for f in female:
    lengths.extend(
        (
            audio_to_spectrogram(f"cmu_arctic/female1/{f}").shape[1],
            audio_to_spectrogram(f"cmu_arctic/female2/{f}").shape[1],
        )
    )

print(max(lengths))
