import librosa
from PIL import Image
import numpy as np
from model import TransformerModel

y, r = librosa.load('ryderhi.wav')
stft = librosa.core.spectrum.stft(y, n_fft=512, hop_length=128)
stft = stft.real
stft = np.swapaxes(stft, 0, 1)
# stft = librosa.util.normalize(stft, axis=1)

img = stft.copy()
img -= img.min()
img *= 255/img.max()
img = np.flip(img, 0)
Image.fromarray(img).convert('RGB').save('img'+'.png')