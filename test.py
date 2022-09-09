import librosa
import numpy as np
import scipy
import torch
import torchaudio
import time
import os

# script that times which fft method is faster

# generate random data

start = time.time()
for i in os.listdir("cmu_arctic/female1")[:10]:
    data, sr = torchaudio.load(f"cmu_arctic/female1/{i}")
    data = torch.stft(data, n_fft=512, hop_length=128, return_complex=True)
    data = data.real
    torch.swapaxes(data, 0, 1)
end = time.time()
print("librosa load fft time: ", end - start)


start = time.time()
for i in os.listdir("cmu_arctic/female1")[:10]:
    data, sr = torchaudio.load(f"cmu_arctic/female1/{i}")
    data = torch.stft(data, n_fft=512, hop_length=128, return_complex=False)

    data = data[:][0]

end = time.time()
print("librosa load fft time: ", end - start)

# # librosa load
# start = time.time()
# for i in os.listdir("cmu_arctic/female1")[:10]:
#     data, sr = librosa.core.load(f"cmu_arctic/female1/{i}")
#     data = librosa.core.stft(data, n_fft=512, hop_length=128)
#     data = torch.tensor(data)
# end = time.time()
# print("librosa load fft time: ", end - start)

# # librosa load
# start = time.time()
# for i in os.listdir("cmu_arctic/female1")[:10]:
#     data, sr = librosa.core.load(f"cmu_arctic/female1/{i}")
#     data = torch.tensor(data)
#     data = torch.stft(data, n_fft=512, hop_length=128, return_complex=True)
#     data = data.real
# end = time.time()
# print("librosa load fft time: ", end - start)

# # # torchaudio
# start = time.time()
# for i in os.listdir("cmu_arctic/female1")[:5]:
#     data = torchaudio.load(f"cmu_arctic/female1/{i}")
#     data = torch.stft(data, n_fft=512, hop_length=128)
# end = time.time()
# print("torch load fft time: ", end - start)


# # librosa
# start = time.time()
# for i in data:
#     librosa.core.stft(y=i, n_fft=512, hop_length=128)
# end = time.time()
# print("librosa fft time: ", end - start)
# print(a)
# print(b)
