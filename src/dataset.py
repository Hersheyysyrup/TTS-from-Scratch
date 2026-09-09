import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
from tokenizer import tokenizer
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def load_wav(path_to_audio, sr = 22050):

    audio, orig_sr = librosa.load(path_to_audio)

    if sr != orig_sr:
        audio = librosa.resample(audio, orig_freq = orig_sr, new_freq = sr)

    return audio
# converts to decible 
def amp_to_db(x, min_db = -100):
     #Formula = 20*torch.log10(1e-5) = 20* -5 = -100
    clip_val = 10 ** (min_db / 20)
    return 20 * torch.log10(torch.clamp(x, min=clip_val))

def db_to_amp(x):
    return 10**(x/20)

#normalizes the decible value
def normalize(x,
              min_db = -100,
              max_abs_val=4):

    x = (x - min_db) / -min_db
    x = 2 * max_abs_val * x - max_abs_val
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
    return x 

#denormalizes the decible value
def denormalize (x,
                 min_db = -100,
                 max_abs_val=4):

    x = torch.clip(x, min = -max_abs_val, max = max_abs_val)
    x = (x + max_abs_val) / (2 * max_abs_val)
    x = x* -min_db + min_db
    return x

class AudioMelConversion:
    def __init__(self,
                 num_mels = 80,
                 sampling_rate = 22050,
                 n_fft = 1024,
                 window_size = 256,
                 fmin = 0,
                 fmax = 11025,
                 center = False,
                 min_db = -100,
                 max_scaled_abs = 4):  #Nyquist theorem = f max = 1/2(sampling rate)

        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.window_size = window_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        self.spec2mel = self.__get__spec2mel_proj()

    def __get__spec2mel_proj(self):
        mel = librosa.filters.mel( sr = self.sampling_rate,
                                  n_fft = self.n_fft,
                                  n_mels = self.num_mels,
                                  fmin = self.fmin,
                                  fmax = self.fmax)

        return torch.from_numpy(mel)

    def audio2mel(self, audio_ , do_norm = False):

        if not isinstance(audio_, torch.Tensor):
            audio_ = torch.tensor(audio_, dtype = torch.float32)

        spectrogram = torch.stft(audio_,
                               n_fft = self.n_fft,
                               hop_length = self.window_size,
                               win_length = self.window_size,
                               window = torch.hann_window(self.window_size),
                               center = self.center,
                               pad_mode = "reflect",
                               normalized = False,
                                onesided = True,
                               return_complex = True)

        spectrogram = torch.abs(spectrogram)

        mel = torch.matmul(self.spec2mel.to(spectrogram.device), spectrogram)
        mel = amp_to_db(mel, self.min_db)

        if do_norm:
            mel = normalize(mel, min_db = self.min_db, max_abs_val= self.max_scaled_abs)

        return (mel)


if __name__ == "__main__":
    path_to_audio = r"D:\TTS\data\LJSpeech-1.1\wavs\LJ034-0199.wav"
    audio = load_wav(path_to_audio)
    print(audio)

    amc = AudioMelConversion()
    mel = amc.audio2mel(audio, do_norm=True)
    print(mel)
    

