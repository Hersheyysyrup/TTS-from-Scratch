import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import librosa
from tokenizer import tokenizer
import numpy as np

def load_wav(path_to_audio, sr = 22050):

    audio, orig_sr = torchaudio.load(path_to_audio)

    if sr != orig_sr:
        audio = torchaudio.functional.resample(audio, orig_freq = orig_sr, new_freq = sr)

    return audio.squeeze(0)