import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
from tokenizer import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import warnings
from tokenizer import Tokenizer
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

class AudioMelConversions:
    def __init__(self,
                 num_mels = 80,
                 sampling_rate = 22050,
                 n_fft = 1024,
                 window_size = 256,
                 hop_size = 256,
                 fmin = 0,
                 fmax = 11025,
                 center = False,
                 min_db = -100,
                 max_scaled_abs = 4):  #Nyquist theorem = f max = 1/2(sampling rate)

        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        self.spec2mel = self.__get__spec2mel_proj()
        self.mel2spec = torch.linalg.pinv(self.spec2mel)

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
                               hop_length = self.hop_size,
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

    def mel2audio(self, mel, do_norm = False, griffin_lim_iter=50): #griffin lim iter turns a spectrogram back to audio file

        if do_norm:
            mel = denormalize(mel,min_db=self.min_db, max_abs_val=self.max_scaled_abs)

        mel = db_to_amp(mel)

        spectrogram = torch.matmul(self.mel2spec.to(mel.device), mel).cpu().numpy()

        audio = librosa.griffinlim(S = spectrogram,
                                   n_iter = griffin_lim_iter,
                                    hop_length = self.hop_size,
                                     win_length= self.window_size,
                                      n_fft=self.n_fft,
                                       window = "hann" )

        audio *= 32767/ max (0.01, np.max(np.abs(audio)))
        audio = audio.astype(np.int16)
        return audio

class TTSDataset(Dataset):
        def __init__(self,
                     path_to_metadata,
                     sample_rate = 22050,
                     n_fft = 1024,
                     window_size = 256,
                 hop_size = 256,
                     fmin = 0,
                     fmax = 8000,
                     num_mels = 80,
                     center = False,
                     normalized = False,
                     min_db = -100,
                     max_scaled_abs = 4):

            self.metadata = pd.read_csv(path_to_metadata)
            self.sample_rate = sample_rate
            self.n_fft = n_fft
            self.win_size = window_size
            self.hop_size = hop_size
            self.fmin = fmin
            self.fmax = fmax
            self.num_mels = num_mels
            self.center = center
            self.normalized = normalized
            self.min_db = min_db
            self.max_scaled_abs = max_scaled_abs

            self.transcript_length = [len(Tokenizer().encode(t)) for t in self.metadata["normalized_transcript"]]
            self.audio_proc = AudioMelConversions(num_mels = self.num_mels,
                                                  sampling_rate = self.sample_rate,
                                                  n_fft= self.n_fft,
                                                  window_size= self.win_size,
                                                  hop_size = self.hop_size,
                                                  fmin= self.fmin,
                                                  fmax = self.fmax,
                                                  center=self.center,
                                                  min_db= self.min_db,
                                                  max_scaled_abs= self.max_scaled_abs)

        def __len__(self):
            return len(self.metadata)

        def __getitem__(self, idx):

            sample = self.metadata.iloc[idx]

            path_to_audio = sample["file_path"]
            transcript = sample["normalized_transcript"]

            audio = load_wav(path_to_audio, sr = self.sample_rate)
            mel = self.audio_proc.audio2mel(audio, do_norm = True)

            return transcript, mel.squeeze(0)

def build_padding_mask(lengths):

    B = lengths.shape[0]
    T = torch.max(lengths).item()

    mask = torch.zeros(B,T)
    for i in range(B):
        mask[i, lengths[i]:]=1
        return mask.bool()

    
def TTSCollator():
    tokenizer = Tokenizer()

    def _collate_fn(batch):
        texts = [tokenizer.encode(b[0]) for b in batch]
        mels = [b[1] for b in batch]

        input_lengths = torch.tensor([t.shape[0] for t in texts ], dtype = torch.long)
        output_lengths = torch.tensor([m.shape[1] for m in mels], dtype = torch.long)

        input_lengths,  sorted_idx = input_lengths.sort(descending = True)
        texts =[texts[i] for i in sorted_idx]
        mels = [mels[i] for i in sorted_idx]
        output_lengths = output_lengths[sorted_idx]

        text_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first = True, padding_value = tokenizer.pad_token_id)


        max_target_len = max(output_lengths).item()
        num_mels = mels[0].shape[0]

        mel_padded = torch.zeros((len(mels), num_mels, max_target_len))
        gate_padded = torch.zeros((len(mels), max_target_len))

        for i, mel in enumerate(mels):
            t = mel.shape[1]
            mel_padded [i, :, :t] = mel
            gate_padded [i, t-1:] = 1  #does padding with 1 instead of 0

        mel_padded = mel_padded.transpose(1,2)

        return text_padded, input_lengths, mel_padded, gate_padded, build_padding_mask(input_lengths), build_padding_mask(output_lengths)

    return _collate_fn


class BatchSampler:
    def __init__(self, dataset, batch_size, drop_last = False):

        self.sampler = torch.utils.data.SequentialSampler(dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_batches = self.make_batches()

    def make_batches(self):
        indices = [i for i in self.sampler]

        if self.drop_last:
            total_size = (len(indices) // self.batch_size) * self.batch_size
            indices = indices[:total_size]

        batches = [indices[i:i+self.batch_size] for i in range (0, len(indices), self.batch_size)]
        random_indices = torch.randperm(len(batches))
        return [batches[i] for i in random_indices]

    def __iter__(self):
        for batch in self.random_batches:
            yield batch

    def __len__(self):
        return len(self.random_batches)
if __name__ == "__main__":
    path_to_audio = r"D:\TTS\data\LJSpeech-1.1\wavs\LJ034-0199.wav"
    audio = load_wav(path_to_audio)
    print(audio)

    amc = AudioMelConversions()
    mel = amc.audio2mel(audio, do_norm=True)
    print(mel)
    
