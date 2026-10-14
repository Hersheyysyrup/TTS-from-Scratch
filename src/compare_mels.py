import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.utils.fetching import LocalStrategy

BASE = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mel_path = BASE / "generated_speech" / "tacotron2_mel.npy"
mel = np.load(mel_path).astype(np.float32)

print("Predicted mel shape:", mel.shape)
print("Predicted mel range:", mel.min(), mel.max())

# Reverse the exact normalization used during Tacotron2 training.
mel_db = ((np.clip(mel, -4.0, 4.0) + 4.0) / 8.0) * 100.0 - 100.0
mel_magnitude = np.power(10.0, mel_db / 20.0)

# HiFi-GAN's reference pipeline applies natural-log compression.
log_mel = np.log(np.maximum(mel_magnitude, 1e-5))

# A second candidate: convert predicted features into a linear
# mel magnitude, then apply the reference pipeline's log compression.
# This is numerically the same conversion as above; it is saved
# separately to make the comparison explicit.
candidate = torch.from_numpy(log_mel.T.copy()).unsqueeze(0).to(DEVICE)

print("Vocoder input shape:", tuple(candidate.shape))
print("Vocoder input range:", candidate.min().item(), candidate.max().item())

print("Loading HiFi-GAN...")
vocoder = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir=str(BASE / "pretrained_models" / "tts-hifigan-ljspeech"),
    run_opts={"device": DEVICE},
    local_strategy=LocalStrategy.COPY,
)

with torch.inference_mode():
    waveform = vocoder.decode_batch(candidate)

audio = waveform.squeeze().detach().float().cpu().numpy()
audio = np.nan_to_num(audio)

out = BASE / "generated_speech" / "speech_diagnostic.wav"
sf.write(out, audio, 22050)

print("Saved diagnostic audio:", out)
print("Duration:", round(len(audio) / 22050, 2), "seconds")
print("Peak:", float(np.max(np.abs(audio))))
