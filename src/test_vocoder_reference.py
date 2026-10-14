
import torch
import soundfile as sf
from pathlib import Path
from huggingface_hub import hf_hub_download
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectrogram
from speechbrain.utils.fetching import LocalStrategy

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

print("Loading HiFi-GAN...")
vocoder = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="pretrained_models/tts-hifigan-ljspeech",
    run_opts={"device": device},
    local_strategy=LocalStrategy.COPY,
)

print("Downloading reference audio...")
wav_path = hf_hub_download(
    repo_id="speechbrain/tts-hifigan-ljspeech",
    filename="example.wav",
)

audio, sample_rate = sf.read(wav_path, dtype="float32")
assert sample_rate == 22050, f"Unexpected sample rate: {sample_rate}"

# Mono audio, as expected by the reference preprocessing.
audio_tensor = torch.from_numpy(audio).to(device)

print("Computing reference mel spectrogram...")
mel, _ = mel_spectrogram(
    audio=audio_tensor,
    sample_rate=22050,
    hop_length=256,
    win_length=None,
    n_mels=80,
    n_fft=1024,
    f_min=0.0,
    f_max=8000.0,
    power=1,
    normalized=False,
    min_max_energy_norm=True,
    norm="slaney",
    mel_scale="slaney",
    compression=True,
)

with torch.inference_mode():
    reconstructed = vocoder.decode_batch(mel.unsqueeze(0))

output = reconstructed.squeeze().cpu().numpy()
output_path = Path("vocoder_reference_reconstruction.wav")
sf.write(output_path, output, 22050)

print("Saved:", output_path.resolve())
print("Listen to this reference reconstruction.")

