
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from model import Tacotron2, Tacotron2Config
from tokenizer import Tokenizer
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectrogram
from speechbrain.utils.fetching import LocalStrategy


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    BASE_DIR
    / "work_dir"
    / "tacotron2"
    / "final_checkpoint"
    / "pytorch_model.bin"
)

OUTPUT_DIR = BASE_DIR / "generated_speech"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050



tokenizer = Tokenizer()

config = Tacotron2Config(
    num_chars=tokenizer.vocab_size,
    character_embed_dim=512,
    pad_token_id=tokenizer.pad_token_id,
    encoder_kernel_size=5,
    encoder_n_convolutions=3,
    encoder_embed_dim=512,
    encoder_dropout_p=0.5,
    decoder_embed_dim=1024,
    decoder_dropout_p=0.1,
    decoder_prenet_dim=256,
    decoder_prenet_depth=2,
    decoder_prenet_dropout_p=0.5,
    decoder_postnet_num_convs=5,
    decoder_postnet_n_filters=512,
    decoder_postnet_kernel_size=5,
    decoder_postnet_dropout_p=0.5,
    attention_dim=128,
    attention_dropout_p=0.1,
    attention_location_n_filters=32,
    attention_location_kernel_size=31,
)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("\nLoading Tacotron2 checkpoint...")

model = Tacotron2(config)

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

print("Tacotron2 loaded successfully!")


print("\nLoading pretrained HiFi-GAN...")

vocoder = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir=str(BASE_DIR / "pretrained_models" / "tts-hifigan-ljspeech"),
    run_opts={"device": str(DEVICE)},
    local_strategy=LocalStrategy.COPY,
)

print("HiFi-GAN loaded successfully!")


text = "Hello, I am Aarav, and I am a first-year student."

unsupported = [
    char for char in text
    if char not in tokenizer.char2id
]

if unsupported:
    raise ValueError(f"Unsupported characters: {unsupported}")

text_tensor = tokenizer.encode(text).to(DEVICE)

print("\nInput text:", text)

with torch.inference_mode():
    mel, attention = model.inference(
        text_tensor,
        max_decode_steps=1000,
    )

    
print("Generated decoder steps:", attention.shape[1])


attn = attention[0].detach().float().cpu().numpy()

# Plot decoder steps against input characters
plt.figure(figsize=(14, 6))
plt.imshow(attn.T, aspect="auto", origin="lower", interpolation="nearest")
plt.xlabel("Decoder timestep")
plt.ylabel("Input token position")
plt.title("Tacotron2 Attention Alignment")
plt.colorbar(label="Attention weight")
plt.tight_layout()

plot_path = OUTPUT_DIR / "attention.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print("Attention plot saved to:", plot_path)


mel = mel[0].float().detach().cpu().numpy()

print("\n--- Raw Tacotron2 mel diagnostics ---")
print("Shape:", mel.shape)
print("Minimum:", float(mel.min()))
print("Maximum:", float(mel.max()))
print("Mean:", float(mel.mean()))
print("Standard deviation:", float(mel.std()))
print("Values below -4:", float((mel < -4).mean() * 100), "%")
print("Values above 4:", float((mel > 4).mean() * 100), "%")

np.save(OUTPUT_DIR / "tacotron2_mel.npy", mel)

np.save(
    OUTPUT_DIR / "attention.npy",
    attention.detach().cpu().numpy(),
)


mel_db = ((np.clip(mel, -4.0, 4.0) + 4.0) / 8.0) * 100.0 - 100.0

mel_magnitude = np.power(10.0, mel_db / 20.0)

print("\n--- Converted mel diagnostics ---")
print("Shape:", mel_magnitude.shape)
print("Minimum:", float(mel_magnitude.min()))
print("Maximum:", float(mel_magnitude.max()))
print("Mean:", float(mel_magnitude.mean()))




mel_tensor = torch.from_numpy(
    np.log(np.maximum(mel_magnitude, 1e-5)).T.copy()
).float().unsqueeze(0).to(DEVICE)


print("\n--- HiFi-GAN input diagnostics ---")
print("Shape:", tuple(mel_tensor.shape))
print("Minimum:", mel_tensor.min().item())
print("Maximum:", mel_tensor.max().item())
print("Mean:", mel_tensor.mean().item())
print("Standard deviation:", mel_tensor.std().item())


print("\nGenerating audio...")

with torch.inference_mode():
    audio = vocoder.decode_batch(mel_tensor)

audio = audio.squeeze().detach().float().cpu().numpy()
audio = np.nan_to_num(audio)

if audio.size == 0:
    raise RuntimeError("HiFi-GAN returned empty audio.")

print("Audio samples:", len(audio))
print("Audio peak amplitude:", float(np.max(np.abs(audio))))
print("Audio duration:", round(len(audio) / SAMPLE_RATE, 2), "seconds")

output_path = OUTPUT_DIR / "speech.wav"

sf.write(
    output_path,
    audio,
    SAMPLE_RATE,
)

print("\nSaved audio:", output_path.resolve())
print("Speech generation completed.")
