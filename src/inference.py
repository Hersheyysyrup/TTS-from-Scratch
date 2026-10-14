
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model import Tacotron2, Tacotron2Config
from tokenizer import Tokenizer

###configure

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = Path(
    "work_dir/tacotron2/final_checkpoint/pytorch_model.bin"
)

OUTPUT_DIR = Path("inference_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

###Initialize tokenizer

tokenizer = Tokenizer()

print(f"Vocabulary size: {tokenizer.vocab_size}")


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


#load checkpoint

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
    )

model = Tacotron2(config)

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

print("Checkpoint loaded successfully!")


text = "Hello, this is a test."

# Check for unsupported characters.
unsupported = [
    char for char in text
    if char not in tokenizer.char2id
]

if unsupported:
    raise ValueError(
        f"Unsupported characters: {unsupported}. "
        "Use characters supported by your tokenizer."
    )

text_tensor = tokenizer.encode(text).to(DEVICE)

print(f"Input text: {text}")


with torch.inference_mode():
    mel, attention = model.inference(
        text_tensor,
        max_decode_steps=1000,
    )

mel = mel[0].float().cpu().numpy()
attention = attention[0].float().cpu().numpy()

print(f"Generated mel shape: {mel.shape}")
print(f"Generated attention shape: {attention.shape}")


np.save(OUTPUT_DIR / "generated_mel.npy", mel)
np.save(OUTPUT_DIR / "attention.npy", attention)

plt.figure(figsize=(12, 4))
plt.imshow(mel.T, origin="lower", aspect="auto")
plt.xlabel("Time frames")
plt.ylabel("Mel channels")
plt.title("Generated Mel Spectrogram")
plt.colorbar()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mel_spectrogram.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.imshow(attention.T, origin="lower", aspect="auto")
plt.xlabel("Decoder steps")
plt.ylabel("Encoder positions")
plt.title("Attention Alignment")
plt.colorbar()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "attention.png", dpi=150)
plt.close()

print(f"Saved outputs to: {OUTPUT_DIR.resolve()}")
print("Inference completed!")
