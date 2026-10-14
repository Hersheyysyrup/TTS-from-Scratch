import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from model import Tacotron2, Tacotron2Config
from tokenizer import Tokenizer
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.utils.fetching import LocalStrategy


BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 22050

CHECKPOINT_PATH = (
    BASE_DIR / "work_dir" / "tacotron2" /
    "final_checkpoint" / "pytorch_model.bin"
)

OUTPUT_DIR = BASE_DIR / "generated_speech"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech with Tacotron2 and HiFi-GAN."
    )
    parser.add_argument(
        "text",
        help="The sentence you want the model to speak."
    )
    parser.add_argument(
        "-o", "--output",
        default=str(OUTPUT_DIR / "output.wav"),
        help="Output WAV path (default: generated_speech/output.wav)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum decoder steps."
    )
    args = parser.parse_args()

    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero.")

    tokenizer = Tokenizer()

    unsupported = sorted({
        char for char in args.text
        if char not in tokenizer.char2id
    })
    if unsupported:
        raise ValueError(
            f"Unsupported characters: {unsupported}. "
            "Use characters supported by your tokenizer."
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

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

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading Tacotron2 checkpoint...")
    model = Tacotron2(config)
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    print("Loading pretrained HiFi-GAN...")
    vocoder = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech",
        savedir=str(
            BASE_DIR / "pretrained_models" / "tts-hifigan-ljspeech"
        ),
        run_opts={"device": str(DEVICE)},
        local_strategy=LocalStrategy.COPY,
    )

    print("Input text:", args.text)
    text_tensor = tokenizer.encode(args.text).to(DEVICE)

    with torch.inference_mode():
        mel, attention = model.inference(
            text_tensor,
            max_decode_steps=args.max_steps,
        )

    mel = mel[0].float().detach().cpu().numpy()

    # Reverse the normalization used during Tacotron2 training.
    mel_db = ((np.clip(mel, -4.0, 4.0) + 4.0) / 8.0) * 100.0 - 100.0
    mel_magnitude = np.power(10.0, mel_db / 20.0)
    log_mel = np.log(np.maximum(mel_magnitude, 1e-5))

    mel_tensor = torch.from_numpy(
        log_mel.T.copy()
    ).float().unsqueeze(0).to(DEVICE)

    print("Mel shape:", mel.shape)
    print("Generating audio...")

    with torch.inference_mode():
        waveform = vocoder.decode_batch(mel_tensor)

    audio = waveform.squeeze().detach().float().cpu().numpy()
    audio = np.nan_to_num(audio)

    if audio.size == 0:
        raise RuntimeError("HiFi-GAN returned empty audio.")

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, SAMPLE_RATE)

    print("Duration:", round(len(audio) / SAMPLE_RATE, 2), "seconds")
    print("Saved audio:", output_path.resolve())


if __name__ == "__main__":
    main()
