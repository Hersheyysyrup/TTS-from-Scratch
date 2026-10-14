
from email.mime import audio

import torch
import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.utils.fetching import LocalStrategy
import soundfile as sf

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading pretrained HiFi-GAN...")
    vocoder = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="pretrained_models/tts-hifigan-ljspeech",
    run_opts={"device": device},
    local_strategy=LocalStrategy.COPY,
    )

    # Temporary test input: checks the vocoder itself,
    # not the quality of your Tacotron2 output.
    mel = torch.randn(1, 80, 100, device=device)

    print("Input mel shape:", tuple(mel.shape))

    with torch.inference_mode():
        audio = vocoder.decode_batch(mel)

    audio = audio.detach().cpu()
    print("Generated audio shape:", tuple(audio.shape))

    sf.write(
    "vocoder_smoke_test.wav",
    audio.squeeze().numpy(),
    22050,
)

    print("Saved: src/vocoder_smoke_test.wav")
    print("Vocoder test completed.")

if __name__ == "__main__":
    main()
