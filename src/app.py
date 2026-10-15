import html
import io
import os
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SYNTHESIS_SCRIPT = BASE_DIR / "synthesize.py"

MAX_CHARS = 2000
TIMEOUT_S = 600

SAMPLE_TEXT = "Hello! Welcome to my text to speech application."


st.set_page_config(
    page_title="Text to Speech Generator",
    layout="centered",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap');

:root {
    --indigo: #5B4DB1;
    --indigo-deep: #463A9C;
    --violet: #7C6CF0;
    --orchid: #B26EE0;
    --ink: #2E2A5E;
    --slate: #6B6A8A;
    --lilac: #F1EDFD;
    --line: #DAD2F7;
    --mint: #34C38F;
    --alert: #E5484D;
}

/* ---------- page and card ---------- */
.stApp {
    background: linear-gradient(135deg, #EBDCFB 0%, #E4E4FC 52%, #DCE8FB 100%) !important;
    color-scheme: light;
}

[data-testid="stHeader"] {
    background: transparent;
}

.stApp p, .stApp label, .stApp button,
.stApp textarea, .stApp input {
    font-family: 'Instrument Sans', sans-serif;
}

.stApp h1, .stApp h2, .stApp h3 {
    font-family: 'Newsreader', serif;
    font-weight: 600;
    letter-spacing: -0.015em;
}

.stApp .block-container {
    max-width: 46rem;
    margin: 3rem auto 2rem auto;
    padding: 2.6rem 2.8rem 2.4rem 2.8rem;
    background: #FFFFFF;
    border-radius: 28px;
    box-shadow: 0 28px 70px rgba(91, 77, 177, 0.20);
}

/* ---------- masthead ---------- */
.stApp .masthead h1 {
    font-size: 2.6rem;
    line-height: 1.1;
    margin: 0 0 .5rem 0;
    padding: 0;
    color: var(--indigo);
}

.stApp .masthead p {
    color: var(--slate);
    font-size: 1.05rem;
    margin: 0 0 1.6rem 0;
    max-width: 32rem;
}

.status {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    font-size: .82rem;
    font-weight: 500;
    color: var(--indigo);
    background: var(--lilac);
    padding: .3rem .8rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}

.status .dot {
    width: .55rem;
    height: .55rem;
    border-radius: 50%;
    background: var(--alert);
}

.status.live .dot {
    background: var(--mint);
    box-shadow: 0 0 0 4px rgba(52, 195, 143, .2);
}

/* ---------- text box ---------- */
.stTextArea label p {
    color: var(--ink);
    font-weight: 600;
}

.stTextArea [data-baseweb="textarea"],
.stTextArea [data-baseweb="base-input"] {
    background: var(--lilac);
    border: 1.5px solid var(--line);
    border-radius: 16px;
}

.stTextArea [data-baseweb="textarea"]:focus-within {
    border-color: var(--violet);
    box-shadow: 0 0 0 3px rgba(124, 108, 240, .18);
}

.stTextArea textarea {
    background: transparent !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    caret-color: #FFFFFF !important;
    font-family: 'Newsreader', serif;
    font-size: 1.15rem;
    line-height: 1.55;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--slate);
}

[data-testid="stSpinner"],
[data-testid="stSpinner"] p {
    color: var(--ink);
}

/* ---------- buttons ---------- */
.stButton button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, var(--violet), var(--orchid));
    color: #FFFFFF;
    border: none;
    border-radius: 14px;
    font-weight: 600;
    padding: .7rem 1rem;
    box-shadow: 0 10px 24px rgba(124, 108, 240, .35);
}

.stButton button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    filter: brightness(1.06);
    color: #FFFFFF;
}

[data-testid="stDownloadButton"] button {
    background: #FFFFFF;
    color: var(--indigo);
    border: 1.5px solid var(--line);
    border-radius: 14px;
    font-weight: 600;
    padding: .7rem 1rem;
}

[data-testid="stDownloadButton"] button:hover {
    background: var(--lilac);
    border-color: var(--violet);
    color: var(--indigo);
}

audio {
    width: 100%;
    color-scheme: light;
}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--indigo) 0%, var(--indigo-deep) 100%);
    border-right: none;
}

[data-testid="stSidebar"] > div:first-child {
    background: transparent;
}

[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] svg {
    color: #FFFFFF;
    fill: #FFFFFF;
}

.sidebar-brand .logo {
    width: 2.8rem;
    height: 2.8rem;
    display: grid;
    place-items: center;
    font-size: 1.4rem;
    border-radius: 14px;
    background: linear-gradient(135deg, var(--violet), var(--orchid));
    box-shadow: 0 10px 24px rgba(0, 0, 0, .18);
    margin-bottom: 1rem;
}

[data-testid="stSidebar"] .sidebar-brand h2 {
    color: #FFFFFF;
    font-size: 1.7rem;
    line-height: 1.15;
    margin: 0 0 .6rem 0;
    padding: 0;
}

[data-testid="stSidebar"] .sidebar-brand p {
    color: rgba(255, 255, 255, .78);
    font-size: .95rem;
    line-height: 1.55;
    margin: 0 0 1rem 0;
}

.sidebar-brand .chips {
    display: flex;
    flex-wrap: wrap;
    gap: .4rem;
    margin-bottom: 1.6rem;
}

.sidebar-brand .chips span {
    background: rgba(255, 255, 255, .14);
    color: #FFFFFF;
    font-size: .78rem;
    font-weight: 500;
    padding: .25rem .7rem;
    border-radius: 999px;
}

[data-testid="stSidebar"] .sidebar-note {
    border-top: 1px solid rgba(255, 255, 255, .18);
    padding-top: .9rem;
    color: rgba(255, 255, 255, .62);
    font-size: .78rem;
    line-height: 1.5;
    word-break: break-all;
}

@media (prefers-reduced-motion: reduce) {
    * {
        transition: none !important;
        animation: none !important;
    }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


class SynthesisError(Exception):
    """Raised when speech generation fails."""



def prepare_text(text: str) -> str:
    """Normalize text for the model's supported character set."""
    # Convert curly punctuation to standard ASCII punctuation.
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-")

    # Expand digits into words.
    digit_words = {
        "0": " zero ",
        "1": " one ",
        "2": " two ",
        "3": " three ",
        "4": " four ",
        "5": " five ",
        "6": " six ",
        "7": " seven ",
        "8": " eight ",
        "9": " nine ",
    }
    text = "".join(digit_words.get(char, char) for char in text)

    return re.sub(r"\s+", " ", text).strip()



@st.cache_data(show_spinner=False, max_entries=20)
def synthesize(text: str) -> bytes:
    """
    Use the existing synthesis script and return generated WAV bytes.
    Repeated identical inputs can use cached audio.
    """
    if not SYNTHESIS_SCRIPT.is_file():
        raise SynthesisError(
            f"Synthesis script not found: {SYNTHESIS_SCRIPT}"
        )

    output_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp_file:
            output_path = Path(temp_file.name)

        try:
            result = subprocess.run(
                [
    sys.executable,
    str(SYNTHESIS_SCRIPT),
    text,
    "-o",
    str(output_path),
    "--max-steps",
    "1500",
],
                cwd=str(BASE_DIR),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                env={
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                },
                timeout=TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SynthesisError(
                f"Generation timed out after {TIMEOUT_S // 60} minutes."
            ) from exc

        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip()
            raise SynthesisError(
                error or "Speech generation failed without an error message."
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise SynthesisError("No audio file was generated.")

        return output_path.read_bytes()

    finally:
        if output_path is not None:
            output_path.unlink(missing_ok=True)


def wav_seconds(data: bytes):
    """Return WAV duration in seconds when readable."""
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            if audio.getframerate() == 0:
                return None

            return audio.getnframes() / audio.getframerate()

    except (wave.Error, EOFError, OSError):
        return None

###sidebar

script_ok = SYNTHESIS_SCRIPT.is_file()

with st.sidebar:
    st.markdown(
        f"""
<div class="sidebar-brand">
<div class="logo">🎙️</div>
<h2>Text to Speech Converter</h2>
<p>A voice built from scratch. Tacotron2 turns your text into a spectrogram, and HiFi-GAN turns that into sound.</p>
<div class="chips"><span>Tacotron2</span><span>HiFi-GAN</span><span>LJSpeech</span></div>
</div>
<div class="sidebar-note">Speech environment:<br>{html.escape(sys.executable)}<br><br>Identical text can use cached audio. New text starts a new synthesis process.</div>
""",
        unsafe_allow_html=True,
    )


status_text = (
    "Voice script found"
    if script_ok
    else "Voice script missing"
)

st.markdown(
    f"""
<div class="masthead">
<div class="status {'live' if script_ok else ''}"><span class="dot"></span>{status_text}</div>
<h1>Text to Speech Generator</h1>
<p>Type your text, press Generate speech, and hear it read in your own Tacotron2 voice.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not script_ok:
    st.error(
        f"Synthesis script not found: {SYNTHESIS_SCRIPT}"
    )

###tts

text = st.text_area(
    "What should it say?",
    value=SAMPLE_TEXT,
    height=180,
    max_chars=MAX_CHARS,
    placeholder="Type something here...",
    key="tts_text",
)

st.caption(f"{len(text)} / {MAX_CHARS} characters")

if st.button(
    "Generate speech",
    type="primary",
    use_container_width=True,
    disabled=not script_ok,
):
    cleaned_text = prepare_text(text)

    if not cleaned_text:
        st.warning("Please enter some text first.")

    else:
        try:
            with st.spinner(
                "Generating speech with Tacotron2 + HiFi-GAN..."
            ):
                audio_bytes = synthesize(cleaned_text)

            st.session_state["last_audio"] = audio_bytes

            st.success("Speech generated successfully!")

        except SynthesisError as exc:
            st.error("Speech generation failed.")
            st.code(str(exc))

        except Exception as exc:
            st.error(f"An unexpected error occurred: {exc}")

audio_bytes = st.session_state.get("last_audio")

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    seconds = wav_seconds(audio_bytes)

    if seconds is not None:
        st.caption(f"{seconds:.1f} seconds of audio")

    st.download_button(
        "Download WAV",
        data=audio_bytes,
        file_name="generated_speech.wav",
        mime="audio/wav",
        use_container_width=True,
    )