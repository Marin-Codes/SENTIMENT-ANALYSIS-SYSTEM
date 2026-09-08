"""
S.A.R.A. -- Sentiment Analysis & Response AI
Web Application & Cloud Deployment Portal
==========================================
Streamlit-based multi-modal dashboard providing:
- Real-time Text Sentiment Analysis (Multilingual BERT + TextBlob)
- Voice & Speech Sentiment Analysis (Faster-Whisper STT + BERT)
- Facial Expression & Emotion Recognition (FER + OpenCV)
- Multi-Modal Affective Fusion & NVIDIA NIM Cognitive Reasoning
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd

# Set environment defaults
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Page Configuration
st.set_page_config(
    page_title="S.A.R.A. -- Sentiment & Emotion AI",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Cyberpunk / Cosmic Glassmorphism CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary: #0A0B1A;
    --card-bg: rgba(22, 24, 48, 0.7);
    --accent-violet: #8F75FF;
    --accent-cyan: #00F0FF;
    --accent-emerald: #00F078;
    --accent-rose: #FF3B69;
    --border-glass: rgba(227, 225, 250, 0.12);
    --text-primary: #F4F3FC;
    --text-muted: #8B8FB5;
}

html, body, [class*="css"] {
    font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(108, 91, 158, 0.18), transparent 45%),
                radial-gradient(circle at 85% 85%, rgba(46, 38, 104, 0.25), transparent 50%),
                #0A0B1A;
    color: #F4F3FC;
}

/* Glassmorphic Cards */
.sara-card {
    background: linear-gradient(135deg, rgba(227, 225, 250, 0.08) 0%, rgba(227, 225, 250, 0.02) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.sara-card:hover {
    border-color: rgba(143, 117, 255, 0.35);
}

.sara-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(143, 117, 255, 0.15);
    color: #C2B5FF;
    border: 1px solid rgba(143, 117, 255, 0.3);
}

.sara-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-glass);
}

.sara-title {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(120deg, #FFFFFF 30%, #C2B5FF 70%, #00F0FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.sara-metric-val {
    font-size: 28px;
    font-weight: 700;
    color: #FFFFFF;
}

.sentiment-pill-pos {
    background: rgba(0, 240, 120, 0.15);
    color: #00F078;
    border: 1px solid rgba(0, 240, 120, 0.3);
    padding: 6px 14px;
    border-radius: 12px;
    font-weight: 700;
}

.sentiment-pill-neu {
    background: rgba(0, 240, 255, 0.15);
    color: #00F0FF;
    border: 1px solid rgba(0, 240, 255, 0.3);
    padding: 6px 14px;
    border-radius: 12px;
    font-weight: 700;
}

.sentiment-pill-neg {
    background: rgba(255, 59, 105, 0.15);
    color: #FF3B69;
    border: 1px solid rgba(255, 59, 105, 0.3);
    padding: 6px 14px;
    border-radius: 12px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Cached Model Loaders
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model(model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    """Load HuggingFace sequence classification model for sentiment analysis."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        return f"Error loading sentiment model: {e}"


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size="base", device="cpu", compute_type="int8"):
    """Load faster-whisper model for speech transcription."""
    try:
        from faster_whisper import WhisperModel
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        return f"Error loading Whisper model: {e}"


@st.cache_resource(show_spinner=False)
def load_face_detector():
    """Load Facial Expression Recognition (FER) detector."""
    try:
        try:
            from fer.fer import FER
        except ImportError:
            from fer import FER
        return FER(mtcnn=True)
    except Exception as e:
        return f"Error loading FER face detector: {e}"


# ------------------------------------------------------------------------------
# Helper Utilities
# ------------------------------------------------------------------------------
def check_auth_status():
    """Check local auth server token if present."""
    token_file = Path(__file__).parent / ".sara_token"
    if not token_file.exists():
        return False, None
    try:
        payload = json.loads(token_file.read_text(encoding="utf-8"))
        if time.time() > payload.get("expires", 0):
            token_file.unlink(missing_ok=True)
            return False, None
        return True, payload.get("email")
    except Exception:
        return False, None


def analyze_text_sentiment(text: str, sentiment_pipe):
    """Analyze sentiment with BERT and TextBlob."""
    if not text.strip():
        return None

    # TextBlob
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        polarity = round(blob.sentiment.polarity, 3)
        subjectivity = round(blob.sentiment.subjectivity, 3)
    except Exception:
        polarity, subjectivity = 0.0, 0.0

    # BERT Multilingual
    bert_result = None
    if sentiment_pipe and not isinstance(sentiment_pipe, str):
        try:
            raw = sentiment_pipe(text[:512])[0]
            stars = int(raw["label"].split()[0])
            score = float(raw["score"])

            if stars <= 2:
                label = "Negative"
            elif stars == 3:
                label = "Neutral"
            else:
                label = "Positive"

            bert_result = {
                "stars": stars,
                "label": label,
                "score": score,
                "confidence": f"{score * 100:.1f}%"
            }
        except Exception as e:
            bert_result = {"error": str(e)}

    return {
        "text": text,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "bert": bert_result
    }


def call_nvidia_nim_reasoning(text: str, sentiment: str, emotions: str = None) -> str:
    """Query NVIDIA NIM / OpenAI API for affective reasoning if configured."""
    try:
        import config
        model_name = "nemotron_3_nano_30b"
        api_key = config.get_api_key(model_name)
        if not api_key:
            return "NVIDIA NIM API Key not set. Add your key to .env (NVIDIA_API_KEY_NEMOTRON_3_NANO_30B_A3B) to enable intelligent AI reasoning."

        from openai import OpenAI
        client = OpenAI(base_url=config.NVIDIA_BASE_URL, api_key=api_key)

        prompt = (
            f"User input: \"{text}\"\n"
            f"Detected Sentiment: {sentiment}\n"
            f"Detected Facial Emotions: {emotions or 'Not provided'}\n\n"
            "As S.A.R.A. (Sentiment Analysis & Response AI), provide an empathetic, concise, "
            "and psychologically insightful 2-3 sentence response acknowledging their emotional state."
        )

        response = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-30b-a3b",
            messages=[
                {"role": "system", "content": "You are S.A.R.A., an advanced empathetic multimodal emotional AI companion."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Reasoning response unavailable: {e}"


# ------------------------------------------------------------------------------
# Sidebar Navigation & Settings
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌌 S.A.R.A. PORTAL")
    st.markdown("<div class='sara-badge'>SYSTEM v2.5 • ONLINE</div>", unsafe_allow_html=True)
    st.write("")

    is_auth, email = check_auth_status()
    if is_auth:
        st.success(f"Session Active: `{email}`")
    else:
        st.info("Desktop Auth: Run `python auth_server.py` on port 5001 for local desktop sessions.")

    st.markdown("---")
    st.markdown("#### ⚙️ Engine Settings")
    whisper_size = st.selectbox("Whisper Model", ["tiny", "base", "small"], index=1)
    sentiment_threshold = st.slider("Confidence Threshold", 0.5, 0.99, 0.70, 0.05)

    st.markdown("---")
    st.markdown("#### 🚀 Deployment Modes")
    st.markdown("""
    - **Cloud Mode**: Streamlit Web UI
    - **Desktop Live Mode**: `python main.py` (Local Camera & Mic HUD)
    - **Auth Server**: `python auth_server.py` (Port 5001)
    - **Docker**: `docker compose up --build`
    """)

    st.markdown("---")
    st.caption("© 2026 S.A.R.A. • Sentiment Analysis & Response AI")


# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------
st.markdown("""
<div class="sara-header">
    <div>
        <h1 class="sara-title">S.A.R.A. Multimodal AI</h1>
        <p style="color: #8B8FB5; margin: 4px 0 0 0; font-size: 15px;">
            Next-Generation Sentiment Analysis, Vocal Acoustic Transcriptions, and Facial Emotion Analytics.
        </p>
    </div>
    <div style="text-align: right;">
        <span class="sara-badge">⚡ BERT + WHISPER + FER</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Main Tabs
# ------------------------------------------------------------------------------
tab_text, tab_audio, tab_face, tab_multimodal, tab_system = st.tabs([
    "📝 Text Sentiment",
    "🎙️ Voice & Audio",
    "👁️ Facial Emotion",
    "🧠 Multimodal Fusion",
    "📋 System Overview"
])

# ------------------------------------------------------------------------------
# TAB 1: TEXT SENTIMENT
# ------------------------------------------------------------------------------
with tab_text:
    st.markdown("<div class='sara-card'>", unsafe_allow_html=True)
    st.subheader("Multilingual Text Sentiment Classifier")
    st.write("Analyze emotional polarity, subjectivity, and star rating across multiple languages using fine-tuned BERT.")

    sample_options = [
        "Select a sample text...",
        "I absolutely love how responsive and intuitive this sentiment system is! Remarkable experience.",
        "The delivery was delayed by three days and the product arrived damaged. Very disappointed.",
        "The conference was held in Frankfurt yesterday with over 500 international attendees.",
        "Es ist ein wunderschöner Tag und ich freue mich riesig auf das Treffen!",
        "Ce film était incroyablement ennuyeux, une perte de temps totale."
    ]
    selected_sample = st.selectbox("Load Sample Expression", sample_options)

    default_val = selected_sample if selected_sample != sample_options[0] else ""
    user_text = st.text_area("Input Expression / Speech Transcript", value=default_val, height=120,
                             placeholder="Enter any text or conversation snippet...")

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_btn or (user_text and user_text != ""):
        with st.spinner("Classifying sentiment vectors..."):
            pipe = load_sentiment_model()
            results = analyze_text_sentiment(user_text, pipe)

        if results and results.get("bert") and "error" not in results["bert"]:
            bert = results["bert"]
            label = bert["label"]
            score = bert["score"]
            stars = bert["stars"]

            st.write("")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)

            with mcol1:
                pill_cls = "sentiment-pill-pos" if label == "Positive" else ("sentiment-pill-neu" if label == "Neutral" else "sentiment-pill-neg")
                st.markdown(f"**Classification**<br><span class='{pill_cls}'>{label.upper()}</span>", unsafe_allow_html=True)

            with mcol2:
                st.metric("Confidence", f"{score * 100:.1f}%")

            with mcol3:
                st.metric("Subjectivity", f"{results['subjectivity'] * 100:.1f}%")

            with mcol4:
                st.metric("Polarity Index", f"{results['polarity']:+.2f}")

            # Star representation & Progress
            st.write("")
            st.write(f"**Fine-Grained Rating**: {'★' * stars}{'☆' * (5 - stars)} ({stars}/5 Stars)")
            st.progress(score)

            # NVIDIA NIM / S.A.R.A. Cognitive Reasoning
            st.markdown("---")
            st.markdown("#### 🤖 S.A.R.A. Empathetic Cognitive Reasoning")
            ai_insight = call_nvidia_nim_reasoning(user_text, f"{label} ({stars}/5 stars)")
            st.info(ai_insight)

        elif results and results.get("bert") and "error" in results["bert"]:
            st.error(f"Error executing BERT pipeline: {results['bert']['error']}")
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 2: VOICE & AUDIO SENTIMENT
# ------------------------------------------------------------------------------
with tab_audio:
    st.markdown("<div class='sara-card'>", unsafe_allow_html=True)
    st.subheader("Speech-to-Text & Acoustic Sentiment Analysis")
    st.write("Upload an audio recording or speak to transcribe voice with Whisper and classify vocal sentiment.")

    audio_file = st.file_uploader("Upload Audio Sample (.wav, .mp3, .ogg, .m4a)", type=["wav", "mp3", "ogg", "m4a"])

    if audio_file is not None:
        st.audio(audio_file)

        if st.button("Transcribe & Analyze Voice", type="primary"):
            with st.spinner("Transcribing speech with Faster-Whisper..."):
                whisper = load_whisper_model(model_size=whisper_size)
                if isinstance(whisper, str):
                    st.error(whisper)
                else:
                    # Save to temp file for Whisper processing
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp:
                        tmp.write(audio_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        segments, info = whisper.transcribe(tmp_path, beam_size=5)
                        transcription = "".join([seg.text for seg in segments]).strip()
                        os.remove(tmp_path)

                        if transcription:
                            st.success(f"**Detected Language**: `{info.language.upper()}` (Probability: {info.language_probability:.2%})")
                            st.markdown(f"**Transcript**: *\"{transcription}\"*")

                            # Sentiment on transcribed text
                            pipe = load_sentiment_model()
                            res = analyze_text_sentiment(transcription, pipe)

                            if res and res.get("bert") and "error" not in res["bert"]:
                                bert = res["bert"]
                                acol1, acol2, acol3 = st.columns(3)
                                with acol1:
                                    st.metric("Vocal Sentiment", bert["label"])
                                with acol2:
                                    st.metric("Confidence", bert["confidence"])
                                with acol3:
                                    st.metric("Subjectivity", f"{res['subjectivity'] * 100:.1f}%")
                        else:
                            st.warning("No speech detected in the audio file.")
                    except Exception as ex:
                        st.error(f"Failed processing audio: {ex}")
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 3: FACIAL EMOTION RECOGNITION
# ------------------------------------------------------------------------------
with tab_face:
    st.markdown("<div class='sara-card'>", unsafe_allow_html=True)
    st.subheader("Facial Expression & Emotion Analysis")
    st.write("Upload a photo or capture an image via webcam to detect micro-expressions using FER with MTCNN.")

    face_mode = st.radio("Input Source", ["Camera Snapshot", "Upload Image"], horizontal=True)

    img_data = None
    if face_mode == "Camera Snapshot":
        img_data = st.camera_input("Capture Face Snapshot")
    else:
        img_data = st.file_uploader("Upload Portrait or Image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

    if img_data is not None:
        try:
            import cv2
            file_bytes = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            fcol1, fcol2 = st.columns([1.2, 1])

            with fcol1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Frame", use_container_width=True)

            with fcol2:
                with st.spinner("Detecting facial landmarks and emotions..."):
                    detector = load_face_detector()
                    if isinstance(detector, str):
                        st.error(detector)
                    else:
                        emotions_data = detector.detect_emotions(frame)
                        if emotions_data:
                            main_face = max(emotions_data, key=lambda f: (f["box"][2] * f["box"][3]))
                            emotions = main_face["emotions"]
                            dominant = max(emotions, key=emotions.get)

                            st.markdown(f"### Dominant Emotion: **{dominant.capitalize()}**")
                            st.progress(emotions[dominant])

                            # Chart of emotions
                            df_emotions = pd.DataFrame({
                                "Emotion": [e.capitalize() for e in emotions.keys()],
                                "Probability": list(emotions.values())
                            }).sort_values(by="Probability", ascending=False)

                            st.bar_chart(df_emotions.set_index("Emotion"))
                        else:
                            st.info("No face detected in the frame. Please ensure your face is well-lit and facing the camera.")
        except Exception as e:
            st.error(f"Error running facial emotion analysis: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 4: MULTIMODAL FUSION
# ------------------------------------------------------------------------------
with tab_multimodal:
    st.markdown("<div class='sara-card'>", unsafe_allow_html=True)
    st.subheader("Multimodal Emotional Congruence Analysis")
    st.write("S.A.R.A. cross-references facial micro-expressions with vocal sentiment to identify authentic emotional states and emotional incongruence (e.g. masking negative feelings with a polite smile).")

    test_col1, test_col2 = st.columns(2)
    with test_col1:
        mock_face = st.selectbox("Observed Facial Emotion", ["Happy", "Sad", "Angry", "Fear", "Surprise", "Neutral", "Disgust"])
    with test_col2:
        mock_sent = st.selectbox("Observed Vocal / Text Sentiment", ["Positive", "Neutral", "Negative"])

    congruent = (
        (mock_face in ["Happy"] and mock_sent == "Positive") or
        (mock_face in ["Sad", "Angry", "Disgust", "Fear"] and mock_sent == "Negative") or
        (mock_face == "Neutral" and mock_sent == "Neutral")
    )

    st.write("")
    if congruent:
        st.success(f"✅ **Congruent Affect Detected**: Facial expression ({mock_face}) matches verbal sentiment ({mock_sent}). High authenticity index.")
    else:
        st.warning(f"⚠️ **Incongruent Emotional Affect**: Facial expression ({mock_face}) diverges from verbal sentiment ({mock_sent}). Potential emotional suppression, sarcasm, or situational tension.")

    st.markdown("---")
    st.markdown("#### Holistic Affective State")
    st.markdown(f"""
    - **Affect Profile**: `{mock_face} Facial State` + `{mock_sent} Acoustic Sentiment`
    - **Recommended AI Response Tone**: {'Encouraging & Enthusiastic' if mock_face == 'Happy' and mock_sent == 'Positive' else ('Supportive & Empathetic' if mock_sent == 'Negative' else 'Attentive & Inquisitive')}
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 5: SYSTEM OVERVIEW
# ------------------------------------------------------------------------------
with tab_system:
    st.markdown("<div class='sara-card'>", unsafe_allow_html=True)
    st.subheader("System Architecture & Components")

    st.markdown("""
    | Component | Technology | Role |
    | :--- | :--- | :--- |
    | **Web Cloud Portal** | Streamlit, HTML5/CSS3 | Browser-accessible multimodal interface for cloud environments |
    | **Desktop Live HUD** | OpenCV, SoundDevice, Keyboard | Low-latency local desktop camera & microphone spacebar analyzer |
    | **Authentication Server** | Python `http.server`, SHA-256 | Local glassmorphism security portal issuing session tokens on port 5001 |
    | **Speech-to-Text** | `faster-whisper` (CTranslate2) | High-speed multilingual speech transcription |
    | **Sentiment Classifier** | Hugging Face Transformers (BERT) | `nlptown/bert-base-multilingual-uncased-sentiment` 5-star scoring |
    | **Emotion Recognition** | FER (Facial Expression Recognition) | MTCNN face detection with 7 universal emotion classifications |
    | **Cognitive Reasoning** | NVIDIA NIM API | Nemotron-3 MoE LLM for empathetic conversational reasoning |
    """)

    st.markdown("---")
    st.subheader("Quick Launch Commands")
    st.code("""
# 1. Launch Web Application (Recommended for Cloud & Local Browsers)
streamlit run app.py

# 2. Launch Local Auth Server
python auth_server.py

# 3. Launch Desktop Live HUD (Requires local webcam & mic)
python main.py

# 4. Launch Containerized Deployment
docker compose up --build
    """, language="bash")
    st.markdown("</div>", unsafe_allow_html=True)
