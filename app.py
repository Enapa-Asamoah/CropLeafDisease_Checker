import streamlit as st
import json
import os
import hashlib
from pathlib import Path
import plotly.graph_objects as go

from model_utils import (
    load_model,
    predict_disease_for_crop,
    get_available_crops,
    format_label,
    CONFIDENCE_THRESHOLD,
)
from chat_utils import chat, get_welcome_message, set_openai_api_key
from speech_utils import (
    speech_dependencies_ready,
    synthesize_speech_bytes,
    transcribe_audio_bytes,
)


def load_env_file(env_path: Path, override_existing: bool = True) -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (override_existing or key not in os.environ):
                os.environ[key] = value


load_env_file(Path(__file__).resolve().parent / ".env", override_existing=True)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropScan AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Background */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

/* Cards */
.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
}

.healthy-badge {
    background: linear-gradient(135deg, #1a4731, #2d6a4f);
    border: 1px solid #40916c;
    color: #b7e4c7;
    padding: 8px 18px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
}

.disease-badge {
    background: linear-gradient(135deg, #4a1a1a, #7b2d2d);
    border: 1px solid #c9184a;
    color: #ffb3c1;
    padding: 8px 18px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
}

.review-badge {
    background: linear-gradient(135deg, #3d2e00, #7a5c00);
    border: 1px solid #f4a261;
    color: #ffd166;
    padding: 8px 18px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
}

.confidence-number {
    font-family: 'DM Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
}

.confidence-high { color: #56d364; }
.confidence-mid  { color: #f4a261; }
.confidence-low  { color: #f85149; }

.chat-bubble-user {
    background: #1c2d3a;
    border: 1px solid #1f6feb;
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px;
    margin: 6px 0 6px 40px;
    font-size: 0.92rem;
}

.chat-bubble-bot {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px 12px 12px 2px;
    padding: 12px 16px;
    margin: 6px 40px 6px 0;
    font-size: 0.92rem;
}

.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 8px;
}

/* Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div > div {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
}

/* Buttons */
.stButton > button {
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    padding: 10px 22px;
    transition: background 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: #2ea043;
}

/* Upload area */
[data-testid="stFileUploadDropzone"] {
    background: #0d1117 !important;
    border: 1px dashed #30363d !important;
    border-radius: 10px !important;
}

/* Divider */
hr { border-color: #30363d; }

/* Expander */
[data-testid="stExpander"] {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model_init_attempted" not in st.session_state:
    st.session_state.model_init_attempted = False
if "model_error" not in st.session_state:
    st.session_state.model_error = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": get_welcome_message()}
    ]
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = False
if "voice_auto_play" not in st.session_state:
    st.session_state.voice_auto_play = True
if "voice_output_lang" not in st.session_state:
    st.session_state.voice_output_lang = "auto"
if "voice_last_audio_hash" not in st.session_state:
    st.session_state.voice_last_audio_hash = ""
if "voice_last_reply_audio" not in st.session_state:
    st.session_state.voice_last_reply_audio = b""

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_CANDIDATES = [
    MODEL_DIR / "crop_model.onnx",
    MODEL_DIR / "crop_model.keras",
]
DEFAULT_CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"


def load_preloaded_model() -> None:
    if st.session_state.model_loaded or st.session_state.model_init_attempted:
        return

    st.session_state.model_init_attempted = True

    model_path = next((candidate for candidate in MODEL_CANDIDATES if candidate.exists()), None)
    if model_path is None:
        expected = ", ".join(str(p.name) for p in MODEL_CANDIDATES)
        st.session_state.model_error = f"Model file not found. Expected one of: {expected}"
        return
    if not DEFAULT_CLASS_NAMES_PATH.exists():
        st.session_state.model_error = f"Class names file not found: {DEFAULT_CLASS_NAMES_PATH}"
        return

    with st.spinner("Loading preloaded model..."):
        try:
            load_model(str(model_path), str(DEFAULT_CLASS_NAMES_PATH))
            st.session_state.model_loaded = True
            st.session_state.model_error = ""
            st.session_state.model_source_name = model_path.name
        except Exception as e:
            st.session_state.model_error = f"Failed to load preloaded model: {e}"


load_preloaded_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 CropScan AI")
    st.markdown("<p class='section-label'>Voice Assistant (Optional)</p>", unsafe_allow_html=True)
    st.session_state.voice_enabled = st.toggle(
        "Enable voice input/output",
        value=st.session_state.voice_enabled,
        help="Uses local open-source speech models for English/Twi.",
    )

    st.session_state.voice_auto_play = st.checkbox(
        "Auto-generate spoken reply",
        value=st.session_state.voice_auto_play,
        disabled=not st.session_state.voice_enabled,
    )

    voice_lang_label = st.selectbox(
        "Reply voice language",
        options=["Auto", "English", "Twi"],
        index={"auto": 0, "en": 1, "ak": 2}.get(st.session_state.voice_output_lang, 0),
        disabled=not st.session_state.voice_enabled,
    )
    st.session_state.voice_output_lang = {"Auto": "auto", "English": "en", "Twi": "ak"}[voice_lang_label]

    if st.session_state.voice_enabled:
        deps_ok, deps_error = speech_dependencies_ready()
        if deps_ok:
            st.caption("Voice dependencies are available.")
        else:
            st.warning("Voice features need extra packages. Run: pip install faster-whisper transformers torch soundfile")
            st.caption(f"Dependency detail: {deps_error}")

    st.divider()

    if st.session_state.model_loaded:
        st.markdown("**Model ready**", unsafe_allow_html=False)
    else:
        st.markdown("⚠️ **Model not loaded**")

    if st.session_state.model_error:
        st.error(st.session_state.model_error)

    source_name = st.session_state.get("model_source_name", MODEL_CANDIDATES[0].name)
    st.caption(f"Loaded from: {source_name} + {DEFAULT_CLASS_NAMES_PATH.name}")

    st.divider()
    st.markdown("<p class='section-label'>About</p>", unsafe_allow_html=True)
    st.markdown(
        "Upload a leaf image to detect diseases. "
        "Chat with AgriBot in **English** or **Twi** 🇬🇭"
    )

if st.session_state.openai_api_key:
    set_openai_api_key(st.session_state.openai_api_key)


def resolve_tts_lang(detected_lang: str = "") -> str:
    selected = st.session_state.voice_output_lang
    if selected in {"en", "ak"}:
        return selected

    detected = (detected_lang or "").lower().strip()
    if detected in {"ak", "tw", "twi"}:
        return "ak"
    return "en"


def run_chat_turn(user_text: str, detected_lang: str = "") -> None:
    st.session_state.chat_history.append({"role": "user", "content": user_text})
    with st.spinner("AgriBot is thinking..."):
        reply = chat(
            st.session_state.chat_history,
            st.session_state.prediction,
        )
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.session_state.voice_enabled and st.session_state.voice_auto_play:
        deps_ok, _ = speech_dependencies_ready()
        if deps_ok:
            try:
                tts_lang = resolve_tts_lang(detected_lang)
                st.session_state.voice_last_reply_audio = synthesize_speech_bytes(reply, lang_code=tts_lang)
            except Exception:
                st.session_state.voice_last_reply_audio = b""


# ── Main layout ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1], gap="large")

# ─── LEFT: Scan panel ────────────────────────────────────────────────────────
with left_col:
    st.markdown("### 🔬 Leaf Scanner")

    # Crop selection
    st.markdown("<p class='section-label'>Crop Name</p>", unsafe_allow_html=True)

    crops = get_available_crops()
    if crops:
        crop_input = st.selectbox(
            "Select crop",
            options=crops,
            format_func=format_label,
            label_visibility="collapsed"
        )
    else:
        crop_input = st.text_input(
            "Enter crop name",
            placeholder="e.g. tomato, potato, corn ...",
            label_visibility="collapsed"
        )

    # Image upload
    st.markdown("<p class='section-label' style='margin-top:16px'>Leaf Image</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.image(uploaded_file, use_column_width=True, caption="Uploaded leaf")

    # Predict button
    st.markdown("<div style='margin-top:12px'>", unsafe_allow_html=True)
    predict_btn = st.button("Analyse Leaf", disabled=not st.session_state.model_loaded)
    st.markdown("</div>", unsafe_allow_html=True)

    if not st.session_state.model_loaded:
        st.caption("⚠️ Load a model from the sidebar first.")

    if predict_btn and uploaded_file and crop_input:
        with st.spinner("Analysing..."):
            try:
                img_bytes = uploaded_file.read()
                result = predict_disease_for_crop(crop_input, img_bytes)
                st.session_state.prediction = result
                # Add prediction context to chat
                disease_label = format_label(result["predicted_disease"])
                crop_label = format_label(result["crop"])
                context_msg = (
                    f"I just scanned a **{crop_label}** leaf. "
                    f"The result is: **{disease_label}** "
                    f"({result['confidence']:.1%} confidence). "
                    f"Can you tell me more about this?"
                )
                st.session_state.chat_history.append({"role": "user", "content": context_msg})
                bot_reply = chat(st.session_state.chat_history, result)
                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif predict_btn and not uploaded_file:
        st.warning("Please upload a leaf image first.")

    # ── Result card ──────────────────────────────────────────────────────────
    pred = st.session_state.prediction
    if pred:
        st.divider()
        st.markdown("### 📊 Result")

        disease = pred["predicted_disease"]
        confidence = pred["confidence"]
        status = pred["status"]

        is_healthy = disease == "healthy"
        conf_class = (
            "confidence-high" if confidence >= 0.75
            else "confidence-mid" if confidence >= CONFIDENCE_THRESHOLD
            else "confidence-low"
        )
        badge_html = (
            f"<span class='healthy-badge'>Healthy</span>"
            if is_healthy
            else (
                f"<span class='disease-badge'>🦠 {format_label(disease)}</span>"
                if status == "confirmed"
                else f"<span class='review-badge'>⚠️ {format_label(disease)} — Low confidence</span>"
            )
        )

        st.markdown(f"""
        <div class='result-card'>
            <div class='section-label'>Detected Condition</div>
            {badge_html}
            <div style='margin-top:20px; display:flex; align-items:baseline; gap:8px'>
                <span class='confidence-number {conf_class}'>{confidence:.1%}</span>
                <span style='color:#8b949e; font-size:0.85rem'>confidence</span>
            </div>
            <div style='margin-top:4px; color:#8b949e; font-size:0.8rem'>
                Crop: <strong style='color:#e6edf3'>{format_label(pred['crop'])}</strong>
                &nbsp;·&nbsp;
                Class: <code style='font-family:DM Mono,monospace; font-size:0.78rem'>{pred['class_name']}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if status == "review":
            st.warning(
                "Confidence is below the 60% threshold. "
                "Consider retaking the photo with better lighting, or consult an expert.",
                icon="⚠️"
            )

        # All-class confidence breakdown
        with st.expander("📈 Full confidence breakdown"):
            all_scores = pred.get("all_scores", {})
            if all_scores:
                labels = [format_label(k.split("___")[-1] if "___" in k else k) for k in all_scores]
                values = list(all_scores.values())
                colors = [
                    "#56d364" if v == max(values) else "#1f6feb"
                    for v in values
                ]
                fig = go.Figure(go.Bar(
                    x=values,
                    y=labels,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:.1%}" for v in values],
                    textposition='outside',
                    textfont=dict(color='#8b949e', size=11),
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e6edf3', family='Sora'),
                    margin=dict(l=0, r=60, t=10, b=10),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='#30363d',
                        tickformat='.0%',
                        range=[0, max(values) * 1.25],
                    ),
                    yaxis=dict(autorange='reversed'),
                    height=max(220, len(labels) * 38),
                )
                st.plotly_chart(fig, use_container_width=True)


# ─── RIGHT: Chat panel ───────────────────────────────────────────────────────
with right_col:
    st.markdown("### 💬 AgriBot")
    st.caption("Ask about your crop, disease treatment, or prevention — in English or Twi 🇬🇭")

    # Chat history
    chat_container = st.container(height=480)
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "assistant":
                st.markdown(
                    f"<div class='chat-bubble-bot'>🌿 {msg['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='chat-bubble-user'>👤 {msg['content']}</div>",
                    unsafe_allow_html=True
                )

    # Chat input
    if st.session_state.voice_enabled:
        if not hasattr(st, "audio_input"):
            st.info("Voice input is not available in this Streamlit version. You can still type messages.")
        else:
            speech_input = st.audio_input("Speak to AgriBot (English or Twi)")
            if speech_input is not None:
                audio_bytes = speech_input.getvalue()
                audio_hash = hashlib.sha256(audio_bytes).hexdigest() if audio_bytes else ""
                if audio_bytes and audio_hash != st.session_state.voice_last_audio_hash:
                    st.session_state.voice_last_audio_hash = audio_hash
                    deps_ok, deps_error = speech_dependencies_ready()
                    if not deps_ok:
                        st.error(f"Voice transcription unavailable: {deps_error}")
                    else:
                        with st.spinner("Transcribing speech..."):
                            try:
                                transcript, detected_lang = transcribe_audio_bytes(audio_bytes)
                            except Exception as e:
                                st.error(f"Transcription failed: {e}")
                                transcript = ""
                                detected_lang = ""

                        if transcript:
                            st.caption(f"Voice transcript ({detected_lang or 'auto'}): {transcript}")
                            try:
                                run_chat_turn(transcript, detected_lang=detected_lang)
                                st.session_state.chat_input_key += 1
                                st.rerun()
                            except Exception as e:
                                st.error(f"Chat error: {e}")

    user_input = st.chat_input(
        "Ask AgriBot anything... (English or Twi)",
        key=f"chat_input_{st.session_state.chat_input_key}"
    )

    if user_input:
        try:
            run_chat_turn(user_input)
            st.session_state.chat_input_key += 1
            st.rerun()
        except Exception as e:
            st.error(f"Chat error: {e}")

    if st.session_state.voice_last_reply_audio:
        st.audio(st.session_state.voice_last_reply_audio, format="audio/wav")

    # Clear chat button
    if st.button("Clear chat", key="clear_chat"):
        st.session_state.chat_history = [
            {"role": "assistant", "content": get_welcome_message()}
        ]
        st.session_state.voice_last_reply_audio = b""
        st.session_state.voice_last_audio_hash = ""
        st.session_state.chat_input_key += 1
        st.rerun()
