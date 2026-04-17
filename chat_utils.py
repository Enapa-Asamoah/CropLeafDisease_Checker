import os

OpenAI = None
_import_error = None

try:
    from openai import OpenAI as _OpenAI
    OpenAI = _OpenAI
except Exception as exc:
    _import_error = exc

_client = None


def set_openai_api_key(api_key: str | None):
    """Set or clear the OpenAI API key for subsequent chat requests."""
    global _client

    cleaned_key = (api_key or "").strip()
    if cleaned_key:
        os.environ["OPENAI_API_KEY"] = cleaned_key
        if OpenAI is None:
            _client = None
            return
        _client = OpenAI(api_key=cleaned_key)
    else:
        os.environ.pop("OPENAI_API_KEY", None)
        _client = None


def _get_client():
    global _client

    if OpenAI is None:
        detail = f" ({_import_error})" if _import_error else ""
        raise RuntimeError(
            "OpenAI package is not installed. Install with: pip install openai" + detail
        )

    if _client is None:
        _client = OpenAI()
    return _client

SYSTEM_PROMPT = """You are AgriBot, a friendly and knowledgeable agricultural assistant specializing in crop diseases.
You help farmers and agronomists understand crop health, diseases, treatments, and prevention strategies.

Language rules:
- Detect the user's language automatically from their message.
- If the user writes in Twi (Akan/Ghanaian language), respond ENTIRELY in Twi.
- If the user writes in English, respond in English.
- If mixed, match the dominant language.
- Keep responses concise, warm, and practical.

When given a prediction context (crop name, disease, confidence), use it to give targeted advice.
Always be encouraging and constructive — farmers need actionable help, not just diagnosis.

Common Twi phrases you may encounter:
- "Mepa wo kyɛw" = Please
- "Meda wo ase" = Thank you  
- "Ɛyɛ den" = It is serious/difficult
- "Sɛdeɛ" = How/What
- "Aba" = Hello/Welcome
"""


def build_context_message(prediction: dict | None) -> str:
    """Build a context string from prediction result to seed the chat."""
    if not prediction:
        return ""
    crop = prediction.get("crop", "").replace("_", " ").title()
    disease = prediction.get("predicted_disease", "").replace("_", " ").title()
    confidence = prediction.get("confidence", 0)
    status = prediction.get("status", "")

    return (
        f"[Current scan result — Crop: {crop}, "
        f"Predicted condition: {disease}, "
        f"Confidence: {confidence:.1%}, "
        f"Status: {status}]"
    )


def chat(messages: list[dict], prediction: dict | None = None) -> str:
    """
    Send chat messages to OpenAI and return response text.
    messages: list of {"role": "user"/"assistant", "content": str}
    prediction: optional dict from model_utils.predict_disease_for_crop
    """
    system = SYSTEM_PROMPT
    if prediction:
        system += f"\n\nCurrent prediction context:\n{build_context_message(prediction)}"

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o",
            temperature=0.4,
            max_tokens=600,
            messages=[
                {"role": "system", "content": system},
                *messages,
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        error_str = str(e)
        if "insufficient_quota" in error_str or "429" in error_str:
            return (
                "⚠️ **OpenAI Quota Exceeded**: Your OpenAI account has hit a quota limit.\n\n"
                "**To fix this:**\n"
                "1. Visit [OpenAI Billing](https://platform.openai.com/account/billing/overview)\n"
                "2. Check your usage and billing status\n"
                "3. Add a valid payment method if needed\n"
                "4. Increase your quota limits if capped\n\n"
                "Then restart the app. For now, I can help with general crop disease questions, "
                "but chat responses require a valid OpenAI account."
            )
        elif "OPENAI_API_KEY" in error_str or "api_key" in error_str.lower():
            return (
                "⚠️ **Missing or Invalid OpenAI API Key**: No valid API key found.\n\n"
                "**To fix:**\n"
                "1. Create a `.env` file in the app folder (copy from `.env.example`)\n"
                "2. Add your OpenAI API key: `OPENAI_API_KEY=sk-...`\n"
                "3. Restart the app\n\n"
                "Get a key from: https://platform.openai.com/account/api-keys"
            )
        else:
            return f"❌ **Chat Error**: {error_str}\n\nPlease check your OpenAI account and API key."


def get_welcome_message() -> str:
    return (
        "👋 Hello! I'm **AgriBot** — your crop health assistant.\n\n"
        "Upload a leaf image and select your crop, then ask me anything about the result. "
        "I speak **English** and **Twi** — just write in whichever feels natural!\n\n"
        "_Mede ɔhaw biara a efa aba yareɛ ho bɛboa wo._ 🌱"
    )
