# 🌿 CropScan AI

A Streamlit app for crop disease detection using your trained MobileNetV2 model, with a bilingual chatbot (English + Twi).

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

Optional voice features (local only):
```bash
pip install -r requirements-voice.txt
```

### 2. Set your OpenAI API key (for AgriBot chat)
```bash
set OPENAI_API_KEY="your-key-here"
```

Or create a local `.env` file in the project root:

```bash
OPENAI_API_KEY=your-key-here
```

### 3. Save your model and class names from your notebook

After training, add these two cells to your notebook:

```python
# Save the model
model.save("models/crop_model.keras")

# Save class names
import json
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)
```

Then place `models/` in the same directory as `app.py`.

---

## Run

```bash
streamlit run app.py
```

---

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub (already done for this repo).
2. Go to https://share.streamlit.io and click **Create app**.
3. Select repository: `Enapa-Asamoah/CropLeafDisease_Checker`.
4. Set branch to `main`.
5. Set main file path to `app.py`.
6. In **Advanced settings -> Secrets**, add:

```toml
OPENAI_API_KEY = "your_real_key_here"
```

7. Click **Deploy**.

Notes for cloud deployment:
- `runtime.txt` is pinned to Python 3.11 for compatibility.
- Voice dependencies are excluded from cloud install to keep build reliable.
- The app still runs without voice; it will automatically show voice unavailable if optional packages are missing.
- Ensure `models/crop_model.keras` and `models/class_names.json` remain in the repo.

---

## Usage

1. **Model loads automatically** from `models/crop_model.keras` and `models/class_names.json`.
2. **Select crop** — Pick from the dropdown (auto-populated from your class names) or type manually.
3. **Upload leaf image** — JPG, PNG, or WebP.
4. **Click Analyse Leaf** — See the predicted disease, confidence score, and full class breakdown.
5. **Chat with AgriBot** — Ask follow-up questions in English or Twi. AgriBot auto-detects the language.
6. **Optional voice mode** — Enable voice in the sidebar, use microphone input, and listen to spoken replies in English or Twi.

---

## File structure

```
crop_disease_app/
├── app.py              # Main Streamlit UI
├── model_utils.py      # Model loading + prediction logic
├── chat_utils.py       # AgriBot chat with OpenAI API
├── speech_utils.py     # Local STT/TTS helpers (English + Twi)
├── requirements.txt
├── README.md
└── models/             # (you create this)
    ├── crop_model.keras
    └── class_names.json
```

---

## Notes

- The model uses **MobileNetV2 preprocessing** (pixel values scaled to [-1, 1]).
- Predictions are **crop-filtered**: only class names matching the selected crop are considered, then confidence is renormalized.
- Confidence below **60%** triggers a "review" warning.
- The full class confidence breakdown is available via the **dropdown expander** under the result card.
- AgriBot receives the scan result as context and can explain diseases, suggest treatments, and answer general crop questions — in English or Twi.
- Voice mode uses local open-source models: Whisper (STT) and MMS-TTS (English/Twi-Akan). First run downloads model weights.
