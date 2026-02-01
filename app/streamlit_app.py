# =========================================================
# 📜 Old Assyrian → English Translator (Streamlit App)
# CIS6005 – Computational Intelligence
# =========================================================

import re
import time
from pathlib import Path

import torch
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, T5ForConditionalGeneration

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Akkadian → English Translator",
    page_icon="📜",
    layout="wide",
)

# =========================================================
# CUSTOM UI STYLES
# =========================================================
st.markdown("""
<style>
.stApp { background-color: #fafafa; }

h2 { font-weight: 700; }

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 16px;
    margin-top: -10px;
}

.card {
    background-color: #ffffff;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 22px;
}

.stButton>button {
    background-color: #4f46e5;
    color: white;
    border-radius: 12px;
    height: 46px;
    font-size: 16px;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #4338ca;
}

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <h2 style="text-align:center;">📜 Akkadian → English Translator</h2>
    <p class="subtitle">
    Translating 4,000-year-old Old Assyrian texts using Deep Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model" / "t5_baseline"

# =========================================================
# LOAD MODEL (CACHED)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=True,
        local_files_only=True
    )
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

# =========================================================
# TEXT CLEANING (ASSIGNMENT-COMPLIANT)
# =========================================================
def clean_source(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("!", "").replace("?", "")
    text = text.replace("/", " ")
    text = text.replace(":", " ").replace(".", " ")
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)
    text = text.replace("˹", "").replace("˺", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_prefix(text: str) -> str:
    return "translate Akkadian to English: " + text

# =========================================================
# LOAD MODEL
# =========================================================
tokenizer, model, device = load_model()

# =========================================================
# SAMPLE TEXT
# =========================================================
EXAMPLE_TEXT = (
    "um-ma kà-ru-um kà-ni-ia-ma a-na aa-qí-il "
    "i-na mup-pì-im aa a-lim(ki)"
)

# =========================================================
# MAIN INPUT CARD
# =========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔤 Enter Akkadian Transliteration")

user_text = st.text_area(
    "Paste Akkadian text below",
    height=160,
    placeholder="Example: um-ma kà-ru-um kà-ni-ia-ma a-na..."
)

col1, col2 = st.columns(2)
with col1:
    translate_clicked = st.button("Translate", use_container_width=True)
with col2:
    if st.button("Use Sample Text", use_container_width=True):
        st.session_state["sample"] = EXAMPLE_TEXT
        st.experimental_rerun()

if "sample" in st.session_state:
    user_text = st.session_state.pop("sample")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TRANSLATION OUTPUT
# =========================================================
if translate_clicked and user_text.strip():
    with st.spinner("Translating ancient text…"):
        start = time.time()

        cleaned = clean_source(user_text)
        prefixed = add_prefix(cleaned)

        inputs = tokenizer(
            prefixed,
            return_tensors="pt",
            truncation=True,
            max_length=160
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True
            )

        prediction = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        elapsed = time.time() - start

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.success(f"Translation completed in {elapsed:.2f} seconds")

    st.subheader("🇬🇧 English Translation")
    st.text_area(
        "Result",
        value=prediction,
        height=180
    )

    with st.expander("🔍 Preprocessed input (for explanation)"):
        st.code(cleaned)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# OPTIONAL BATCH FEATURE
# =========================================================
with st.expander("📄 Translate multiple texts (CSV – optional)"):
    st.caption("Upload a CSV file with a column named `transliteration`")
    file = st.file_uploader("Choose CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        if "transliteration" not in df.columns:
            st.error("CSV must contain a 'transliteration' column.")
        else:
            with st.spinner("Translating batch…"):
                texts = (
                    df["transliteration"]
                    .astype(str)
                    .apply(clean_source)
                    .apply(add_prefix)
                    .tolist()
                )

                preds = []
                for t in texts:
                    enc = tokenizer(
                        t,
                        return_tensors="pt",
                        truncation=True,
                        max_length=160
                    ).to(device)

                    with torch.no_grad():
                        out = model.generate(
                            **enc,
                            max_new_tokens=128,
                            num_beams=4
                        )
                    preds.append(
                        tokenizer.decode(out[0], skip_special_tokens=True)
                    )

                df["translation"] = preds
                st.dataframe(df.head(10), use_container_width=True)

                st.download_button(
                    "⬇ Download Translations",
                    df.to_csv(index=False),
                    "translations.csv",
                    "text/csv"
                )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
    CIS6005 – Computational Intelligence · Deep Past Kaggle Challenge · Streamlit Demo
    </div>
    """,
    unsafe_allow_html=True
)
