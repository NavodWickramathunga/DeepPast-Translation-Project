import streamlit as st


def set_page():
    st.set_page_config(
        page_title="Deep Past Akkadian Translator",
        page_icon="📜",
        layout="wide",
    )


def header():
    st.markdown(
        """
        <div style="padding:14px 18px; border-radius:16px; background:linear-gradient(90deg, rgba(240,240,255,1) 0%, rgba(245,255,245,1) 100%);">
          <h2 style="margin:0;">📜 Deep Past Akkadian → English Translator</h2>
          <p style="margin:6px 0 0 0; color:#333;">
            Local demo web app using a trained T5 model. Works offline for presentation day.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls():
    st.sidebar.header("⚙️ Controls")

    model_dir = st.sidebar.text_input(
        "Model folder",
        value="models/t5_baseline",
        help="Path to your trained model folder (config.json, tokenizer.json, model.safetensors...).",
    )

    st.sidebar.subheader("Decoding")
    num_beams = st.sidebar.slider("Beam search (num_beams)", 1, 8, 4)
    max_source_len = st.sidebar.slider("Max source length", 64, 512, 160, step=16)
    max_new_tokens = st.sidebar.slider("Max new tokens", 16, 512, 128, step=16)

    st.sidebar.subheader("Batch")
    batch_size = st.sidebar.selectbox("Batch size", [4, 8, 16, 32], index=2)

    return model_dir, num_beams, max_source_len, max_new_tokens, batch_size
