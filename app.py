# app.py
import streamlit as st
import tempfile
import os
from transformer import SpeakerIdentificationTransformer

st.set_page_config(page_title="Speaker Identification")

@st.cache_resource
def load_transformer():
    return SpeakerIdentificationTransformer(
        model_path="latest_cnn_model.pth",
        target_label="Rowdy Delaney"
    )

transformer = load_transformer()


st.title("Speaker Identification System")
st.write("")
uploaded_file = st.file_uploader(
    "Upload an audio file to check if the speaker is **Rowdy Delaney**.",
    type=["wav", "mp3", "flac"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    with st.spinner("Analyzing..."):
        result = transformer.predict(audio_path)

    os.remove(audio_path)

    st.write(result)

