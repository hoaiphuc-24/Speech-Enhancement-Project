import os
import tempfile
from pathlib import Path

import streamlit as st
import torch

from inference import enhance_file, load_model

DEFAULT_MODEL_PATH = "experiments/DCCRN/best_model_cl.pt"


@st.cache_resource
def get_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, args_dict = load_model(model_path, device)
    sampling_rate = args_dict.get("sampling_rate", 16000)
    return model, sampling_rate, device


def run_enhancement(uploaded_file, model, sampling_rate: int, device: torch.device):
    suffix = Path(uploaded_file.name).suffix or ".wav"

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, f"input{suffix}")
        output_path = os.path.join(tmp_dir, "enhanced.wav")

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        enhance_file(
            model=model,
            input_path=input_path,
            output_path=output_path,
            sampling_rate=sampling_rate,
            device=device,
        )

        if not os.path.isfile(output_path):
            raise RuntimeError("Enhancement failed: output file was not created.")

        with open(output_path, "rb") as f:
            return f.read()


def main():
    st.set_page_config(page_title="Speech Enhancement Demo", page_icon=":studio_microphone:", layout="centered")
    st.title("Speech Enhancement Demo (DCCRN)")
    st.write("Upload an audio file, and the system will return an enhanced version processed by the DCCRN model.")
    if "enhanced_audio" not in st.session_state:
        st.session_state.enhanced_audio = None
        st.session_state.enhanced_name = None

    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)

    if not os.path.isfile(model_path):
        st.error(f"Model not found: {model_path}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload audio file", type=["wav", "flac", "mp3", "m4a", "ogg"]
    )

    if uploaded_file is None:
        st.info("Please upload an audio file to start.")
        st.stop()

    if st.button("Enhance audio", type="primary"):
        with st.spinner("Processing..."):
            try:
                model, sampling_rate, device = get_model(model_path)
                enhanced_audio = run_enhancement(uploaded_file, model, sampling_rate, device)
            except Exception as e:
                st.exception(e)
                st.stop()

        st.session_state.enhanced_audio = enhanced_audio
        st.session_state.enhanced_name = f"enhanced_{Path(uploaded_file.name).stem}.wav"
        st.success("Processing complete!")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original audio")
        st.caption(f"Original audio: {uploaded_file.name}")
        st.audio(uploaded_file.getvalue(), format="audio/wav")
    with col2:
        st.subheader("Enhanced audio")
        if st.session_state.enhanced_audio is None:
            st.info("No enhanced audio yet. Click 'Enhance audio' to begin.")
        else:
            st.caption(f"Enhanced audio: {st.session_state.enhanced_name}")
            st.audio(st.session_state.enhanced_audio, format="audio/wav")
            st.download_button(
                label="Download enhanced file",
                data=st.session_state.enhanced_audio,
                file_name=st.session_state.enhanced_name,
                mime="audio/wav",
            )


if __name__ == "__main__":
    main()
