import streamlit as st
import os
import tempfile
import torch
import nest_asyncio
from prediction import vids, faceforensics, timit, dfdc, celeb
from model.config import load_config
from model.genconvit_vae import GenConViTVAE

nest_asyncio.apply()

def main():
    st.title("Deepfake Video Detection Using GenConViT")
    st.write("Upload a video or select a dataset to detect deepfakes.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default model weights
    default_weights = {
        "vae": "genconvit_vae_Mar_14_2025_15_54_27.pth",
        "ed": "genconvit_ed_inference.pth",
        "genconvit": "genconvit_inference.pkl"
    }

    # Sidebar options
    st.sidebar.header("Settings")
    dataset = st.sidebar.selectbox("Select Dataset", ["other", "dfdc", "faceforensics", "timit", "celeb"])
    num_frames = st.sidebar.number_input("Number of Frames", min_value=1, max_value=30, value=15)
    fp16 = st.sidebar.checkbox("Enable Half Precision (FP16)")
    model_variant = st.sidebar.selectbox("Model Variant", ["vae", "ed", "genconvit"])

    # Get model weight file path
    selected_weight = default_weights[model_variant]
    model_weights = st.sidebar.text_input("Model Weights", selected_weight)
    final_path = model_weights
    # Ensure correct path without double "weight/"
    # final_path = model_weights if os.path.exists(model_weights) else os.path.join("weight", model_weights)
    
    # Debugging: Show path
    # st.write(f"Checking for model weights at: {final_path}")

    # Load Model
    st.write("Loading model...")
    weights = torch.load('weight/genconvit_vae_Mar_14_2025_15_54_27.pth', map_location=device)
    state_dict = weights.get('state_dict', weights)

    config = load_config()
    model = GenConViTVAE(config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    st.success("✅ Model Loaded Successfully")

    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "jpg", "png", "jpeg"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name

        st.video(tmp_file_path)

        # Run prediction
        if st.button("Detect Deepfake"):
            with st.spinner("Processing..."):
                try:
                    result = vids(
                        None if model_variant == "vae" else final_path,
                        final_path if model_variant == "vae" else None,
                        tmp_file_path,
                        dataset,
                        num_frames,
                        model_variant,
                        fp16
                    )

                    st.write("Prediction Results:")
                    st.json(result)

                finally:
                    os.unlink(tmp_file_path)  # Clean up temporary file

    # Dataset selection
    if dataset != "other":
        if st.button(f"Run Detection on {dataset}"):
            with st.spinner("Processing..."):
                dataset_functions = {
                    "dfdc": dfdc,
                    "faceforensics": faceforensics,
                    "timit": timit,
                    "celeb": celeb
                }
                if dataset in dataset_functions:
                    result = dataset_functions[dataset](
                        None if model_variant == "vae" else final_path,
                        final_path if model_variant == "vae" else None,
                        dataset=dataset,
                        num_frames=num_frames,
                        net=model_variant,
                        fp16=fp16
                    )
                    st.write("Prediction Results:")
                    st.json(result)

if __name__ == "__main__":
    main()
