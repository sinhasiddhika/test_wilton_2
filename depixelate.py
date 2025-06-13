import streamlit as st
from PIL import Image
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import AutoProcessor
from torchvision import transforms
import numpy as np
import tempfile

st.set_page_config(page_title="Pixel Art to Realistic Image", layout="centered")
st.title(" Convert Pixel Art to Realistic Image")

st.sidebar.header("Prompt & Settings")
prompt = st.sidebar.text_input("Prompt", value="a cute realistic fox, highly detailed, bright orange fur")
num_inference_steps = st.sidebar.slider("Inference Steps", 20, 100, 50)
guidance_scale = st.sidebar.slider("Guidance Scale", 3.0, 15.0, 9.0, step=0.5)

uploaded_file = st.file_uploader("Upload a pixel art image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    pixel_art_img = Image.open(uploaded_file).convert("RGB")
    st.image(pixel_art_img, caption=" Pixel Art Input", use_container_width=True)
    st.subheader("Generating Realistic Image...")

    import cv2
    img = pixel_art_img.resize((512, 512))
    np_img = np.array(img)
    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(np_img, low_threshold, high_threshold)
    edges = Image.fromarray(edges)
    st.image(edges, caption=" Edge Detection Map", use_container_width=True)
    with st.spinner("Loading AI model (Stable Diffusion + ControlNet)..."):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            st.error("CUDA/GPU not available. Please run this app in a GPU environment like Google Colab.")

    with st.spinner("Generating image with ControlNet..."):
        output = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image=edges
        )

        result_img = output.images[0]
        st.image(result_img, caption=" Realistic Output", use_container_width=True)

        buf = BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label=" Download Realistic Image",
            data=byte_im,
            file_name="realistic_output.png",
            mime="image/png"
        )

else:
    st.info(" Upload a pixel image to start.")
