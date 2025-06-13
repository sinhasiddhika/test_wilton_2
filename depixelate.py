import streamlit as st
from PIL import Image
import torch
from io import BytesIO
import numpy as np
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# ------------------- SETTINGS -------------------
st.set_page_config(page_title="Pixel Art to Realistic Image", layout="centered")
st.title("Convert Pixel Art to Realistic Image")

# ------------------- SIDEBAR CONFIG -------------------
st.sidebar.header("Prompt & Settings")
prompt = st.sidebar.text_input("Prompt", value="a cute cartoon fox, highly detailed, studio ghibli style")
num_inference_steps = st.sidebar.slider("Inference Steps", 20, 100, 50)
guidance_scale = st.sidebar.slider("Guidance Scale", 3.0, 20.0, 12.0, step=0.5)

# ------------------- IMAGE UPLOAD -------------------
uploaded_file = st.file_uploader("Upload a pixel art image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    pixel_art_img = Image.open(uploaded_file).convert("RGB")
    st.image(pixel_art_img, caption="Pixel Art Input", use_container_width=True)

    # Resize for processing
    image_resized = pixel_art_img.resize((512, 512))
    image_np = np.array(image_resized)

    # ------------------- CONVERT TO SKETCH (for scribble model) -------------------
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    sketch = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    sketch_img = Image.fromarray(sketch)
    st.image(sketch_img, caption="Scribble Sketch", use_container_width=True)

    # ------------------- LOAD SCRIBBLE CONTROLNET MODEL -------------------
    with st.spinner("Loading AI model (Stable Diffusion + ControlNet Scribble)..."):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble",
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
            st.warning("GPU not available. This may not work correctly without CUDA.")

    # ------------------- GENERATE IMAGE -------------------
    with st.spinner("Generating image from pixel sketch..."):
        result = pipe(
            prompt,
            image=sketch_img,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )

        result_img = result.images[0]
        st.image(result_img, caption="Realistic Output", use_container_width=True)

        # ------------------- DOWNLOAD -------------------
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Realistic Image",
            data=byte_im,
            file_name="realistic_output.png",
            mime="image/png"
        )

else:
    st.info("⬆ Upload a pixel image to start.")
