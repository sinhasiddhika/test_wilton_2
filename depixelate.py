import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# Set page config
st.set_page_config(page_title="Depixelate Image App")

# App title
st.title("Convert Pixel Art Back to Image (Depixelate)")

# Image uploader
uploaded_file = st.file_uploader("Upload a pixelated image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Show original pixelated image
    st.image(img_array, caption="Pixelated Input", use_container_width=True)

    # Resize parameters
    orig_height, orig_width = img_array.shape[:2]

    st.subheader("Depixelation Settings")
    target_width = st.number_input(
        "Target Width",
        min_value=orig_width,
        max_value=4 * orig_width,
        value=2 * orig_width,
        step=8
    )

    target_height = st.number_input(
        "Target Height",
        min_value=orig_height,
        max_value=4 * orig_height,
        value=2 * orig_height,
        step=8
    )

    method = st.selectbox(
        "Upsampling Method",
        ["Bicubic", "Bilinear", "Lanczos", "Nearest"],
        index=0
    )

    interpolation_map = {
        "Bicubic": cv2.INTER_CUBIC,
        "Bilinear": cv2.INTER_LINEAR,
        "Lanczos": cv2.INTER_LANCZOS4,
        "Nearest": cv2.INTER_NEAREST
    }

    # Perform upsampling
    depixelated_img = cv2.resize(img_array, (target_width, target_height), interpolation=interpolation_map[method])

    # Show output
    st.image(depixelated_img, caption="Depixelated Output", use_container_width=False)

    # Download button
    result = Image.fromarray(depixelated_img)
    buf = BytesIO()
    result.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Depixelated Image",
        data=byte_im,
        file_name="depixelated_output.png",
        mime="image/png"
    )

else:
    st.info("⬆ Please upload a pixelated image to continue.")
