from style_transfer_model import StyleTransfer, STYLE_IMAGES
import streamlit as st
from PIL import Image, ImageEnhance
from io import BytesIO
import os

# Luminance & Greyscale Adjustment Functions
def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def desaturate_image(image, factor):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


# Progress Callback Function
def update_progress(progress, loss):
    clamped_progress = min(max(progress, 0.0), 1.0)
    progress_bar.progress(clamped_progress)
    status_text.text(f"Progress: {int(clamped_progress*100)}% | Current losses: {loss:.2f}")

# Streamlit Interface
st.title("AI-Powered Chinese Gongbi Painting Style Transfer Tool —— VoidScape")
st.markdown("Upload your image, choose the art style and adjust the parameters")

# Choose a style
style_name = st.selectbox("Choose a style", list(STYLE_IMAGES.keys()))

# Display style image
col1, col2, col3 = st.columns(3)
with col2:
    style_path = os.path.join("styles", STYLE_IMAGES[style_name])
    st.image(style_path, 
            caption=f"Current Style: {style_name}", 
            width=200, 
            use_container_width=True)

# User controlled brightness and Greyness
brightness_factor = st.slider("Adjust Brightness", 0.5, 2.0, 1.0, 0.05)
saturation_factor = st.slider("Adjust saturation", 0.0, 1.0, 1.0, 0.05)

# Upload an image
uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Adjusting Brightness and Greyness
    adjusted_image = adjust_brightness(image, brightness_factor)
    grayish_image = desaturate_image(adjusted_image, saturation_factor)

    # Displaying the adjusted image
    st.image(grayish_image, caption="Adjusted image", use_container_width=True)

    # Convert to byte stream
    buffered = BytesIO()
    grayish_image.save(buffered, format="PNG")
    img_byte_array = buffered.getvalue()

    # Initialising the processor
    processor = StyleTransfer()

    # Creating progress components
    progress_bar = st.progress(0)
    status_text = st.empty()

    if st.button("Start conversion"):
        with st.spinner("Undergoing conversion..."):
            result = processor.transfer(
                img_byte_array, 
                style_name,
                progress_callback=update_progress
                )

            # Display the final image
            st.image(result, caption="Conversion results", use_container_width=True)

            # Download
            buf = BytesIO()
            result.save(buf, format="PNG")

            # Set the download format to JPG
            download_format = "JPG"

            buf = BytesIO()
            result.save(buf, format="JPEG")

            buf.seek(0)

            st.download_button(
                label="Download JPG file",
                data=buf.getvalue(),
                file_name="styled_image.jpg",
                mime="image/jpeg"
            )