

from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from processing import process_image

# Must be the FIRST Streamlit command
st.set_page_config(page_title="Scanify", page_icon="📄", layout="wide")

CSS = """
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }
    .header { text-align: center; padding: 2rem 0;
              border-bottom: 2px solid #e0e0e0; margin-bottom: 2rem; }
    .title { color: #2c3e50; font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; }
    .subheader { color: #7f8c8d; font-size: 1.1rem; }
    .stDownloadButton button { background: #3498db !important; color: white !important;
                               border-radius: 25px; padding: 0.5rem 2rem; transition: all 0.3s; }
    .stDownloadButton button:hover { background: #2980b9 !important; transform: translateY(-1px); }
</style>
"""

MODES = {
    "Auto (recommended)": "auto",
    "Text page": "text",
    "Photo (face / colour picture)": "photo",
}


OUTPUTS = {
    "Black & white (like a printer)": "bw",
    "Colour": "color",
}


def to_pil(scanned):
    """Turns the result of process_image into a PIL image.
    Text page -> grey (2D). Photo -> colour (3D, BGR order)."""
    if scanned.ndim == 3:
        scanned = cv2.cvtColor(scanned, cv2.COLOR_BGR2RGB)
    return Image.fromarray(scanned).convert("RGB")


def main():
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="header">
            <h1 class="title">📄 Scanify</h1>
            <p class="subheader">Professional Document Scanning Solution</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        mode_name = st.radio("Picture type", list(MODES.keys()), horizontal=True)
    with col2:
        out_name = st.radio("Output", list(OUTPUTS.keys()), horizontal=True)
    mode = MODES[mode_name]
    output = OUTPUTS[out_name]

    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload document images",
        key="multi_uploader",
    )

    if uploaded_files:
        scanned_images = []

        with st.spinner("🔍 Scanning documents..."):
            for uploaded_file in uploaded_files:
                try:
                    image = Image.open(uploaded_file)
                    image = ImageOps.exif_transpose(image).convert("RGB")  # fix phone rotation
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    scanned = process_image(img, mode=mode, output=output)
                    scanned_images.append(to_pil(scanned))
                except Exception as e:
                    st.error(f"⚠️ Error processing {uploaded_file.name}: {e}")

        if scanned_images:
            st.markdown("### 📄 Scanned Results")
            for i, scanned_pil in enumerate(scanned_images):
                st.image(scanned_pil, caption=f"Page {i + 1}")

            st.markdown("---")
            st.markdown("### 📥 Download Your Scanned PDF")
            pdf_bytes = BytesIO()
            scanned_images[0].save(
                pdf_bytes,
                format="PDF",
                save_all=True,
                append_images=scanned_images[1:],
            )
            pdf_bytes.seek(0)

            st.download_button(
                label="Download Scanned PDF",
                data=pdf_bytes,
                file_name="scanned_document.pdf",
                mime="application/pdf",
            )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #7f8c8d; margin-top: 3rem;">
            <p>Scanify • v1.0 • Professional Document Scanning Solution</p>
            <p>Powered by OpenCV & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()