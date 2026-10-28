from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from processing import (
    apply_histogram_equalization,
    apply_noise_filter,
    apply_threshold,
    process_image,
    render_histogram_image,
    visualize_document_boundary,
)

# Must be the FIRST Streamlit command
st.set_page_config(page_title="Scanify - Document Scanner", page_icon="📄", layout="wide")

CSS = """
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }
    .header { text-align: center; padding: 1.5rem 0;
              border-bottom: 2px solid #e0e0e0; margin-bottom: 1.5rem; }
    .title { color: #2c3e50; font-size: 2.3rem; font-weight: 700; margin: 0.5rem 0; }
    .subheader { color: #7f8c8d; font-size: 1.05rem; }
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


def to_pil(img_array):
    """Turns a numpy image array into a PIL Image for display and PDF saving.
    Grayscale (2D) or BGR (3D) -> RGB PIL Image."""
    if img_array.ndim == 3:
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def main():
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="header">
            <h1 class="title">📄 Scanify</h1>
            <p class="subheader">Document Scanner & Image Enhancement Suite</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar: Image Tools ---
    st.sidebar.markdown("## 🛠️ Image Tools")

    TOOL_DESCRIPTIONS = {
        "Scan a Document": "Automatically detect document edges, straighten the page, and improve readability.",
        "Make Text Stand Out": "Convert the image into black and white to make text easier to read.",
        "Improve Image Clarity": "Adjust image contrast to make faded documents clearer.",
        "Remove Image Noise": "Reduce graininess and unwanted speckles while preserving important details.",
    }

    selected_tool = st.sidebar.selectbox(
        "What would you like to do?",
        list(TOOL_DESCRIPTIONS.keys()),
        index=0,
        help="Choose a document or image processing tool.",
    )

    # Visible description beneath selection for mobile/touch accessibility
    st.sidebar.info(f"💡 {TOOL_DESCRIPTIONS[selected_tool]}")

    filter_ksize = 5
    if selected_tool == "Scan a Document":
        operation = "Full Document Scanner"

    elif selected_tool == "Make Text Stand Out":
        operation = st.sidebar.radio(
            "Method",
            [
                "Otsu Thresholding",
                "Adaptive Gaussian Thresholding",
                "Adaptive Mean Thresholding",
            ],
            index=0,
            help="Choose how the black and white conversion is computed.",
        )

    elif selected_tool == "Improve Image Clarity":
        operation = st.sidebar.radio(
            "Method",
            [
                "Global Histogram Equalization",
                "CLAHE (Adaptive Histogram Equalization)",
            ],
            index=0,
            help="Choose standard or localized contrast balancing.",
        )

    elif selected_tool == "Remove Image Noise":
        operation = st.sidebar.radio(
            "Filter Type",
            [
                "Gaussian Smoothing",
                "Median Filtering",
            ],
            index=0,
            help="Gaussian blurs smoothly; Median removes salt-and-pepper grain without blurring edges.",
        )
        filter_ksize = st.sidebar.slider(
            "Filter Strength (Kernel Size)",
            min_value=3,
            max_value=15,
            value=5,
            step=2,
            help="Odd window size for spatial filter kernel.",
        )

    # Optional Pre-Filter (when not already on the Remove Image Noise tool)
    noise_prefilter = "None"
    prefilter_ksize = 5
    if selected_tool != "Remove Image Noise":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎛️ Optional Noise Filter")
        noise_prefilter = st.sidebar.selectbox(
            "Clean up noise before processing?",
            ["None", "Gaussian Blur", "Median Filter"],
            index=0,
            help="Pre-smooth document to remove scanner or camera grain.",
        )
        if noise_prefilter != "None":
            prefilter_ksize = st.sidebar.slider(
                "Filter Strength",
                min_value=3,
                max_value=15,
                value=5,
                step=2,
            )

    # Display & Analysis Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👁️ Display & Analysis")
    show_boundary = st.sidebar.checkbox(
        "Document Boundary & Corners",
        value=False,
        help="Overlays detected paper boundary polygon and corner markers.",
    )
    side_by_side = st.sidebar.checkbox(
        "Side-by-Side Comparison",
        value=True,
        help="Displays original and processed images side by side.",
    )
    show_histogram = st.sidebar.checkbox(
        "Intensity Histograms",
        value=False,
        help="Displays 256-bin grayscale intensity histograms.",
    )

    # --- Main Page Settings (Relevant for Document Scanner) ---
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

        with st.spinner("🔍 Processing documents..."):
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    image = Image.open(uploaded_file)
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    # 1. Apply Optional Pre-filter
                    if noise_prefilter == "Gaussian Blur":
                        img_processed = apply_noise_filter(img, method="gaussian", ksize=prefilter_ksize)
                    elif noise_prefilter == "Median Filter":
                        img_processed = apply_noise_filter(img, method="median", ksize=prefilter_ksize)
                    else:
                        img_processed = img.copy()

                    # 2. Document boundary visualization
                    if show_boundary:
                        display_original = visualize_document_boundary(img_processed)
                    else:
                        display_original = img

                    # 3. Main Operation Execution
                    if operation == "Full Document Scanner":
                        result = process_image(img_processed, mode=mode, output=output)
                    elif operation == "Otsu Thresholding":
                        result = apply_threshold(img_processed, method="otsu")
                    elif operation == "Adaptive Gaussian Thresholding":
                        result = apply_threshold(img_processed, method="adaptive_gaussian", block_size=11, c=2)
                    elif operation == "Adaptive Mean Thresholding":
                        result = apply_threshold(img_processed, method="adaptive_mean", block_size=11, c=2)
                    elif operation == "Global Histogram Equalization":
                        result = apply_histogram_equalization(img_processed, method="global")
                    elif operation == "CLAHE (Adaptive Histogram Equalization)":
                        result = apply_histogram_equalization(img_processed, method="clahe")
                    elif operation == "Gaussian Smoothing":
                        result = apply_noise_filter(img_processed, method="gaussian", ksize=filter_ksize)
                    elif operation == "Median Filtering":
                        result = apply_noise_filter(img_processed, method="median", ksize=filter_ksize)
                    else:
                        result = process_image(img_processed, mode=mode, output=output)

                    scanned_pil = to_pil(result)
                    scanned_images.append(scanned_pil)

                    # 4. Display Results
                    st.markdown(f"#### 📄 Page {idx + 1}: {uploaded_file.name}")
                    if side_by_side:
                        c_left, c_right = st.columns(2)
                        with c_left:
                            orig_cap = "Original Image" + (" (with Boundary & Corners)" if show_boundary else "")
                            st.image(to_pil(display_original), caption=orig_cap, use_container_width=True)
                        with c_right:
                            st.image(scanned_pil, caption=f"Result: {operation}", use_container_width=True)
                    else:
                        st.image(scanned_pil, caption=f"Result: {operation}", use_container_width=True)

                    # 5. Optional Histogram Display
                    if show_histogram:
                        h_left, h_right = st.columns(2)
                        with h_left:
                            st.image(
                                to_pil(render_histogram_image(img)),
                                caption=f"Original Histogram (Page {idx + 1})",
                                use_container_width=True,
                            )
                        with h_right:
                            st.image(
                                to_pil(render_histogram_image(result)),
                                caption=f"Processed Histogram (Page {idx + 1})",
                                use_container_width=True,
                            )

                    st.markdown("---")

                except Exception as e:
                    st.error(f"⚠️ Error processing {uploaded_file.name}: {e}")

        if scanned_images:
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

    st.markdown(
        """
        <div style="text-align: center; color: #7f8c8d; margin-top: 3rem;">
            <p>Scanify • Document Scanner & Image Enhancement Suite</p>
            <p>Powered by OpenCV & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
