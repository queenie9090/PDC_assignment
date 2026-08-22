import streamlit as st
import subprocess
import os
import re
import base64
import cv2
import numpy as np
import streamlit.components.v1 as components

EXE_PATH = r"..\x64\Release\PDC_assignment.exe"
TEMP_DIR = "app_output"

os.makedirs(TEMP_DIR, exist_ok=True)

SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MIN_DIMENSION = 32
MAX_DIMENSION = 30000

# SAFETY LIMITS TO PREVENT BROWSER / SYSTEM CRASHES
MAX_DISPLAY_PIXELS = 50_000_000   # ~7000x7000 px limit for HTML/Base64 browser rendering
MAX_TOTAL_PIXELS = 150_000_000     # ~12000x12000 px maximum computational limit

st.set_page_config(
    page_title="Parallel Image Resizer",
    layout="wide"
)

st.title("Parallel Image Resizer")
st.write("Resize an image using Bicubic interpolation with Sequential, OpenMP, CUDA, or MPI implementations.")

if not os.path.exists(EXE_PATH):
    st.error(f"Executable not found:\n\n{EXE_PATH}\n\nPlease compile the C++ project first.")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=SUPPORTED_FORMATS)

if uploaded_file is None:
    st.info("Please upload a JPG, JPEG, or PNG image.")
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    image = cv2.imdecode(
        np.frombuffer(file_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Unable to decode image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
except Exception as e:
    st.error(f"Unable to read image: {e}")
    st.stop()

original_height, original_width = image.shape[:2]

st.subheader("Image Information")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Width", f"{original_width} px")
with col2:
    st.metric("Height", f"{original_height} px")
with col3:
    st.metric("Channels", "3 (RGB)")
with col4:
    st.metric("Pixels", f"{original_width * original_height:,}")

st.image(image, caption=f"Original Image ({original_width} x {original_height})", use_container_width=True)

st.subheader("Resize Settings")
resize_method = st.radio("Resize by:", ["Scale Factor", "Custom Resolution"], horizontal=True)

if resize_method == "Scale Factor":
    scale_percent = st.number_input(
        "Scale Percentage (%)",
        min_value=10.0,
        max_value=1000.0,
        value=100.0,
        step=5.0,
        format="%.1f"
    )
    scale_factor = scale_percent / 100.0
    new_width = max(1, int(original_width * scale_factor))
    new_height = max(1, int(original_height * scale_factor))
else:
    keep_ratio = st.checkbox("Keep aspect ratio", value=True)
    col1, col2 = st.columns(2)
    with col1:
        new_width = st.number_input(
            "Output Width",
            min_value=MIN_DIMENSION,
            max_value=MAX_DIMENSION,
            value=min(MAX_DIMENSION, max(MIN_DIMENSION, original_width)),
            step=1
        )
    with col2:
        if keep_ratio:
            calculated_height = int(new_width * original_height / original_width)
            new_height = calculated_height
            st.number_input(
                "Output Height",
                min_value=MIN_DIMENSION,
                max_value=MAX_DIMENSION,
                value=min(MAX_DIMENSION, max(MIN_DIMENSION, calculated_height)),
                disabled=True
            )
        else:
            new_height = st.number_input(
                "Output Height",
                min_value=MIN_DIMENSION,
                max_value=MAX_DIMENSION,
                value=min(MAX_DIMENSION, max(MIN_DIMENSION, original_height)),
                step=1
            )
            
    if not keep_ratio:
        st.warning("Aspect ratio disabled: Image stretching will occur.")

st.subheader("Implementation")
mode = st.selectbox(
    "Choose the implementation:",
    ["baseline", "openmp", "cuda", "mpi"],
    format_func=lambda x: {
        "baseline": "Sequential Baseline",
        "openmp": "OpenMP",
        "cuda": "CUDA",
        "mpi": "MPI"
    }[x]
)

if new_width < MIN_DIMENSION or new_height < MIN_DIMENSION:
    st.error(f"Output dimensions must be at least {MIN_DIMENSION} x {MIN_DIMENSION} pixels.")
    st.stop()

if new_width > MAX_DIMENSION or new_height > MAX_DIMENSION:
    st.error(f"Output dimensions cannot exceed {MAX_DIMENSION} x {MAX_DIMENSION} pixels.")
    st.stop()

output_pixels = new_width * new_height

# COMPUTATIONAL SAFETY CHECK
if output_pixels > MAX_TOTAL_PIXELS:
    st.error(
        f"Target resolution ({new_width} x {new_height} = {output_pixels:,} pixels) exceeds "
        f"the maximum safety compute limit of {MAX_TOTAL_PIXELS:,} pixels to prevent memory crash."
    )
    st.stop()

st.info(
    f"Output resolution: {new_width} x {new_height} pixels\n\n"
    f"Output pixels: {output_pixels:,}\n\n"
    f"Algorithm: Bicubic interpolation"
)

def image_to_base64(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, encoded_image = cv2.imencode(".png", img_bgr)
    if not success:
        raise ValueError("Unable to encode image as PNG.")
    return base64.b64encode(encoded_image.tobytes()).decode()

def render_multi_zoom_viewer(items):
    """
    Renders an HTML view containing multiple syncable/zoomable viewports.
    items: list of tuples -> (b64_str, title, width, height)
    """
    cards_html = ""
    setup_scripts = ""

    for idx, (b64_str, title, w, h) in enumerate(items, 1):
        vp_id = f"vp{idx}"
        img_id = f"img{idx}"
        cards_html += f"""
        <div class="img-card">
            <div class="img-title">{title} ({w} x {h})</div>
            <div class="viewport" id="{vp_id}"><img class="zoom-img" id="{img_id}" src="data:image/png;base64,{b64_str}"></div>
            <button class="reset-btn" onclick="resetView('{vp_id}', '{img_id}')">Reset View</button>
        </div>
        """
        setup_scripts += f"setupViewer('{vp_id}', '{img_id}');\n"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: sans-serif; background: transparent; }}
        .container {{ display: flex; gap: 16px; width: 100%; }}
        .img-card {{ flex: 1; min-width: 0; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; background: #fafafa; }}
        .img-title {{ text-align: center; font-weight: 600; font-size: 13px; margin-bottom: 8px; color: #333; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .viewport {{ width: 100%; height: 420px; border: 1px solid #ccc; border-radius: 4px; overflow: hidden; position: relative; background: #1a1a1a; cursor: grab; }}
        .viewport.dragging {{ cursor: grabbing; }}
        .zoom-img {{ position: absolute; left: 0; top: 0; max-width: none; user-select: none; -webkit-user-drag: none; }}
        .reset-btn {{ display: block; margin: 8px auto 0 auto; padding: 5px 14px; font-size: 12px; background: white; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; }}
        .reset-btn:hover {{ background: #f0f0f0; }}
    </style>
    </head>
    <body>
    <div class="container">
        {cards_html}
    </div>
    <script>
    const viewers = {{}};
    function setupViewer(viewportId, imageId) {{
        const viewport = document.getElementById(viewportId);
        const img = document.getElementById(imageId);
        viewers[viewportId] = {{ scale: 1, x: 0, y: 0, dragging: false, lastX: 0, lastY: 0, baseWidth: 0, baseHeight: 0 }};
        const state = viewers[viewportId];
        img.onload = function() {{ fitImage(viewportId, imageId); }};
        if (img.complete) {{ fitImage(viewportId, imageId); }}

        viewport.addEventListener("wheel", function(event) {{
            event.preventDefault();
            const rect = viewport.getBoundingClientRect();
            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;
            const oldScale = state.scale;
            let zoomFactor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
            let newScale = Math.max(0.1, Math.min(20, oldScale * zoomFactor));
            const actualZoom = newScale / oldScale;
            state.x = mouseX - (mouseX - state.x) * actualZoom;
            state.y = mouseY - (mouseY - state.y) * actualZoom;
            state.scale = newScale;
            updateImage(viewportId, imageId);
        }}, {{ passive: false }});

        viewport.addEventListener("mousedown", function(event) {{
            if (event.button !== 0) return;
            state.dragging = true;
            state.lastX = event.clientX;
            state.lastY = event.clientY;
            viewport.classList.add("dragging");
            event.preventDefault();
        }});

        window.addEventListener("mousemove", function(event) {{
            if (!state.dragging) return;
            state.x += event.clientX - state.lastX;
            state.y += event.clientY - state.lastY;
            state.lastX = event.clientX;
            state.lastY = event.clientY;
            updateImage(viewportId, imageId);
        }});

        window.addEventListener("mouseup", function() {{
            state.dragging = false;
            viewport.classList.remove("dragging");
        }});
    }}

    function fitImage(viewportId, imageId) {{
        const viewport = document.getElementById(viewportId);
        const img = document.getElementById(imageId);
        const state = viewers[viewportId];
        if (img.naturalWidth === 0 || img.naturalHeight === 0) return;
        const viewportWidth = viewport.clientWidth;
        const viewportHeight = viewport.clientHeight;
        const imageRatio = img.naturalWidth / img.naturalHeight;
        const viewportRatio = viewportWidth / viewportHeight;
        let width, height;
        if (imageRatio > viewportRatio) {{
            width = viewportWidth;
            height = width / imageRatio;
        }} else {{
            height = viewportHeight;
            width = height * imageRatio;
        }}
        state.baseWidth = width;
        state.baseHeight = height;
        state.scale = 1;
        state.x = (viewportWidth - width) / 2;
        state.y = (viewportHeight - height) / 2;
        img.style.width = width + "px";
        img.style.height = height + "px";
        updateImage(viewportId, imageId);
    }}

    function updateImage(viewportId, imageId) {{
        const img = document.getElementById(imageId);
        const state = viewers[viewportId];
        img.style.transform = "translate(" + state.x + "px, " + state.y + "px) scale(" + state.scale + ")";
    }}

    function resetView(viewportId, imageId) {{
        fitImage(viewportId, imageId);
    }}

    {setup_scripts}
    </script>
    </body>
    </html>
    """

if st.button("Resize Image", type="primary", use_container_width=True):
    input_path = os.path.join(TEMP_DIR, "input_image.png")
    output_path = os.path.join(TEMP_DIR, "resized_image.png")

    try:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(input_path, image_bgr):
            raise ValueError("Unable to save input image.")

        # 1. Run Selected C++ Implementation
        command = [EXE_PATH, input_path, output_path, str(new_width), str(new_height), mode]
        if mode == "mpi":
            command = ["mpiexec", "-n", "4"] + command

        with st.spinner(f"Running {mode.upper()} implementation..."):
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

        if process.returncode != 0:
            st.error("The resizing program failed.")
            if process.stdout: st.code(process.stdout)
            if process.stderr: st.code(process.stderr)
            st.stop()

        match = re.search(r"TIME_MS:\s*([\d.]+)", process.stdout)
        execution_time = float(match.group(1)) if match else None

        if not os.path.exists(output_path):
            st.error("The program completed, but no output image was generated.")
            if process.stdout: st.code(process.stdout)
            st.stop()

        # 2. Benchmark Algorithms with OpenCV
        opencv_bicubic_bgr = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        opencv_knn_bgr = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        opencv_bilinear_bgr = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        st.session_state["has_run"] = True
        st.session_state["output_path"] = output_path
        st.session_state["execution_time"] = execution_time
        st.session_state["opencv_bicubic_rgb"] = cv2.cvtColor(opencv_bicubic_bgr, cv2.COLOR_BGR2RGB)
        st.session_state["opencv_knn_rgb"] = cv2.cvtColor(opencv_knn_bgr, cv2.COLOR_BGR2RGB)
        st.session_state["opencv_bilinear_rgb"] = cv2.cvtColor(opencv_bilinear_bgr, cv2.COLOR_BGR2RGB)
        st.session_state["mode"] = mode
        st.session_state["stdout"] = process.stdout
        st.session_state["stderr"] = process.stderr

    except Exception as e:
        st.error(f"Error executing command: {e}")

if st.session_state.get("has_run", False):
    output_path = st.session_state["output_path"]
    execution_time = st.session_state["execution_time"]
    run_mode = st.session_state["mode"]

    st.success(f"{run_mode.upper()} resizing completed successfully.")
    
    # -------------------------------------------------------------
    # SECTION 1: Standard Results (Original vs Selected Output)
    # -------------------------------------------------------------
    st.subheader("Performance Result")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Execution Time", f"{execution_time:.3f} ms" if execution_time else "N/A")
    with col2:
        st.metric("Output Width", f"{new_width} px")
    with col3:
        st.metric("Output Height", f"{new_height} px")

    st.subheader("Original vs Resized")

    try:
        result_image = cv2.imread(output_path, cv2.IMREAD_COLOR)

        if result_image is None:
            raise ValueError("Unable to read output image.")

        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        actual_height, actual_width = result_image.shape[:2]

        if actual_width * actual_height <= MAX_DISPLAY_PIXELS:
            img1_b64 = image_to_base64(image)
            img2_b64 = image_to_base64(result_image)

            html_view1 = render_multi_zoom_viewer([
                (img1_b64, "Original", original_width, original_height),
                (img2_b64, f"{run_mode.upper()} Bicubic", actual_width, actual_height)
            ])
            components.html(html_view1, height=520)
        else:
            st.warning(
                f"Output image is too large ({actual_width} x {actual_height}) to load safely inside the browser viewer. "
                "Interactive viewer disabled to prevent browser freeze. Use the download button below to inspect the full file."
            )

        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Resized Image",
                data=file,
                file_name="resized_image.png",
                mime="image/png",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Unable to display output image: {e}")

    # -------------------------------------------------------------
    # SECTION 2: Dynamic Comparison (OpenCV vs Selected Mode)
    # -------------------------------------------------------------
    st.divider()
    st.header(f"OpenCV vs C++ {run_mode.upper()} Image Comparison")

    opencv_bicubic_rgb = st.session_state.get("opencv_bicubic_rgb")

    try:
        if actual_width * actual_height <= MAX_DISPLAY_PIXELS:
            cv_b64 = image_to_base64(opencv_bicubic_rgb)
            mode_b64 = image_to_base64(result_image)

            html_view2 = render_multi_zoom_viewer([
                (cv_b64, "OpenCV (cv2.INTER_CUBIC)", actual_width, actual_height),
                (mode_b64, f"C++ {run_mode.upper()} Implementation", actual_width, actual_height)
            ])
            components.html(html_view2, height=520)
        else:
            st.warning("Comparison image size exceeds interactive rendering threshold.")

    except Exception as e:
        st.error(f"Failed to generate comparison with OpenCV: {e}")

    # -------------------------------------------------------------
    # SECTION 3: Interpolation Method Comparison (KNN vs Bilinear vs Selected C++)
    # -------------------------------------------------------------
    st.divider()
    st.header("Interpolation Method Comparison (KNN vs Bilinear vs Our Bicubic)")
    st.write("Compare visual clarity and artifacting across three interpolation techniques at target resolution:")

    opencv_knn_rgb = st.session_state.get("opencv_knn_rgb")
    opencv_bilinear_rgb = st.session_state.get("opencv_bilinear_rgb")

    try:
        if actual_width * actual_height <= MAX_DISPLAY_PIXELS:
            knn_b64 = image_to_base64(opencv_knn_rgb)
            bilinear_b64 = image_to_base64(opencv_bilinear_rgb)
            our_bicubic_b64 = image_to_base64(result_image)

            html_view3 = render_multi_zoom_viewer([
                (knn_b64, "Nearest Neighbor (KNN)", actual_width, actual_height),
                (bilinear_b64, "Bilinear (OpenCV)", actual_width, actual_height),
                (our_bicubic_b64, f"Our Bicubic ({run_mode.upper()})", actual_width, actual_height)
            ])
            components.html(html_view3, height=520)
        else:
            st.warning("Quality comparison image size exceeds interactive rendering threshold.")

    except Exception as e:
        st.error(f"Failed to render algorithm comparison: {e}")

    # Output Logs
    with st.expander("Program Output"):
        if st.session_state["stdout"]:
            st.code(st.session_state["stdout"])
        if st.session_state["stderr"]:
            st.write("Error output:")
            st.code(st.session_state["stderr"])

st.divider()
st.caption("Parallel Image Resizer | Bicubic | Sequential | OpenMP | CUDA | MPI")