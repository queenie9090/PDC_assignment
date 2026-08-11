import streamlit as st
import subprocess
import os
import re
from PIL import Image

EXE_PATH = r"..\x64\Release\PDC_assignment.exe"
TEMP_DIR = "app_output"

os.makedirs(TEMP_DIR, exist_ok=True)

SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]

MIN_SCALE = 0.25
MAX_SCALE = 3.0
MIN_DIMENSION = 64
MAX_DIMENSION = 10000

st.set_page_config(
    page_title="Parallel Image Resizer",
    layout="wide"
)

st.title("Parallel Image Resizer")

st.write(
    "Resize an image using Bicubic interpolation with "
    "Sequential, OpenMP, CUDA, or MPI implementations."
)

if not os.path.exists(EXE_PATH):
    st.error(
        "Executable not found:\n\n"
        + EXE_PATH
        + "\n\nPlease compile the C++ project first."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=SUPPORTED_FORMATS
)

if uploaded_file is None:
    st.info("Please upload a JPG, JPEG, or PNG image.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as e:
    st.error("Unable to read image: " + str(e))
    st.stop()

original_width, original_height = image.size

st.subheader("Image Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Width", str(original_width) + " px")

with col2:
    st.metric("Height", str(original_height) + " px")

with col3:
    st.metric("Channels", "3 (RGB)")

with col4:
    st.metric(
        "Pixels",
        f"{original_width * original_height:,}"
    )

st.image(
    image,
    caption=f"Original Image ({original_width} x {original_height})",
    width="stretch"
)

st.subheader("Resize Settings")

resize_method = st.radio(
    "Resize by:",
    ["Scale Factor", "Custom Resolution"],
    horizontal=True
)

if resize_method == "Scale Factor":

    scale_percent = st.slider(
        "Scale Percentage",
        min_value=25,
        max_value=300,
        value=100,
        step=5
    )

    scale_factor = scale_percent / 100.0

    new_width = max(
        1,
        int(original_width * scale_factor)
    )

    new_height = max(
        1,
        int(original_height * scale_factor)
    )

else:

    keep_ratio = st.checkbox(
        "Keep aspect ratio",
        value=True
    )

    col1, col2 = st.columns(2)

    with col1:
        new_width = st.number_input(
            "Output Width",
            min_value=MIN_DIMENSION,
            max_value=MAX_DIMENSION,
            value=min(
                MAX_DIMENSION,
                max(MIN_DIMENSION, original_width)
            ),
            step=1
        )

    with col2:
        if keep_ratio:

            calculated_height = int(
                new_width * original_height / original_width
            )

            new_height = calculated_height

            st.number_input(
                "Output Height",
                min_value=MIN_DIMENSION,
                max_value=MAX_DIMENSION,
                value=min(
                    MAX_DIMENSION,
                    max(MIN_DIMENSION, calculated_height)
                ),
                disabled=True
            )

        else:

            new_height = st.number_input(
                "Output Height",
                min_value=MIN_DIMENSION,
                max_value=MAX_DIMENSION,
                value=min(
                    MAX_DIMENSION,
                    max(MIN_DIMENSION, original_height)
                ),
                step=1
            )

    scale_factor_x = new_width / original_width
    scale_factor_y = new_height / original_height

    if keep_ratio:
        scale_factor = scale_factor_x
    else:
        st.warning(
            "Stretching is enabled. Width and height use "
            "different scale factors."
        )

        scale_factor = scale_factor_x

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
    st.error(
        f"Output dimensions must be at least "
        f"{MIN_DIMENSION} x {MIN_DIMENSION} pixels."
    )
    st.stop()

if new_width > MAX_DIMENSION or new_height > MAX_DIMENSION:
    st.error(
        f"Output dimensions cannot exceed "
        f"{MAX_DIMENSION} x {MAX_DIMENSION} pixels."
    )
    st.stop()

output_pixels = new_width * new_height

st.info(
    f"Output resolution: {new_width} x {new_height} pixels\n\n"
    f"Scale: {scale_factor:.2f}x\n\n"
    f"Output pixels: {output_pixels:,}\n\n"
    f"Algorithm: Bicubic interpolation"
)

if output_pixels > 50000000:
    st.warning(
        "This output is very large and may require significant "
        "memory and processing time."
    )

if st.button(
    "Resize Image",
    type="primary",
    width="stretch"
):

    input_path = os.path.join(
        TEMP_DIR,
        "input_image.jpg"
    )

    output_path = os.path.join(
        TEMP_DIR,
        "resized_image.jpg"
    )

    try:

        image.save(
            input_path,
            format="JPEG",
            quality=95
        )

        command = [
            EXE_PATH,
            input_path,
            output_path,
            str(scale_factor),
            mode
        ]

        if mode == "mpi":
            command = [
                "mpiexec",
                "-n",
                "4"
            ] + command

        with st.spinner(
            f"Running {mode.upper()} implementation..."
        ):

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

        if process.returncode != 0:

            st.error(
                "The resizing program failed."
            )

            if process.stdout:
                st.code(process.stdout)

            if process.stderr:
                st.code(process.stderr)

            st.stop()

        match = re.search(
            r"TIME_MS:\s*([\d.]+)",
            process.stdout
        )

        execution_time = None

        if match:
            execution_time = float(match.group(1))

        if not os.path.exists(output_path):

            st.error(
                "The program completed, but no output "
                "image was generated."
            )

            if process.stdout:
                st.code(process.stdout)

            st.stop()

        st.success(
            f"{mode.upper()} resizing completed successfully."
        )

        st.subheader("Performance Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            if execution_time is not None:
                st.metric(
                    "Execution Time",
                    f"{execution_time:.3f} ms"
                )
            else:
                st.metric(
                    "Execution Time",
                    "Not available"
                )

        with col2:
            st.metric(
                "Output Width",
                f"{new_width} px"
            )

        with col3:
            st.metric(
                "Output Height",
                f"{new_height} px"
            )

        st.subheader("Resized Image")

        try:

            result_image = Image.open(
                output_path
            ).convert("RGB")

            actual_width, actual_height = result_image.size

            st.image(
                result_image,
                caption=(
                    f"{mode.upper()} - Bicubic "
                    f"({actual_width} x {actual_height})"
                ),
                width="stretch"
            )

            with open(
                output_path,
                "rb"
            ) as file:

                st.download_button(
                    label="Download Resized Image",
                    data=file,
                    file_name="resized_image.jpg",
                    mime="image/jpeg",
                    width="stretch"
                )

        except Exception as e:

            st.error(
                "Unable to display the output image: "
                + str(e)
            )

        with st.expander("Program Output"):

            if process.stdout:
                st.code(process.stdout)
            else:
                st.write("No standard output.")

            if process.stderr:
                st.write("Error output:")
                st.code(process.stderr)

    except FileNotFoundError as e:

        st.error(
            "Unable to start the program: " + str(e)
        )

    except Exception as e:

        st.error(
            "Unexpected error: " + str(e)
        )

st.divider()

st.caption(
    "Parallel Image Resizer | Bicubic | "
    "Sequential | OpenMP | CUDA | MPI"
)