import os
import subprocess
import re
import urllib.request
import zipfile
import shutil
import tempfile
import pandas as pd
from PIL import Image

DATASET_DIR = "./dataset_ordered"
OUTPUT_DIR = "./output_images"
EXE_PATH = r"..\x64\Release\PDC_assignment.exe"

DATASET_URL = "https://github.com/queenie9090/PDC_assignment/archive/refs/heads/main.zip"

SCALE_DOWN = 0.5
SCALE_UP = 1.5

TARGET_BACKEND = "openmp"

if TARGET_BACKEND == "all":
    MODES = ["baseline", "cuda", "openmp", "mpi"]
else:
    MODES = [TARGET_BACKEND]

MPI_NPROCS = 4

# Single combined CSV filename per scale factor
OUTPUT_CSV_DOWN = (
    f"benchmark_downscale_{SCALE_DOWN}x.csv"
)

OUTPUT_CSV_UP = (
    f"benchmark_upscale_{SCALE_UP}x.csv"
)

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def ensure_dataset():

    os.makedirs(DATASET_DIR, exist_ok=True)

    existing_images = [
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith(SUPPORTED_FORMATS)
    ]

    if existing_images:
        print(
            f"Dataset already available "
            f"({len(existing_images)} images)."
        )
        print("Skipping dataset download.")
        print()
        return True

    print("=" * 60)
    print("DATASET NOT FOUND")
    print("=" * 60)
    print("Dataset folder is empty.")
    print("Downloading dataset from GitHub...")
    print()

    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "dataset.zip")
    extract_path = os.path.join(temp_dir, "extracted")

    try:
        urllib.request.urlretrieve(
            DATASET_URL,
            zip_path
        )

        print("Download completed.")
        print("Extracting dataset...")

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(extract_path)

        dataset_source = None

        for root, dirs, files in os.walk(extract_path):

            if os.path.basename(root) == DATASET_FOLDER_NAME:
                dataset_source = root
                break

        if dataset_source is None:
            print(
                f"ERROR: Could not find "
                f"'{DATASET_FOLDER_NAME}' "
                f"inside downloaded repository."
            )
            return False

        copied_images = 0

        for file_name in os.listdir(dataset_source):

            if file_name.lower().endswith(SUPPORTED_FORMATS):

                source_file = os.path.join(
                    dataset_source,
                    file_name
                )

                destination_file = os.path.join(
                    DATASET_DIR,
                    file_name
                )

                shutil.copy2(
                    source_file,
                    destination_file
                )

                copied_images += 1

        if copied_images == 0:
            print("ERROR: No images found in downloaded dataset.")
            return False

        print(
            f"Dataset ready: "
            f"{copied_images} images downloaded."
        )
        print(
            f"Saved to: {DATASET_DIR}"
        )
        print()

        return True

    except Exception as e:
        print("ERROR: Dataset download failed.")
        print(e)
        return False

    finally:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

def get_image_information(img_path):
    try:
        with Image.open(img_path) as im:
            width, height = im.size
            channels = len(im.getbands())

            megapixels = (
                width * height
            ) / 1_000_000

            raw_bytes = (
                width
                * height
                * channels
            )

            raw_mb = (
                raw_bytes
                / (1024 * 1024)
            )

        compressed_kb = (
            os.path.getsize(img_path)
            / 1024.0
        )

        return {
            "width": width,
            "height": height,
            "channels": channels,
            "megapixels": megapixels,
            "raw_bytes": raw_bytes,
            "raw_mb": raw_mb,
            "compressed_kb": compressed_kb
        }

    except Exception as e:
        print(f"Error reading image information: {img_path}")
        print(e)
        return None


def run_program(
    mode,
    img_path,
    output_path,
    resized_w,
    resized_h
):
    command = [
        EXE_PATH,
        img_path,
        output_path,
        str(resized_w),
        str(resized_h),
        mode
    ]

    if mode == "mpi":
        command = [
            "mpiexec",
            "-n",
            str(MPI_NPROCS)
        ] + command

    try:
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        match = re.search(
            r"TIME_MS:\s*([\d.]+)",
            process.stdout
        )

        if match:
            return float(match.group(1))

        print(f"Warning: TIME_MS not found for {mode}")
        print(process.stdout)

        if process.stderr:
            print(process.stderr)

        return None

    except subprocess.CalledProcessError as e:
        print(f"Error executing {mode}:")

        if e.stdout:
            print(f"stdout: {e.stdout}")

        if e.stderr:
            print(f"stderr: {e.stderr}")

        return None

    except FileNotFoundError as e:
        print(f"Command not found while executing {mode}: {e}")
        return None


def run_benchmark(
    scale_factor,
    output_csv,
    benchmark_name
):
    if not os.path.exists(DATASET_DIR):
        print(
            f"Error: Dataset directory "
            f"'{DATASET_DIR}' does not exist."
        )
        return

    if not os.path.exists(EXE_PATH):
        print(
            f"Error: Executable "
            f"'{EXE_PATH}' does not exist."
        )
        return

    images = sorted(
        f
        for f in os.listdir(DATASET_DIR)
        if f.lower().endswith(SUPPORTED_FORMATS)
    )

    if not images:
        print(
            f"No supported images found "
            f"in '{DATASET_DIR}'."
        )
        return

    # Load existing CSV if present to merge new columns
    existing_data = {}
    if os.path.exists(output_csv):
        try:
            old_df = pd.read_csv(output_csv)
            old_df = old_df.where(pd.notnull(old_df), None)
            existing_data = old_df.set_index("Image").to_dict(orient="index")
            print(f"Loaded existing data from {output_csv}")
        except Exception as e:
            print(f"Could not load existing CSV: {e}")

    print("=" * 60)
    print(
        f"{benchmark_name} BENCHMARK "
        f"[{TARGET_BACKEND.upper()}]"
    )
    print("=" * 60)
    print(f"Dataset      : {DATASET_DIR}")
    print(f"Images       : {len(images)}")
    print(f"Scale factor : {scale_factor}x")
    print(f"Modes        : {', '.join(MODES)}")

    if "mpi" in MODES:
        print(f"MPI Ranks    : {MPI_NPROCS}")

    print(f"Output CSV   : {output_csv}")
    print("=" * 60)
    print()

    results = []

    for index, img_name in enumerate(
        images,
        start=1
    ):
        img_path = os.path.join(
            DATASET_DIR,
            img_name
        )

        info = get_image_information(img_path)

        if info is None:
            continue

        orig_w = info["width"]
        orig_h = info["height"]

        resized_w = max(
            1,
            int(orig_w * scale_factor)
        )

        resized_h = max(
            1,
            int(orig_h * scale_factor)
        )

        print(
            f"[{index}/{len(images)}] "
            f"{img_name}: "
            f"{orig_w}x{orig_h} -> "
            f"{resized_w}x{resized_h}"
        )

        # Reuse existing row data if image was processed in a prior run
        if img_name in existing_data:
            row_data = existing_data[img_name]
            row_data["Image"] = img_name
        else:
            row_data = {
                "Image": img_name,
                "Original_Width": orig_w,
                "Original_Height": orig_h,
                "Original_Resolution":
                    f"{orig_w}x{orig_h}",
                "Output_Width": resized_w,
                "Output_Height": resized_h,
                "Output_Resolution":
                    f"{resized_w}x{resized_h}",
                "Channels": info["channels"],
                "Megapixels":
                    round(info["megapixels"], 3),
                "Raw_Size_Bytes":
                    info["raw_bytes"],
                "Input_Image_Size_MB":
                    round(info["raw_mb"], 3),
                "Compressed_File_Size_KB":
                    round(info["compressed_kb"], 2),
                "Scale_Factor":
                    scale_factor
            }

        for mode in MODES:
            output_name = (
                f"{benchmark_name.lower()}_"
                f"{scale_factor}x_"
                f"{mode}_"
                f"{os.path.splitext(img_name)[0]}.png"
            )

            out_path = os.path.join(
                OUTPUT_DIR,
                output_name
            )

            print(
                f"    Running {mode}...",
                flush=True
            )

            execution_time = run_program(
                mode,
                img_path,
                out_path,
                resized_w,
                resized_h
            )

            if execution_time is not None:
                row_data[f"{mode}_ms"] = execution_time

                print(
                    f"    {mode} finished: "
                    f"{execution_time} ms",
                    flush=True
                )
            else:
                row_data[f"{mode}_ms"] = None

                print(
                    f"    {mode} FAILED",
                    flush=True
                )

        # Calculate speedup for all recorded modes using baseline_ms
        baseline_time = row_data.get("baseline_ms")

        if (
            baseline_time is not None
            and baseline_time > 0
        ):
            # Check all existing timing columns in row_data
            for key in list(row_data.keys()):
                if key.endswith("_ms") and key != "baseline_ms":
                    m_name = key[:-3]
                    m_time = row_data[key]

                    if (
                        m_time is not None
                        and m_time > 0
                    ):
                        row_data[
                            f"{m_name}_speedup"
                        ] = round(
                            baseline_time / m_time,
                            3
                        )

        results.append(row_data)

        print()

    if not results:
        print(
            "No benchmark results were generated."
        )
        return

    dataframe = pd.DataFrame(results)

    dataframe.to_csv(
        output_csv,
        index=False
    )

    print("=" * 60)
    print(f"{benchmark_name} COMPLETE")
    print("=" * 60)
    print(
        f"Results saved to: {output_csv}"
    )
    print(
        f"Images processed: {len(results)}"
    )
    print()


if __name__ == "__main__":

    if not ensure_dataset():
        print("Benchmark cannot start because dataset is unavailable.")
        raise SystemExit(1)

    run_benchmark(
        SCALE_DOWN,
        OUTPUT_CSV_DOWN,
        "DOWNSCALE"
    )

    run_benchmark(
        SCALE_UP,
        OUTPUT_CSV_UP,
        "UPSCALE"
    )