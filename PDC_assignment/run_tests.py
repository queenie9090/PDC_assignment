import os
import subprocess
import re
import pandas as pd
from PIL import Image


# Paths & Settings

DATASET_DIR = "./dataset_ordered"
OUTPUT_DIR = "./output_images"
EXE_PATH = r"..\x64\Release\PDC_assignment.exe"

SCALE_FACTOR = 0.5
OUTPUT_CSV = "benchmark_results.csv"

# Implementations to benchamark
# MODES = ["baseline", "cuda", "openmp", "mpi"]
MODES = ["baseline","cuda"]
#MODES = ["openmp"]
#MODES = ["mpi"]

# Number of MPI processes
MPI_PROCESSES = 4

# Supported image formats
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_image_information(img_path):
    """
    Extract image dimensions, channels, megapixels,
    raw/uncompressed image size and compressed file size.
    """

    try:
        with Image.open(img_path) as im:
            width, height = im.size

            # Convert channel information into a numerical value.
            # RGB/RGBA images are handled directly.
            channels = len(im.getbands())

            megapixels = (width * height) / 1_000_000

            # Approximate raw image size based on decoded pixels.
            # Each channel is assumed to use 1 byte (8-bit image).
            raw_bytes = width * height * channels
            raw_mb = raw_bytes / (1024 * 1024)

        compressed_kb = os.path.getsize(img_path) / 1024.0

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
        print(f"  Error reading image information: {img_path}")
        print(f"  {e}")
        return None


def run_program(mode, img_path, output_path):
    """
    Execute the C++ image resizing program and return
    the measured execution time in milliseconds.
    """

    base_command = [
        EXE_PATH,
        img_path,
        output_path,
        str(SCALE_FACTOR),
        mode
    ]

    # MPI requires mpiexec when the executable itself
    # is an MPI program.
    if mode == "mpi":
        command = [
            "mpiexec",
            "-n",
            str(MPI_PROCESSES)
        ] + base_command
    else:
        command = base_command

    try:
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # Expected output:
        # TIME_MS:<value>
        match = re.search(r"TIME_MS:\s*([\d.]+)", process.stdout)

        if match:
            return float(match.group(1))

        print(f"  Warning: TIME_MS not found for {mode}")
        print(f"  Program output: {process.stdout}")

        if process.stderr:
            print(f"  Program error output: {process.stderr}")

        return None

    except subprocess.CalledProcessError as e:
        print(f"  Error executing {mode}:")
        print(f"  {e}")

        if e.stdout:
            print(f"  stdout: {e.stdout}")

        if e.stderr:
            print(f"  stderr: {e.stderr}")

        return None

    except FileNotFoundError as e:
        print(f"  Command not found while executing {mode}: {e}")
        return None

def run_benchmark():



    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' does not exist.")
        return

    if not os.path.exists(EXE_PATH):
        print(f"Error: Executable '{EXE_PATH}' does not exist.")
        return

    # Sort filenames to ensure consistent processing order.
    images = sorted(
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith(SUPPORTED_FORMATS)
    )

    if not images:
        print(f"No supported images found in '{DATASET_DIR}'.")
        return

    print("=" * 70)
    print("AUTOMATED IMAGE RESIZER BENCHMARK")
    print("=" * 70)
    print(f"Dataset       : {DATASET_DIR}")
    print(f"Images        : {len(images)}")
    print(f"Scale factor  : {SCALE_FACTOR}")
    print(f"Modes         : {', '.join(MODES)}")
    print(f"MPI processes : {MPI_PROCESSES}")
    print("=" * 70)
    print()

    results = []
   

    for index, img_name in enumerate(images, start=1):

        img_path = os.path.join(DATASET_DIR, img_name)

        # Output file name
        out_path = os.path.join(
            OUTPUT_DIR,
            f"resized_{img_name}"
        )
       
        # Get image information

        info = get_image_information(img_path)

        if info is None:
            continue

        orig_w = info["width"]
        orig_h = info["height"]
        channels = info["channels"]

        resized_w = max(1, int(orig_w * SCALE_FACTOR))
        resized_h = max(1, int(orig_h * SCALE_FACTOR))

        megapixels = info["megapixels"]
        raw_bytes = info["raw_bytes"]
        raw_mb = info["raw_mb"]
        compressed_kb = info["compressed_kb"]

        print(
            f"[{index}/{len(images)}] "
            f"{img_name}: "
            f"{orig_w}x{orig_h} -> "
            f"{resized_w}x{resized_h}"
        )

        # Create result row       

        row_data = {
            "Image": img_name,

            "Original_Width": orig_w,
            "Original_Height": orig_h,
            "Original_Resolution": f"{orig_w}x{orig_h}",

            "Output_Width": resized_w,
            "Output_Height": resized_h,
            "Output_Resolution": f"{resized_w}x{resized_h}",

            "Channels": channels,
            "Megapixels": round(megapixels, 3),

            # Actual decoded/uncompressed workload size
            "Raw_Size_Bytes": raw_bytes,
            "Raw_Size_MB": round(raw_mb, 3),

            # Kept only as supplementary information
            "Compressed_File_Size_KB": round(compressed_kb, 2),

            "Scale_Factor": SCALE_FACTOR
        }

        # Execute each implementation        

        for mode in MODES:

            execution_time = run_program(
                mode,
                img_path,
                out_path
            )

            if execution_time is not None:
                row_data[f"{mode}_ms"] = execution_time
                #print(f"{execution_time:.3f} ms")
            else:
                row_data[f"{mode}_ms"] = None
                print("FAILED")
        
        # Calculate speedup       

        baseline_time = row_data.get("baseline_ms")

        if baseline_time is not None and baseline_time > 0:

            for mode in MODES:

                if mode == "baseline":
                    continue

                mode_time = row_data.get(f"{mode}_ms")

                if mode_time is not None and mode_time > 0:

                    speedup = baseline_time / mode_time

                    row_data[f"{mode}_speedup"] = round(
                        speedup,
                        3
                    )

                else:
                    row_data[f"{mode}_speedup"] = None

        results.append(row_data)

        print()

    
    if not results:
        print("No benchmark results were generated.")
        return

    dataframe = pd.DataFrame(results)

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Images processed : {len(results)}")
    print()

if __name__ == "__main__":
    run_benchmark()