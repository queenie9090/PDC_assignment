import os
import subprocess
import re
import pandas as pd
from PIL import Image

DATASET_DIR = "./dataset_ordered"
OUTPUT_DIR = "./output_images"
EXE_PATH = r"..\x64\Release\PDC_assignment.exe"

SCALE_DOWN = 0.5
SCALE_UP = 1.5

OUTPUT_CSV_DOWN = "benchmark_downscale.csv"
OUTPUT_CSV_UP = "benchmark_upscale.csv"

MODES = ["baseline", "cuda"]
# MODES = ["baseline", "openmp"]
# MODES = ["baseline", "mpi"]

MPI_PROCESSES = 4

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_image_information(img_path):
    """
    Extract image dimensions, channels, megapixels,
    uncompressed image size and compressed file size.
    """

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
        print(
            f"Error reading image information: "
            f"{img_path}"
        )
        print(e)
        return None


def run_program(
    mode,
    img_path,
    output_path,
    scale_factor
):
    """
    Execute the C++ image resizing program
    and return execution time in milliseconds.
    """

    base_command = [
        EXE_PATH,
        img_path,
        output_path,
        str(scale_factor),
        mode
    ]

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

        match = re.search(
            r"TIME_MS:\s*([\d.]+)",
            process.stdout
        )

        if match:
            return float(
                match.group(1)
            )

        print(
            f"Warning: TIME_MS not found "
            f"for {mode}"
        )

        print(
            f"Program output: "
            f"{process.stdout}"
        )

        if process.stderr:
            print(
                f"Program error output: "
                f"{process.stderr}"
            )

        return None

    except subprocess.CalledProcessError as e:

        print(
            f"Error executing {mode}:"
        )

        print(e)

        if e.stdout:
            print(
                f"stdout: {e.stdout}"
            )

        if e.stderr:
            print(
                f"stderr: {e.stderr}"
            )

        return None

    except FileNotFoundError as e:

        print(
            f"Command not found while "
            f"executing {mode}: {e}"
        )

        return None


def run_benchmark(
    scale_factor,
    output_csv,
    benchmark_name
):

    if not os.path.exists(
        DATASET_DIR
    ):

        print(
            f"Error: Dataset directory "
            f"'{DATASET_DIR}' does not exist."
        )

        return

    if not os.path.exists(
        EXE_PATH
    ):

        print(
            f"Error: Executable "
            f"'{EXE_PATH}' does not exist."
        )

        return

    images = sorted(
        f
        for f in os.listdir(
            DATASET_DIR
        )
        if f.lower().endswith(
            SUPPORTED_FORMATS
        )
    )

    if not images:

        print(
            f"No supported images found "
            f"in '{DATASET_DIR}'."
        )

        return

    print("=" * 60)
    print(
        f"{benchmark_name} BENCHMARK"
    )
    print("=" * 60)
    print(
        f"Dataset      : {DATASET_DIR}"
    )
    print(
        f"Images       : {len(images)}"
    )
    print(
        f"Scale factor : {scale_factor}"
    )
    print(
        f"Modes        : {', '.join(MODES)}"
    )
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

        out_path = os.path.join(
            OUTPUT_DIR,
            f"{benchmark_name.lower()}_{img_name}"
        )

        info = get_image_information(
            img_path
        )

        if info is None:
            continue

        orig_w = info["width"]
        orig_h = info["height"]

        resized_w = max(
            1,
            int(
                orig_w
                * scale_factor
            )
        )

        resized_h = max(
            1,
            int(
                orig_h
                * scale_factor
            )
        )

        print(
            f"[{index}/{len(images)}] "
            f"{img_name}: "
            f"{orig_w}x{orig_h} -> "
            f"{resized_w}x{resized_h}"
        )

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
                round(
                    info["megapixels"],
                    3
                ),

            "Raw_Size_Bytes":
                info["raw_bytes"],

            "Raw_Size_MB":
                round(
                    info["raw_mb"],
                    3
                ),

            "Compressed_File_Size_KB":
                round(
                    info["compressed_kb"],
                    2
                ),

            "Scale_Factor":
                scale_factor
        }

        for mode in MODES:

            execution_time = run_program(
                mode,
                img_path,
                out_path,
                scale_factor
            )

            if execution_time is not None:

                row_data[
                    f"{mode}_ms"
                ] = execution_time

            else:

                row_data[
                    f"{mode}_ms"
                ] = None

                print(
                    f"{mode} FAILED"
                )

        baseline_time = (
            row_data.get(
                "baseline_ms"
            )
        )

        if (
            baseline_time is not None
            and
            baseline_time > 0
        ):

            for mode in MODES:

                if mode == "baseline":
                    continue

                mode_time = row_data.get(
                    f"{mode}_ms"
                )

                if (
                    mode_time is not None
                    and
                    mode_time > 0
                ):

                    speedup = (
                        baseline_time
                        / mode_time
                    )

                    row_data[
                        f"{mode}_speedup"
                    ] = round(
                        speedup,
                        3
                    )

                else:

                    row_data[
                        f"{mode}_speedup"
                    ] = None

        results.append(
            row_data
        )

        print()

    if not results:

        print(
            "No benchmark results "
            "were generated."
        )

        return

    dataframe = pd.DataFrame(
        results
    )

    dataframe.to_csv(
        output_csv,
        index=False
    )

    print("=" * 60)
    print(
        f"{benchmark_name} COMPLETE"
    )
    print("=" * 60)
    print(
        f"Results saved to: "
        f"{output_csv}"
    )
    print(
        f"Images processed: "
        f"{len(results)}"
    )
    print()


if __name__ == "__main__":

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