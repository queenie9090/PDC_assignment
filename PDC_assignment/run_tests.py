import os
import subprocess
import re
import pandas as pd
from PIL import Image

# Paths & Settings
DATASET_DIR = "./dataset_ordered"
OUTPUT_DIR = "./output_images"
EXE_PATH = r"..\x64\Release\PDC_assignment.exe"  # Compiled executable
SCALE_FACTOR = 0.5                              # 50% resize factor
OUTPUT_CSV = "benchmark_results.csv"

# Active modes to benchmark
MODES = ["baseline", "cuda","openmp", "mpi"]
# MODES = ["baseline", "cuda"]
# MODES = ["baseline", "openmp"]
# MODES = ["baseline", "mpi"]

# Number of MPI processes
MPI_PROCESSES = 4

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_benchmark():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' does not exist.")
        return

    images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        print(f"No images found in '{DATASET_DIR}'.")
        return

    results = []
    print(f"Starting automation on {len(images)} images across modes: {MODES}\n")

    for idx, img_name in enumerate(images, start=1):
        img_path = os.path.join(DATASET_DIR, img_name)
        out_path = os.path.join(OUTPUT_DIR, f"resized_{img_name}")

        # 1. Extract Workload Details (Image Dimensions & Megapixels)
        try:
            with Image.open(img_path) as im:
                orig_w, orig_h = im.size
                resized_w = int(orig_w * SCALE_FACTOR)
                resized_h = int(orig_h * SCALE_FACTOR)
                megapixels = (orig_w * orig_h) / 1e6
        except Exception:
            orig_w = orig_h = resized_w = resized_h = megapixels = 0

        file_size_kb = os.path.getsize(img_path) / 1024.0

        row_data = {
            "Image": img_name,
            "Original_Width": orig_w,
            "Original_Height": orig_h,
            "Resolution": f"{orig_w}x{orig_h}",
            "Resized_Width": resized_w,
            "Resized_Height": resized_h,
            "Megapixels": megapixels,
            "Size_KB": round(file_size_kb, 2)
        }

        print(f"[{idx}/{len(images)}] Benchmarking {img_name} ({orig_w}x{orig_h} -> {resized_w}x{resized_h})...")

        # 2. Execute Modes and Extract Times
        for mode in MODES:
            cmd = [EXE_PATH, img_path, out_path, str(SCALE_FACTOR), mode]

            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )

                # Match execution output printed as TIME_MS:<val>
                match = re.search(r"TIME_MS:([\d.]+)", proc.stdout)
                if match:
                    row_data[f"{mode}_ms"] = float(match.group(1))
                else:
                    print(f"  Warning: Could not parse TIME_MS for {mode} on {img_name}")
                    row_data[f"{mode}_ms"] = None

            except subprocess.CalledProcessError as e:
                print(f"  Error executing {mode} on {img_name}: {e}")
                row_data[f"{mode}_ms"] = None

        # 3. Calculate Speedup Ratio (Baseline Time / Parallel Mode Time)
        baseline_time = row_data.get("baseline_ms")
        if baseline_time and baseline_time > 0:
            for mode in MODES:
                if mode != "baseline":
                    mode_time = row_data.get(f"{mode}_ms")
                    if mode_time and mode_time > 0:
                        speedup = baseline_time / mode_time
                        row_data[f"{mode}_speedup"] = round(speedup, 2)

        results.append(row_data)

    # 4. Save structured results to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAutomation Complete! Results saved to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    run_benchmark()