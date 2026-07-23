import os
import subprocess
import re
import pandas as pd

# Paths
DATASET_DIR = "./dataset_ordered"
OUTPUT_DIR = "./output_images"
EXE_PATH = r"..\x64\Release\PDC_assignment.exe"  # Path to your compiled executable
SCALE_FACTOR = "0.5"                           # 50% resize factor
OUTPUT_CSV = "benchmark_results.csv"

# Modes to benchmark (Add "openmp", "mpi" here as team members complete them)
MODES = ["baseline", "cuda"]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_benchmark():
    # Get and sort images by file size (smallest to largest)
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' does not exist.")
        return

    images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    images.sort(key=lambda x: os.path.getsize(os.path.join(DATASET_DIR, x)))

    if not images:
        print(f"No images found in '{DATASET_DIR}'.")
        return

    results = []

    print(f"Starting automation on {len(images)} images across modes: {MODES}\n")

    for idx, img_name in enumerate(images, start=1):
        img_path = os.path.join(DATASET_DIR, img_name)
        file_size_kb = os.path.getsize(img_path) / 1024.0
        
        # Save output images inside the dedicated output_images directory
        out_path = os.path.join(OUTPUT_DIR, f"resized_{img_name}")

        row_data = {
            "Image": img_name,
            "Size_KB": round(file_size_kb, 2)
        }

        print(f"[{idx}/{len(images)}] Processing {img_name} ({file_size_kb:.1f} KB)...")

        for mode in MODES:
            # Command: PDC_assignment.exe <input_path> <output_path> <scale_factor> <mode>
            cmd = [EXE_PATH, img_path, out_path, SCALE_FACTOR, mode]

            try:
                # Execute process and capture C++ std::cout output
                proc = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True, 
                    check=True
                )
                
                # Extract execution time printed as TIME_MS:<val>
                match = re.search(r"TIME_MS:([\d.]+)", proc.stdout)
                if match:
                    time_ms = float(match.group(1))
                    row_data[mode] = time_ms
                else:
                    print(f"  Warning: Could not parse TIME_MS output for {mode} on {img_name}")
                    row_data[mode] = None

            except subprocess.CalledProcessError as e:
                print(f"  Error running {mode} on {img_name}:")
                if e.stdout:
                    print(f"    STDOUT: {e.stdout.strip()}")
                if e.stderr:
                    print(f"    STDERR: {e.stderr.strip()}")
                row_data[mode] = None

        # Calculate speedup ratios against baseline (Baseline_Time / Parallel_Time)
        if row_data.get("baseline") and row_data["baseline"] > 0:
            for mode in MODES:
                if mode != "baseline" and row_data.get(mode) and row_data[mode] > 0:
                    speedup = row_data["baseline"] / row_data[mode]
                    row_data[f"{mode}_speedup"] = round(speedup, 2)

        results.append(row_data)

    # Save all results to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAutomation Complete! Results saved to '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    run_benchmark()