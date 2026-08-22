import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# FILE PATHS
files = {
    'downscale': [
        'M1_benchmark_downscale_0.5x.csv',
        'M2_benchmark_downscale_0.5x.csv',
        'M3_benchmark_downscale_0.5x.csv'
    ],
    'upscale': [
        'M1_benchmark_upscale_1.5x.csv',
        'M2_benchmark_upscale_1.5x.csv',
        'M3_benchmark_upscale_1.5x.csv'
    ]
}

# Folder for generated graphs
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)


# Metrics to average across the 3 members
metric_cols = [
    'baseline_ms',
    'openmp_ms', 'openmp_speedup',
    'cuda_ms', 'cuda_speedup',
    'mpi_ms', 'mpi_speedup'
]

# Information used to identify each image
meta_cols = [
    'Image',
    'Input_Image_Size_MB',
    'Original_Resolution',
    'Output_Resolution',
    'Megapixels'
]

# AVERAGE RESULTS FROM 3 MEMBERS
def process_averages(file_list):

    # Load all 3 member CSV files
    dfs = [pd.read_csv(f) for f in file_list]

    # Combine all results
    combined = pd.concat(dfs, ignore_index=True)

    # Average the results for each image
    averaged_df = combined.groupby(
        meta_cols,
        as_index=False
    )[metric_cols].mean()

    # Calculate Throughput (MP/s) = Megapixels / (Execution Time (ms) / 1000)
    throughput_mapping = {
        'baseline_ms': 'baseline_throughput_MPps',
        'openmp_ms': 'openmp_throughput_MPps',
        'cuda_ms': 'cuda_throughput_MPps',
        'mpi_ms': 'mpi_throughput_MPps'
    }

    for time_col, tp_col in throughput_mapping.items():
        if time_col in averaged_df.columns:
            averaged_df[tp_col] = np.where(
                (
                    averaged_df[time_col].notna()
                    & (averaged_df[time_col] > 0)
                    & averaged_df['Megapixels'].notna()
                ),
                averaged_df['Megapixels'] / (averaged_df[time_col] / 1000.0),
                np.nan
            )
            averaged_df[tp_col] = averaged_df[tp_col].round(3)

    return averaged_df


# Generate averaged results
avg_downscale = process_averages(files['downscale'])
avg_upscale = process_averages(files['upscale'])



# SAVE AVERAGED CSV FILES

avg_downscale.to_csv(
    'averaged_downscale_0.5x.csv',
    index=False
)

avg_upscale.to_csv(
    'averaged_upscale_1.5x.csv',
    index=False
)

print("Averaged CSV files generated successfully.")



# GRAPH FUNCTION
def generate_graph(
    df,
    title,
    filename,
    implementations
):

    plt.figure(figsize=(10, 6))

    x = df['Input_Image_Size_MB']

    # --------------------------------------------------------
    # Sequential baseline = 1x
    # --------------------------------------------------------

    plt.axhline(
        y=1,
        linestyle='--',
        linewidth=1.5,
        color='black',
        label='Sequential Baseline (1×)'
    )

    # --------------------------------------------------------
    # Scatter + trendline
    # --------------------------------------------------------

    for name, column in implementations.items():

        if column not in df.columns:
            continue

        y = df[column]

        # Scatter plot
        plt.scatter(
            x,
            y,
            s=45,
            alpha=0.75,
            label=name
        )

        # Remove invalid values before calculating trendline
        valid = (
            x.notna()
            & y.notna()
            & np.isfinite(x)
            & np.isfinite(y)
        )

        x_valid = x[valid].to_numpy()
        y_valid = y[valid].to_numpy()

        # Linear trendline
        if len(x_valid) >= 2:

            slope, intercept = np.polyfit(
                x_valid,
                y_valid,
                1
            )

            x_line = np.linspace(
                x_valid.min(),
                x_valid.max(),
                100
            )

            y_line = slope * x_line + intercept

            plt.plot(
                x_line,
                y_line,
                linestyle='--',
                linewidth=1.5,
                alpha=0.8
            )

    # --------------------------------------------------------
    # Graph formatting
    # --------------------------------------------------------

    plt.xlabel(
        'Input Image Size (MB)',
        fontsize=11
    )

    plt.ylabel(
        'Speedup (×)',
        fontsize=11
    )

    plt.title(
        title,
        fontsize=13,
        fontweight='bold'
    )

    plt.grid(
        True,
        linestyle=':',
        alpha=0.5
    )

    plt.legend()

    plt.tight_layout()

    # Save
    output_path = os.path.join(
        GRAPH_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Graph generated: {output_path}")



# GRAPH 1
# DOWNSCALING - ALL PARALLEL IMPLEMENTATIONS
generate_graph(
    avg_downscale,
    'Downscaling (0.5×): Speedup Comparison',
    'downscale_overall.png',
    {
        'OpenMP': 'openmp_speedup',
        'MPI': 'mpi_speedup',
        'CUDA': 'cuda_speedup'
    }
)


# GRAPH 2
# UPSCALING - ALL PARALLEL IMPLEMENTATIONS

generate_graph(
    avg_upscale,
    'Upscaling (1.5×): Speedup Comparison',
    'upscale_overall.png',
    {
        'OpenMP': 'openmp_speedup',
        'MPI': 'mpi_speedup',
        'CUDA': 'cuda_speedup'
    }
)



# GRAPH 3
# DOWNSCALING - OPENMP VS MPI
generate_graph(
    avg_downscale,
    'Downscaling (0.5×): OpenMP vs MPI',
    'downscale_openmp_mpi.png',
    {
        'OpenMP': 'openmp_speedup',
        'MPI': 'mpi_speedup'
    }
)



# GRAPH 4
# UPSCALING - OPENMP VS MPI
generate_graph(
    avg_upscale,
    'Upscaling (1.5×): OpenMP vs MPI',
    'upscale_openmp_mpi.png',
    {
        'OpenMP': 'openmp_speedup',
        'MPI': 'mpi_speedup'
    }
)


print("Averaging and graph generation completed!\n")
print("\nGenerated CSV files:")
print(" - averaged_downscale_0.5x.csv")
print(" - averaged_upscale_1.5x.csv")

print("\nGenerated graphs:")
print(" - graphs/downscale_overall.png")
print(" - graphs/upscale_overall.png")
print(" - graphs/downscale_openmp_mpi.png")
print(" - graphs/upscale_openmp_mpi.png")