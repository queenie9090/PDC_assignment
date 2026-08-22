import pandas as pd

# Define the file paths for each member based on your naming scheme
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

# Metrics to average across members
metric_cols = [
    'baseline_ms', 
    'openmp_ms', 'openmp_speedup', 
    'cuda_ms', 'cuda_speedup', 
    'mpi_ms', 'mpi_speedup'
]

# Structural columns to group by and retain in output
meta_cols = ['Image', 'Input_Image_Size_MB', 'Original_Resolution', 'Output_Resolution']

def process_averages(file_list):
    # Load all 3 member CSVs into a list
    dfs = [pd.read_csv(f) for f in file_list]
    
    # Combine into a single DataFrame
    combined = pd.concat(dfs, ignore_index=True)
    
    # Calculate average across the 3 members grouped by image details
    averaged_df = combined.groupby(meta_cols, as_index=False)[metric_cols].mean()
    
    return averaged_df

# Generate averaged DataFrames
avg_downscale = process_averages(files['downscale'])
avg_upscale = process_averages(files['upscale'])

# Save results
avg_downscale.to_csv('averaged_downscale_0.5x.csv', index=False)
avg_upscale.to_csv('averaged_upscale_1.5x.csv', index=False)

print("Calculation complete! Saved as 'averaged_downscale_0.5x.csv' and 'averaged_upscale_1.5x.csv'.")