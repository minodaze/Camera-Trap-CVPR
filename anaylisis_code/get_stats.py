#%%
import pandas as pd
import numpy as np

# Read CSV file
csv_file = '/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/anaylisis_code/CL + Animal Trap - ZS & UB (30 days).csv'
df = pd.read_csv(csv_file)

print("Original data shape:", df.shape)
print("\nOriginal data columns:", df.columns.tolist())

# Check missing values
print("\nMissing values statistics:")
print(df.isnull().sum())

# Keep only data points with all values for dataset, total images, total ckpt, zs, ub
# Remove rows where zs or ub is empty
complete_data = df.dropna(subset=['total images', 'total ckpt', 'zs', 'ub'])

print(f"\nComplete data points: {len(complete_data)} / {len(df)}")

if len(complete_data) == 0:
    print("No complete data points found!")
else:
    print("\nDatasets with complete data:")
    for idx, row in complete_data.iterrows():
        print(f"  {row['dataset']}: images={row['total images']}, ckpt={row['total ckpt']}, zs={row['zs']:.2f}, ub={row['ub']:.2f}")
    
    # Calculate statistics
    def calculate_stats(data, column_name):
        print(f"\n=== {column_name} Statistics ===")
        print(f"Mean: {data.mean():.4f}")
        print(f"Median: {data.median():.4f}")
        print(f"25th percentile (Q1): {data.quantile(0.25):.4f}")
        print(f"75th percentile (Q3): {data.quantile(0.75):.4f}")
        print(f"Minimum: {data.min():.4f}")
        print(f"Maximum: {data.max():.4f}")
        print(f"Standard deviation: {data.std():.4f}")
        print(f"Number of data points: {len(data)}")
    
    # Calculate statistics for each column
    calculate_stats(complete_data['total images'], 'Total Images')
    calculate_stats(complete_data['total ckpt'], 'Total Checkpoints')
    calculate_stats(complete_data['zs'], 'Zero-Shot Accuracy')
    calculate_stats(complete_data['ub'], 'Upper Bound Accuracy')
    
    # Additional analysis: UB and ZS difference
    ub_zs_diff = complete_data['ub'] - complete_data['zs']
    calculate_stats(ub_zs_diff, 'UB - ZS Difference')
    
    # Display data distribution overview
    print(f"\n=== Data Distribution Overview ===")
    print(f"Total Images range: {complete_data['total images'].min()} - {complete_data['total images'].max()}")
    print(f"Total Checkpoints range: {complete_data['total ckpt'].min()} - {complete_data['total ckpt'].max()}")
    print(f"Zero-Shot Accuracy range: {complete_data['zs'].min():.3f} - {complete_data['zs'].max():.3f}")
    print(f"Upper Bound Accuracy range: {complete_data['ub'].min():.3f} - {complete_data['ub'].max():.3f}")
    
    # Check for outliers (using IQR method)
    def detect_outliers(data, column_name):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        if len(outliers) > 0:
            print(f"\n{column_name} outliers: {outliers.tolist()}")
        else:
            print(f"\n{column_name} no outliers")
    
    print(f"\n=== Outlier Detection (IQR method) ===")
    detect_outliers(complete_data['total images'], 'Total Images')
    detect_outliers(complete_data['zs'], 'Zero-Shot Accuracy')
    detect_outliers(complete_data['ub'], 'Upper Bound Accuracy')

    # New logic: Calculate datasets meeting specific conditions
    print(f"\n=== Dataset Statistics Meeting Conditions ===")
    print("Conditions: 1) Images >= threshold, 2) ZS < UB - X%, 3) Checkpoints >= threshold")
    
    # Define image thresholds, checkpoint thresholds, and performance gap thresholds
    image_thresholds = [5000, 6000, 7000, 8000, 9000, 10000]
    ckpt_thresholds = [5, 8, 10]
    gap_thresholds = [0.03, 0.05, 0.08, 0.10]  # 3%, 5%, 8%, 10%
    gap_labels = ["3%", "5%", "8%", "10%"]
    
    for gap_idx, gap_threshold in enumerate(gap_thresholds):
        gap_label = gap_labels[gap_idx]
        print(f"\n{'='*60}")
        print(f"=== ZS < UB - {gap_label} ===")
        print(f"{'='*60}")
        
        for ckpt_threshold in ckpt_thresholds:
            print(f"\n--- Checkpoints >= {ckpt_threshold} ---")
            for image_threshold in image_thresholds:
                # Condition 1: number of images >= threshold
                condition1 = complete_data['total images'] >= image_threshold
                
                # Condition 2: zs lower than ub by X%, i.e., zs < ub - gap_threshold
                condition2 = complete_data['zs'] < (complete_data['ub'] - gap_threshold)
                
                # Condition 3: total ckpt >= ckpt_threshold
                condition3 = complete_data['total ckpt'] >= ckpt_threshold
                
                # Datasets meeting all three conditions
                qualified_datasets = complete_data[condition1 & condition2 & condition3]
                
                print(f"Images>={image_threshold}, ZS<UB-{gap_label}, Ckpts>={ckpt_threshold}: {len(qualified_datasets)} datasets")
                
                if len(qualified_datasets) > 0:
                    for idx, row in qualified_datasets.iterrows():
                        zs_ub_diff = row['ub'] - row['zs']
                        print(f"  {row['dataset']}: img={int(row['total images'])}, ckpt={int(row['total ckpt'])}, zs={row['zs']:.3f}, ub={row['ub']:.3f}, diff={zs_ub_diff:.3f}")
    
    # Additional statistics: Summary of different performance gap thresholds
    print(f"\n{'='*60}")
    print(f"=== Summary Statistics for Different Performance Gap Thresholds ===")
    print(f"{'='*60}")
    
    for gap_idx, gap_threshold in enumerate(gap_thresholds):
        gap_label = gap_labels[gap_idx]
        datasets_with_large_gap = complete_data[complete_data['zs'] < (complete_data['ub'] - gap_threshold)]
        
        print(f"\nZS < UB - {gap_label}:")
        print(f"  Total datasets: {len(datasets_with_large_gap)}")
        if len(datasets_with_large_gap) > 0:
            print(f"  Average Images: {datasets_with_large_gap['total images'].mean():.0f}")
            print(f"  Average Checkpoints: {datasets_with_large_gap['total ckpt'].mean():.1f}")
            print(f"  Average ZS: {datasets_with_large_gap['zs'].mean():.3f}")
            print(f"  Average UB: {datasets_with_large_gap['ub'].mean():.3f}")
            print(f"  Average gap: {(datasets_with_large_gap['ub'] - datasets_with_large_gap['zs']).mean():.3f}")
        
        # Distribution under different checkpoint thresholds
        for ckpt_threshold in ckpt_thresholds:
            qualified_with_ckpt = datasets_with_large_gap[datasets_with_large_gap['total ckpt'] >= ckpt_threshold]
            print(f"    Among which Ckpts>={ckpt_threshold}: {len(qualified_with_ckpt)} datasets")
    
    print(f"\nOverall average ZS-UB gap: {(complete_data['ub'] - complete_data['zs']).mean():.3f}")

# %%
