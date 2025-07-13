#%%
import pandas as pd
import numpy as np

# Read CSV file
csv_file = '/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/anaylisis_code/CL + Animal Trap - ZS & UB (30 days) - filtered.csv'
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
        print(f"  {row['dataset']}: images={int(row['total images'])}, ckpt={int(row['total ckpt'])}, zs={row['zs']:.3f}, ub={row['ub']:.3f}")
    
    # Define thresholds
    min_ckpt = 8  # Fixed minimum checkpoint requirement
    image_thresholds = [3000, 4000, 5000, 6000, 7000, 8000, 9000]
    gap_thresholds = [0.30, 0.25, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03]  # 30%, 25%, 20%, 15%, 10%, 8%, 5%, 3%
    gap_labels = ["30%", "25%", "20%", "15%", "10%", "8%", "5%", "3%"]
    
    print(f"\n{'='*80}")
    print(f"=== Analysis: Checkpoints >= {min_ckpt}, Various Image and UB-ZS Gap Thresholds ===")
    print(f"{'='*80}")
    
    # First filter by checkpoint requirement
    ckpt_filtered = complete_data[complete_data['total ckpt'] >= min_ckpt]
    print(f"\nDatasets with checkpoints >= {min_ckpt}: {len(ckpt_filtered)}")
    
    if len(ckpt_filtered) == 0:
        print("No datasets meet the checkpoint requirement!")
    else:
        # Analyze for each gap threshold
        for gap_idx, gap_threshold in enumerate(gap_thresholds):
            gap_label = gap_labels[gap_idx]
            print(f"\n{'='*60}")
            print(f"=== UB - ZS >= {gap_label} (Checkpoints >= {min_ckpt}) ===")
            print(f"{'='*60}")
            
            # Filter by UB-ZS gap: ub - zs >= gap_threshold
            gap_filtered = ckpt_filtered[(ckpt_filtered['ub'] - ckpt_filtered['zs']) >= gap_threshold]
            
            print(f"\nDatasets with UB-ZS gap >= {gap_label}: {len(gap_filtered)}")
            
            if len(gap_filtered) == 0:
                print(f"No datasets meet UB-ZS gap >= {gap_label} requirement")
                continue
            
            # For each image threshold
            for image_threshold in image_thresholds:
                # Filter by image count
                final_filtered = gap_filtered[gap_filtered['total images'] >= image_threshold]
                
                print(f"\n--- Images >= {image_threshold}, UB-ZS >= {gap_label}, Checkpoints >= {min_ckpt} ---")
                print(f"Qualified datasets: {len(final_filtered)}")
                
                if len(final_filtered) > 0:
                    print("Dataset details:")
                    for idx, row in final_filtered.iterrows():
                        ub_zs_gap = row['ub'] - row['zs']
                        print(f"  {row['dataset']}: img={int(row['total images'])}, ckpt={int(row['total ckpt'])}, zs={row['zs']:.3f}, ub={row['ub']:.3f}, gap={ub_zs_gap:.3f}")
    
    # Summary table
    print(f"\n{'='*80}")
    print(f"=== SUMMARY TABLE ===")
    print(f"{'='*80}")
    print(f"Conditions: Checkpoints >= {min_ckpt}")
    print()
    
    # Create summary table header
    header = f"{'Gap Threshold':<12}"
    for img_thresh in image_thresholds:
        header += f"{'Img>=' + str(img_thresh):<10}"
    print(header)
    print("-" * len(header))
    
    # Fill summary table
    for gap_idx, gap_threshold in enumerate(gap_thresholds):
        gap_label = gap_labels[gap_idx]
        row_str = f"{gap_label:<12}"
        
        # Filter by UB-ZS gap
        gap_filtered = ckpt_filtered[(ckpt_filtered['ub'] - ckpt_filtered['zs']) >= gap_threshold]
        
        for image_threshold in image_thresholds:
            # Filter by image count
            final_filtered = gap_filtered[gap_filtered['total images'] >= image_threshold]
            count = len(final_filtered)
            row_str += f"{count:<10}"
        
        print(row_str)
    
    # Additional statistics
    print(f"\n{'='*80}")
    print(f"=== ADDITIONAL STATISTICS ===")
    print(f"{'='*80}")
    
    print(f"Total datasets in analysis: {len(complete_data)}")
    print(f"Datasets with checkpoints >= {min_ckpt}: {len(ckpt_filtered)}")
    
    if len(ckpt_filtered) > 0:
        print(f"\nFor datasets with checkpoints >= {min_ckpt}:")
        print(f"  Average images: {ckpt_filtered['total images'].mean():.0f}")
        print(f"  Average checkpoints: {ckpt_filtered['total ckpt'].mean():.1f}")
        print(f"  Average ZS: {ckpt_filtered['zs'].mean():.3f}")
        print(f"  Average UB: {ckpt_filtered['ub'].mean():.3f}")
        print(f"  Average UB-ZS gap: {(ckpt_filtered['ub'] - ckpt_filtered['zs']).mean():.3f}")
        
        print(f"\nImage count distribution (checkpoints >= {min_ckpt}):")
        for img_thresh in image_thresholds:
            img_filtered = ckpt_filtered[ckpt_filtered['total images'] >= img_thresh]
            print(f"  Images >= {img_thresh}: {len(img_filtered)} datasets")
        
        print(f"\nUB-ZS gap distribution (checkpoints >= {min_ckpt}):")
        for gap_idx, gap_threshold in enumerate(gap_thresholds):
            gap_label = gap_labels[gap_idx]
            gap_filtered = ckpt_filtered[(ckpt_filtered['ub'] - ckpt_filtered['zs']) >= gap_threshold]
            print(f"  UB-ZS gap >= {gap_label}: {len(gap_filtered)} datasets")

# %%
