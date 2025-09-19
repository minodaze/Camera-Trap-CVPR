import pandas as pd
import json
import os

# Read the CSV file
csv_path = 'other/CL + Animal Trap - Oracle _= ZS.csv'
df = pd.read_csv(csv_path)

# Add interpolate column if it doesn't exist
if 'interpolate' not in df.columns:
    df['interpolate'] = None

# Process each dataset
for idx, row in df.iterrows():
    dataset = row['dataset']
    ds = dataset.replace('/', '_')
    
    # Construct the JSON path
    json_path = f"/fs/scratch/PAS2099/camera-trap-final/round2_eval_logs/{ds}/eval_full_lora_accum_interpolation/bioclip2/lora_8_text_head_merge_factor_0.8/log/eval_only_summary.json"
    
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract the average balanced accuracy
            if 'average' in data and 'balanced_accuracy' in data['average']:
                interpolation_value = round(data['average']['balanced_accuracy'], 2)
                df.at[idx, 'interpolate'] = interpolation_value
                print(f"✓ {dataset}: {interpolation_value}")
            else:
                print(f"✗ {dataset}: No average balanced_accuracy found in JSON")
        else:
            print(f"✗ {dataset}: JSON file not found - {json_path}")
    
    except Exception as e:
        print(f"✗ {dataset}: Error reading JSON - {e}")

# Save the updated CSV
output_path = '/users/PAS2099/mino/ICICLE/other/CL + Animal Trap - Oracle _= ZS_with_interpolation.csv'
df.to_csv(output_path, index=False)

print(f"\n📊 Summary:")
print(f"Total datasets: {len(df)}")
print(f"Successfully filled: {df['interpolate'].notna().sum()}")
print(f"Missing data: {df['interpolate'].isna().sum()}")
print(f"\n💾 Updated CSV saved to: {output_path}")

# Show a preview of datasets with interpolate data
filled_data = df[df['interpolate'].notna()]
if not filled_data.empty:
    print(f"\n📋 Preview of filled data:")
    print(filled_data[['dataset', 'zs', 'oracle', 'interpolate']].head(10))
