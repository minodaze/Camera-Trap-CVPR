import json
import csv
import os
from pathlib import Path

def load_oracle_results():
    """
    Load oracle results from the best_oracle_logs folder and update the CSV file.
    """
    # Path to the CSV file
    csv_file_path = "other/CL + Animal Trap - Oracle _= ZS.csv"
    base_oracle_path = "/fs/scratch/PAS2099/camera-trap-final/best_accum_logs"
    
    # Read the existing CSV
    rows = []
    header = None
    
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Get the header
        for row in reader:
            rows.append(row)
    
    print(f"Loaded {len(rows)} rows from CSV file")
    print(f"Header: {header}")
    
    # Find the column indices
    dataset_col = 0  # First column is dataset
    oracle_col = header.index('best accum.') if 'best accum.' in header else 9  # Default to column 5

    print(f"Dataset column: {dataset_col}, accum column: {oracle_col}")
    
    # Dictionary to store oracle results
    oracle_results = {}
    # /fs/scratch/PAS2099/camera-trap-final/best_accum_logs/APN_APN_K051/accum_lora_bsm_loss/bioclip2/lora_8_text_head/all/log/final_training_summary.json
    # Process each dataset
    updated_count = 0
    missing_files = []
    
    for i, row in enumerate(rows):
        if len(row) <= dataset_col:
            continue
            
        dataset_name = row[dataset_col].strip()
        if not dataset_name:
            continue
            
        # Convert dataset name to the required format (replace / with _)
        dataset_underscore = dataset_name.replace("/", "_")
        
        # Construct the JSON file path
        json_path = os.path.join(
            base_oracle_path,
            dataset_underscore,
            "accum_lora_bsm_loss",
            "bioclip2",
            "lora_8_text_head",
            "all",
            "log",
            "final_training_summary.json"
        )
        
        # Try to load the JSON file
        # import pdb; pdb.set_trace()
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Extract the balanced accuracy from averages
                balanced_accuracy = data.get('averages', {}).get('balanced_accuracy', None)
                
                if balanced_accuracy is not None:
                    # Update the oracle column in the row
                    while len(row) <= oracle_col:
                        row.append('')  # Extend row if necessary
                    
                    row[oracle_col] = f"{balanced_accuracy:.4f}"
                    oracle_results[dataset_name] = balanced_accuracy
                    updated_count += 1
                    print(f"✅ Updated {dataset_name}: {balanced_accuracy:.4f}")
                else:
                    print(f"⚠️  No balanced_accuracy found in averages for {dataset_name}")
                    missing_files.append(f"{dataset_name} (no balanced_accuracy in averages)")
            else:
                print(f"❌ File not found: {json_path}")
                missing_files.append(f"{dataset_name} (file not found)")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error for {dataset_name}: {e}")
            missing_files.append(f"{dataset_name} (JSON decode error)")
        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
            missing_files.append(f"{dataset_name} (error: {str(e)})")
    
    # Write the updated CSV
    output_file = "other/CL + Animal Trap - Oracle _= ZS_updated.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Write header
        writer.writerows(rows)   # Write updated rows
    
    print(f"\n=== Summary ===")
    print(f"✅ Successfully updated: {updated_count} datasets")
    print(f"❌ Missing/failed: {len(missing_files)} datasets")
    print(f"📁 Updated CSV saved as: {output_file}")
    
    if missing_files:
        print(f"\n📋 Missing/failed datasets:")
        for missing in missing_files[:10]:  # Show first 10
            print(f"  - {missing}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    # Show some statistics
    if oracle_results:
        values = list(oracle_results.values())
        print(f"\n📊 Oracle Results Statistics:")
        print(f"  Average balanced accuracy: {sum(values)/len(values):.4f}")
        print(f"  Best performance: {max(values):.4f}")
        print(f"  Worst performance: {min(values):.4f}")
        
        # Show top 5 performers
        top_performers = sorted(oracle_results.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n🏆 Top 5 Oracle Performers:")
        for i, (dataset, score) in enumerate(top_performers, 1):
            print(f"  {i}. {dataset}: {score:.4f}")

def verify_oracle_results():
    """
    Verify that oracle results are correctly loaded by sampling a few datasets.
    """
    # Sample datasets to verify
    sample_datasets = [
        "APN/APN_6U",
        "serengeti/serengeti_F05", 
        "MAD/MAD_MAD01"
    ]
    
    base_oracle_path = "/fs/scratch/PAS2099/camera-trap-final/best_oracle_logs"
    
    print("🔍 Verifying oracle results for sample datasets:")
    
    for dataset in sample_datasets:
        dataset_underscore = dataset.replace("/", "_")
        json_path = os.path.join(
            base_oracle_path,
            dataset_underscore,
            "oracle_lora_bsm_loss",
            "bioclip2", 
            "lora_8_text_head",
            "log",
            "final_training_summary.json"
        )
        
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                balanced_accuracy = data.get('averages', {}).get('balanced_accuracy')
                total_checkpoints = data.get('total_checkpoints', 'N/A')
                total_samples = data.get('total_samples', 'N/A')
                
                print(f"  ✅ {dataset}:")
                print(f"     Balanced Accuracy: {balanced_accuracy:.4f}")
                print(f"     Total Checkpoints: {total_checkpoints}")
                print(f"     Total Samples: {total_samples}")
            else:
                print(f"  ❌ {dataset}: File not found")
                
        except Exception as e:
            print(f"  ❌ {dataset}: Error - {e}")

if __name__ == "__main__":
    print("🚀 Loading Oracle Results and Updating CSV...")
    
    # First verify some sample results
    verify_oracle_results()
    
    print("\n" + "="*50)
    
    # Load and update all results
    load_oracle_results()
    
    print("\n✅ Script completed!")