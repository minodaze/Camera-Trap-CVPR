import json
import pandas as pd
import os
import shutil
from pathlib import Path

def get_dataset_mapping():
    """
    Get mapping from input dataset names to CSV dataset names
    """
    dataset_mapping = {
        'APN_K024': 'Snapshot Safari 2024 Expansion',
        'ENO_B06': 'Snapshot Safari 2024 Expansion', 
        'ENO_C05': 'Snapshot Safari 2024 Expansion',
        'ENO_E06': 'Snapshot Safari 2024 Expansion',
        'MAD_05': 'Snapshot Safari 2024 Expansion',
        'MAD_MAD05': 'Snapshot Safari 2024 Expansion',
        'serengeti_E03': 'Snapshot Serengeti',
        'serengeti_L10': 'Snapshot Serengeti',
        'serengeti_C02': 'Snapshot Serengeti',
        'na_lebec_CA-22': 'NACTI',
        'na_lebec_CA-14': 'NACTI',
        'na_lebec_CA-24': 'NACTI',
        'nz_EFH_HCAMF02': 'Trail Camera Images of New Zealand Animals',
        'nz_EFH_HCAMF09': 'Trail Camera Images of New Zealand Animals',
        'nz_EFH_HCAMG12': 'Trail Camera Images of New Zealand Animals'
    }
    return dataset_mapping

def extract_dataset_name_from_path(path):
    """
    Extract dataset name from file path
    """
    path_parts = path.split('/')
    for part in path_parts:
        if any(dataset in part for dataset in ['APN_', 'ENO_', 'MAD_', 'serengeti_', 'na_lebec', 'nz_EFH']):
            return part
    return None

def convert_scientific_to_common_names():
    """
    Convert scientific names to common names based on the CSV mapping file
    """
    # Input and output directory paths
    input_dir = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/na/na_lebec_CA-22/"
    output_dir = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/na/na_lebec_CA-22_common_name/"
    
    # Extract dataset name and get CSV dataset mapping
    dataset_name = extract_dataset_name_from_path(input_dir)
    dataset_mapping = get_dataset_mapping()
    
    if dataset_name not in dataset_mapping:
        print(f"❌ Dataset '{dataset_name}' not found in mapping!")
        return
    
    csv_dataset_name = dataset_mapping[dataset_name]
    print(f"📊 Input dataset: '{dataset_name}' -> CSV dataset: '{csv_dataset_name}'")
    
    # Read the taxonomy mapping CSV file
    csv_path = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/scripts/051625/lila-taxonomy-mapping_release.csv"
    df = pd.read_csv(csv_path)
    
    # Filter by target dataset and create mapping dictionary from scientific name to common name
    df_filtered = df[df['dataset_name'] == csv_dataset_name]
    print(f"Found {len(df_filtered)} entries for dataset '{csv_dataset_name}'")
    
    name_mapping = {}
    for _, row in df_filtered.iterrows():
        scientific_name = row['scientific_name']
        common_name = row['common_name']
        # Skip if either scientific_name or common_name is NaN
        if pd.notna(scientific_name) and pd.notna(common_name):
            name_mapping[scientific_name] = common_name
    
    print(f"Created {len(name_mapping)} scientific -> common name mappings")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy image folder from input to output directory
    copy_image_folder(input_dir, output_dir)
    
    # List of JSON files to process
    json_files = ["train.json", "train-all.json", "test.json"]
    
    # Track names across all files
    all_unconverted_names = set()
    all_original_names = set()
    all_converted_names = set()
    
    for filename in json_files:
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Warning: File {input_file} does not exist, skipping")
            continue
        
        # Read JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Track names for this file
        unconverted_names = set()
        original_names = set()
        converted_names = set()
        
        # Convert names in the data
        converted_data = convert_names_in_data(data, name_mapping, unconverted_names, original_names, converted_names)
        
        # Write converted file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Completed conversion for {filename}, saved to {output_file}")
        
        # Add to global sets
        all_unconverted_names.update(unconverted_names)
        all_original_names.update(original_names)
        all_converted_names.update(converted_names)
    
    # Report conversion summary
    print(f"\n=== CONVERSION SUMMARY ===")
    print(f"1. Original class names ({len(all_original_names)}):")
    for name in sorted(all_original_names):
        print(f"   - {name}")
    
    print(f"\n2. Converted class names ({len(all_converted_names)}):")
    for name in sorted(all_converted_names):
        print(f"   - {name}")
    
    if all_unconverted_names:
        print(f"\nTotal unconverted class names: {len(all_unconverted_names)}")
        print("Unconverted class names:")
        for name in sorted(all_unconverted_names):
            print(f"  - {name}")
    else:
        print("\nAll class names were successfully converted!")

def copy_image_folder(input_dir, output_dir):
    """
    Copy the image folder from input directory to output directory
    """
    input_image_dir = os.path.join(input_dir, "images")  # Try "images" first
    if not os.path.exists(input_image_dir):
        input_image_dir = os.path.join(input_dir, "image")  # Try "image" as fallback
    
    output_image_dir = os.path.join(output_dir, "images")  # Always use "images" in output
    
    try:
        if os.path.exists(input_image_dir):
            # Remove existing image folder in output if it exists
            if os.path.exists(output_image_dir):
                print(f"Removing existing image folder: {output_image_dir}")
                shutil.rmtree(output_image_dir)
            
            # Copy the image folder
            print(f"Copying image folder from {input_image_dir} to {output_image_dir}")
            shutil.copytree(input_image_dir, output_image_dir)
            
            # Get folder size information
            total_files = sum(len(files) for _, _, files in os.walk(output_image_dir))
            total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                           for dirpath, _, filenames in os.walk(output_image_dir)
                           for filename in filenames)
            
            print(f"Successfully copied image folder:")
            print(f"  - Total files: {total_files}")
            print(f"  - Total size: {total_size / (1024*1024):.2f} MB")
        else:
            print(f"Warning: Image folder not found at {input_image_dir}")
            print("Continuing without copying image folder...")
    
    except Exception as e:
        print(f"Error copying image folder: {e}")
        print("Continuing with JSON file processing...")

def convert_names_in_data(data, name_mapping, unconverted_names, original_names, converted_names):
    """
    Recursively convert scientific names to common names in the data structure
    """
    if isinstance(data, dict):
        converted_data = {}
        for key, value in data.items():
            # If key is a scientific name, convert to common name
            if key in name_mapping:
                new_key = name_mapping[key]
                original_names.add(key)
                converted_names.add(new_key)
                converted_data[new_key] = convert_names_in_data(value, name_mapping, unconverted_names, original_names, converted_names)
            else:
                # Check if this looks like a class name (not a checkpoint key like "ckp_")
                if not key.startswith("ckp_"):
                    original_names.add(key)
                    unconverted_names.add(key)
                converted_data[key] = convert_names_in_data(value, name_mapping, unconverted_names, original_names, converted_names)
        return converted_data
    
    elif isinstance(data, list):
        return [convert_names_in_data(item, name_mapping, unconverted_names, original_names, converted_names) for item in data]
    
    elif isinstance(data, str):
        # If string itself is a scientific name, convert to common name
        if data in name_mapping:
            original_names.add(data)
            converted_names.add(name_mapping[data])
            return name_mapping[data]
        else:
            # Add to unconverted if it's not a sequence ID or other non-class identifier
            if not (data.startswith("ckp_") or len(data) > 15):  # seq_ids are typically long
                original_names.add(data)
                unconverted_names.add(data)
            return data
    
    else:
        return data

def main():
    """
    Main function to execute the conversion process
    """
    try:
        convert_scientific_to_common_names()
        print("Conversion completed successfully!")
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("Please check if the CSV file path is correct")
    except Exception as e:
        print(f"Error occurred during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
