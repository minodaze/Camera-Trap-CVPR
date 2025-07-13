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
        'Serengeti_E03': 'Snapshot Serengeti',
        'Serengeti_L10': 'Snapshot Serengeti',
        'Serengeti_C02': 'Snapshot Serengeti',
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
        if any(dataset in part for dataset in ['APN_', 'ENO_', 'MAD_', 'Serengeti_', 'na_lebec', 'nz_EFH']):
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
    
    # Track unconverted names across all files
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
        
        # Track unconverted names for this file
        unconverted_names = set()
        original_names = set()
        converted_names = set()
        
        # Convert names in the data
        converted_data = convert_names_in_data(data, name_mapping, unconverted_names, original_names, converted_names)
        
        # Write converted file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Completed conversion for {filename}, saved to {output_file}")
        
        # Report unconverted names for this file
        if unconverted_names:
            print(f"  Unconverted class names in {filename}: {sorted(unconverted_names)}")
        else:
            print(f"  All class names in {filename} were successfully converted")
        
        # Add to global sets
        all_unconverted_names.update(unconverted_names)
        all_original_names.update(original_names)
        all_converted_names.update(converted_names)
    
    # Report all unconverted names
    print(f"\n=== CONVERSION SUMMARY ===")
    print(f"1. Original class names ({len(all_original_names)}):")
    for name in sorted(all_original_names):
        print(f"   - {name}")
    
    print(f"\n2. Converted class names ({len(all_converted_names)}):")
    for name in sorted(all_converted_names):
        print(f"   - {name}")
    
    if all_unconverted_names:
        print(f"\nTotal unconverted class names across all files: {len(all_unconverted_names)}")
        print("Unconverted class names:")
        for name in sorted(all_unconverted_names):
            print(f"  - {name}")
        
        # Save unconverted names to a file for reference
        unconverted_file = os.path.join(output_dir, "unconverted_class_names.txt")
        with open(unconverted_file, 'w', encoding='utf-8') as f:
            f.write("Unconverted class names:\n")
            for name in sorted(all_unconverted_names):
                f.write(f"{name}\n")
        print(f"Unconverted names saved to: {unconverted_file}")
    else:
        print("All class names were successfully converted!")
    
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
        
        # Track unconverted names for this file
        unconverted_names = set()
        
        # Convert names in the data
        converted_data = convert_names_in_data(data, name_mapping, unconverted_names)
        
        # Write converted file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Completed conversion for {filename}, saved to {output_file}")
        
        # Report unconverted names for this file
        if unconverted_names:
            print(f"  Unconverted class names in {filename}: {sorted(unconverted_names)}")
        else:
            print(f"  All class names in {filename} were successfully converted")
        
        # Add to global unconverted set
        all_unconverted_names.update(unconverted_names)
    
    # Report all unconverted names
    print(f"\n=== SUMMARY ===")
    if all_unconverted_names:
        print(f"Total unconverted class names across all files: {len(all_unconverted_names)}")
        print("Unconverted class names:")
        for name in sorted(all_unconverted_names):
            print(f"  - {name}")
        
        # Save unconverted names to a file for reference
        unconverted_file = os.path.join(output_dir, "unconverted_class_names.txt")
        with open(unconverted_file, 'w', encoding='utf-8') as f:
            f.write("Unconverted class names:\n")
            for name in sorted(all_unconverted_names):
                f.write(f"{name}\n")
        print(f"Unconverted names saved to: {unconverted_file}")
    else:
        print("All class names were successfully converted!")

def copy_image_folder(input_dir, output_dir):
    """
    Copy the image folder from input directory to output directory
    
    Args:
        input_dir: Source directory containing the image folder
        output_dir: Destination directory where image folder will be copied
    """
    input_image_dir = os.path.join(input_dir, "image")
    output_image_dir = os.path.join(output_dir, "image")
    
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

def convert_names_in_data(data, name_mapping, unconverted_names):
    """
    Recursively convert scientific names to common names in the data structure
    
    Args:
        data: The data structure to convert (dict, list, or string)
        name_mapping: Dictionary mapping scientific names to common names
        unconverted_names: Set to track names that couldn't be converted
        
    Returns:
        Converted data structure with common names
    """
    if isinstance(data, dict):
        converted_data = {}
        for key, value in data.items():
            # If key is a scientific name, convert to common name
            if key in name_mapping:
                new_key = name_mapping[key]
                converted_data[new_key] = convert_names_in_data(value, name_mapping, unconverted_names)
            else:
                # Check if this looks like a class name (not a checkpoint key like "ckp_1")
                if not key.startswith("ckp_"):
                    unconverted_names.add(key)
                converted_data[key] = convert_names_in_data(value, name_mapping, unconverted_names)
        return converted_data
    
    elif isinstance(data, list):
        return [convert_names_in_data(item, name_mapping, unconverted_names) for item in data]
    
    elif isinstance(data, str):
        # If string itself is a scientific name, convert to common name
        if data in name_mapping:
            return name_mapping[data]
        else:
            # Add to unconverted if it's not a sequence ID or other non-class identifier
            if not (data.startswith("ckp_") or len(data) > 15):  # seq_ids are typically long
                unconverted_names.add(data)
            return data
    
    else:
        return data

def print_mapping_stats(name_mapping):
    """
    Print statistics about the name mapping
    """
    print(f"Total mappings available: {len(name_mapping)}")
    print("Sample mappings:")
    for i, (sci_name, common_name) in enumerate(list(name_mapping.items())[:5]):
        print(f"  {sci_name} -> {common_name}")
    print()

def extract_class_names_from_json(filepath):
    """
    Extract all potential class names from a JSON file
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Set of potential class names found in the file
    """
    class_names = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def extract_names(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Skip checkpoint keys
                    if not key.startswith("ckp_"):
                        class_names.add(key)
                    extract_names(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_names(item)
            elif isinstance(obj, str):
                # Skip sequence IDs (typically long strings)
                if not (obj.startswith("ckp_") or len(obj) > 15):
                    class_names.add(obj)
        
        extract_names(data)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return class_names

def main():
    """
    Main function to execute the conversion process
    """
    try:
        # Read the taxonomy mapping CSV file first to check available mappings
        csv_path = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/scripts/051625/lila-taxonomy-mapping_release.csv"
        df = pd.read_csv(csv_path)
        
        # Create mapping dictionary
        name_mapping = {}
        for _, row in df.iterrows():
            scientific_name = row['scientific_name']
            common_name = row['common_name']
            # Skip if either scientific_name or common_name is NaN
            if pd.notna(scientific_name) and pd.notna(common_name):
                name_mapping[scientific_name] = common_name
        
        print_mapping_stats(name_mapping)
        
        # First, extract all class names from JSON files to see what we're working with
        input_dir = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/na/na_lebec_CA-22/"
        json_files = ["train.json", "train-all.json", "test.json"]
        
        print("=== CLASS NAMES FOUND IN JSON FILES ===")
        all_found_names = set()
        for filename in json_files:
            filepath = os.path.join(input_dir, filename)
            if os.path.exists(filepath):
                found_names = extract_class_names_from_json(filepath)
                all_found_names.update(found_names)
                print(f"{filename}: {sorted(found_names)}")
        
        print(f"\nAll unique class names found: {sorted(all_found_names)}")
        print(f"Total unique class names: {len(all_found_names)}")
        print()
        
        # Execute the conversion
        convert_scientific_to_common_names()
        print("All files conversion completed!")
        
        # Verify conversion results
        print("\n=== VERIFICATION ===")
        output_dir = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/na/na_lebec_CA-22_common_name/"
        for filename in ["train.json", "train-all.json", "test.json"]:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"{filename}: Contains {len(data)} keys")
                # Print first few keys as examples
                keys = list(data.keys())[:3]
                print(f"  Sample keys: {keys}")
            else:
                print(f"{filename}: File does not exist")
                
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("Please check if the CSV file path is correct")
    except Exception as e:
        print(f"Error occurred during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()