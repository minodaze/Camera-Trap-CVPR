import pandas as pd
import json
import os
import shutil

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

def convert_json_class_names():
    # Define source and target directories
    source_dir = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/na/na_lebec_CA-24"
    target_dir = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/na/na_lebec_CA-24_scientific_name"

    # Extract dataset name and get CSV dataset mapping
    dataset_name = extract_dataset_name_from_path(source_dir)
    dataset_mapping = get_dataset_mapping()
    
    if dataset_name not in dataset_mapping:
        print(f"❌ Dataset '{dataset_name}' not found in mapping!")
        return
    
    csv_dataset_name = dataset_mapping[dataset_name]
    print(f"📊 Input dataset: '{dataset_name}' -> CSV dataset: '{csv_dataset_name}'")
    
    # Load the CSV mapping file and filter by dataset
    csv_file = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/scripts/051625/lila-taxonomy-mapping_release.csv"
    df = pd.read_csv(csv_file)
    df_filtered = df[df['dataset_name'] == csv_dataset_name]
    
    print(f"Found {len(df_filtered)} entries for dataset '{csv_dataset_name}'")
    
    # Create mapping from query to scientific_name
    mapping = {}
    for _, row in df_filtered.iterrows():
        query = row['query']
        scientific_name = row['scientific_name']
        
        # Skip if either query or scientific_name is NaN/None
        if pd.notna(query) and pd.notna(scientific_name):
            mapping[query] = scientific_name
    
    print(f"Loaded {len(mapping)} query -> scientific_name mappings")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # JSON files to convert
    json_files = ["train.json", "train-all.json", "test.json"]
    
    all_unconverted = set()
    all_original_names = set()
    all_converted_names = set()
    total_converted = 0
    total_entries = 0
    
    for json_file in json_files:
        source_path = os.path.join(source_dir, json_file)
        target_path = os.path.join(target_dir, json_file)
        
        if not os.path.exists(source_path):
            print(f"Warning: {source_path} not found, skipping")
            continue
            
        print(f"\nProcessing {json_file}...")
        
        # Load JSON data
        with open(source_path, 'r') as f:
            data = json.load(f)
        
        # Convert class names
        converted_data = {}
        file_entries = 0
        converted_count = 0
        unconverted_classes = set()
        class_conversions = {}  # Track conversions: original -> scientific_name
        
        for checkpoint_key, entries in data.items():
            converted_data[checkpoint_key] = []
            
            for entry in entries:
                file_entries += 1
                
                # Check if entry has class_name field
                if 'class_name' in entry:
                    original_class = entry['class_name']
                    all_original_names.add(original_class)
                    
                    if original_class in mapping:
                        scientific_name = mapping[original_class]
                        entry['class_name'] = scientific_name
                        converted_count += 1
                        all_converted_names.add(scientific_name)
                        
                        # Track the conversion
                        if original_class != scientific_name:  # Only track if actually different
                            class_conversions[original_class] = scientific_name
                    else:
                        unconverted_classes.add(original_class)
                
                converted_data[checkpoint_key].append(entry)
        
        # Save converted JSON
        with open(target_path, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        print(f"  Total entries: {file_entries}")
        print(f"  Converted: {converted_count}")
        print(f"  Unconverted: {len(unconverted_classes)}")
        
        # Print class conversions
        if class_conversions:
            print(f"  Class conversions:")
            for original, scientific in sorted(class_conversions.items()):
                print(f"    {original} -> {scientific}")
        
        if unconverted_classes:
            print(f"  Unconverted class names: {sorted(unconverted_classes)}")
            all_unconverted.update(unconverted_classes)
        
        total_entries += file_entries
        total_converted += converted_count
        all_unconverted.update(unconverted_classes)
    
    # Summary
    print(f"\n=== CONVERSION SUMMARY ===")
    print(f"1. Original class names (query) ({len(all_original_names)}):")
    for name in sorted(all_original_names):
        print(f"   - {name}")
    
    print(f"\n2. Converted class names (scientific_name) ({len(all_converted_names)}):")
    for name in sorted(all_converted_names):
        print(f"   - {name}")
    
    print(f"\nTotal entries processed: {total_entries}")
    print(f"Total converted: {total_converted}")
    print(f"Total unconverted: {total_entries - total_converted}")
    print(f"Conversion rate: {total_converted/total_entries*100:.1f}%")
    
    if all_unconverted:
        print(f"\nAll unconverted class names across files:")
        for class_name in sorted(all_unconverted):
            print(f"  - {class_name}")
    
    print(f"\nConversion complete! Files saved to: {target_dir}")

if __name__ == "__main__":
    convert_json_class_names()
