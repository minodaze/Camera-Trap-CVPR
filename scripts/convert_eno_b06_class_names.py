import pandas as pd
import json
import os
import shutil

def convert_json_class_names():
    # Load the CSV mapping file
    csv_file = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/scripts/051625/lila-taxonomy-mapping_release.csv"
    df = pd.read_csv(csv_file)
    
    # Create mapping from query to common_name
    # Handle missing values by filtering them out
    mapping = {}
    for _, row in df.iterrows():
        query = row['query']
        common_name = row['common_name']
        
        # Skip if either query or common_name is NaN/None
        if pd.notna(query) and pd.notna(common_name):
            mapping[query] = common_name
    
    print(f"Loaded {len(mapping)} query -> common_name mappings")
    
    # Define source and target directories
    source_dir = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/ENO/ENO_B06"
    target_dir = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/ENO/ENO_B06_common_name"
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # JSON files to convert
    json_files = ["train.json", "train-all.json", "test.json"]
    
    all_unconverted = set()
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
        
        # Track conversion statistics
        file_entries = len(data)
        converted_count = 0
        unconverted_classes = set()
        
        # Convert class names
        for entry in data:
            original_class = entry['class_name']
            
            if original_class in mapping:
                entry['class_name'] = mapping[original_class]
                converted_count += 1
            else:
                unconverted_classes.add(original_class)
        
        # Save converted JSON
        with open(target_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Total entries: {file_entries}")
        print(f"  Converted: {converted_count}")
        print(f"  Unconverted: {len(unconverted_classes)}")
        
        if unconverted_classes:
            print(f"  Unconverted class names: {sorted(unconverted_classes)}")
            all_unconverted.update(unconverted_classes)
        
        total_entries += file_entries
        total_converted += converted_count
    
    # Copy the images directory
    source_images = os.path.join(source_dir, "images")
    target_images = os.path.join(target_dir, "images")
    
    if os.path.exists(source_images):
        if os.path.exists(target_images):
            shutil.rmtree(target_images)
        shutil.copytree(source_images, target_images)
        print(f"\nCopied images directory to {target_images}")
    else:
        print(f"\nWarning: Images directory not found at {source_images}")
    
    # Summary
    print(f"\n=== CONVERSION SUMMARY ===")
    print(f"Total entries processed: {total_entries}")
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
