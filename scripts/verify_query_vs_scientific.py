import json
import pandas as pd
from collections import Counter

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

def analyze_csv_fields(target_dataset_name=None):
    """
    Analyze the CSV file to understand the difference between query and scientific_name
    """
    csv_path = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/scripts/051625/lila-taxonomy-mapping_release.csv"
    df = pd.read_csv(csv_path)
    
    print("=== CSV FILE ANALYSIS ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Filter by target dataset if specified
    if target_dataset_name:
        df_filtered = df[df['dataset_name'] == target_dataset_name]
        print(f"Filtered to dataset '{target_dataset_name}': {len(df_filtered)} rows")
        df = df_filtered
    
    # Get all unique queries and scientific names
    queries = set(df['query'].dropna().unique())
    scientific_names = set(df['scientific_name'].dropna().unique())
    common_names = set(df['common_name'].dropna().unique())
    
    print(f"Unique queries: {len(queries)}")
    print(f"Unique scientific names: {len(scientific_names)}")
    print(f"Unique common names: {len(common_names)}")
    print()
    
    # Sample comparison
    print("=== SAMPLE QUERY vs SCIENTIFIC_NAME vs COMMON_NAME ===")
    sample_df = df[['query', 'scientific_name', 'common_name']].head(10)
    for _, row in sample_df.iterrows():
        print(f"Query: '{row['query']}' | Scientific: '{row['scientific_name']}' | Common: '{row['common_name']}'")
    print()
    
    return queries, scientific_names, common_names

def extract_class_names_from_json(filepath):
    """
    Extract all class names from JSON file
    """
    class_names = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def extract_names(obj):
            if isinstance(obj, dict):
                if 'class_name' in obj:
                    class_names.add(obj['class_name'])
                for value in obj.values():
                    extract_names(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_names(item)
        
        extract_names(data)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return class_names

def main():
    """
    Main verification function
    """
    print("🔍 VERIFYING: Are JSON class_names using QUERY or SCIENTIFIC_NAME?")
    print("=" * 70)
    
    # Get dataset mapping
    dataset_mapping = get_dataset_mapping()
    
    # Extract class names from JSON files
    json_files = [
        "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/serengeti/serengeti_L10/train-all.json",
        "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/serengeti/serengeti_L10/test.json"
    ]
    
    # Determine dataset name from file path
    input_dataset_name = None
    for file_path in json_files:
        if 'serengeti_L10' in file_path:
            input_dataset_name = 'serengeti_L10'
            break
    
    if input_dataset_name not in dataset_mapping:
        print(f"❌ Dataset '{input_dataset_name}' not found in mapping!")
        return
    
    target_dataset_name = dataset_mapping[input_dataset_name]
    print(f"📊 Input dataset: '{input_dataset_name}' -> CSV dataset: '{target_dataset_name}'")
    
    # Analyze CSV file for this specific dataset
    queries, scientific_names, common_names = analyze_csv_fields(target_dataset_name)
    
    all_json_class_names = set()
    
    for filepath in json_files:
        print(f"\n=== ANALYZING: {filepath.split('/')[-1]} ===")
        class_names = extract_class_names_from_json(filepath)
        all_json_class_names.update(class_names)
        print(f"Found {len(class_names)} unique class names")
        print("Sample class names:", sorted(list(class_names))[:10])
    
    print(f"\n=== OVERALL JSON ANALYSIS ===")
    print(f"Total unique class names across all JSON files: {len(all_json_class_names)}")
    print("All JSON class names:")
    for name in sorted(all_json_class_names):
        print(f"  - {name}")
    
    # Check matches
    print(f"\n=== MATCHING ANALYSIS ===")
    
    # Check against queries
    query_matches = all_json_class_names & queries
    query_misses = all_json_class_names - queries
    
    # Check against scientific names
    scientific_matches = all_json_class_names & scientific_names
    scientific_misses = all_json_class_names - scientific_names
    
    # Check against common names
    common_matches = all_json_class_names & common_names
    common_misses = all_json_class_names - common_names
    
    print(f"📊 MATCHING RESULTS:")
    print(f"JSON class names matching CSV QUERIES: {len(query_matches)}/{len(all_json_class_names)} ({len(query_matches)/len(all_json_class_names)*100:.1f}%)")
    print(f"JSON class names matching CSV SCIENTIFIC_NAMES: {len(scientific_matches)}/{len(all_json_class_names)} ({len(scientific_matches)/len(all_json_class_names)*100:.1f}%)")
    print(f"JSON class names matching CSV COMMON_NAMES: {len(common_matches)}/{len(all_json_class_names)} ({len(common_matches)/len(all_json_class_names)*100:.1f}%)")
    
    print(f"\n🎯 DETAILED ANALYSIS:")
    
    print(f"\n✅ JSON class names found in CSV QUERIES:")
    for name in sorted(query_matches):
        print(f"  - {name}")
    
    if query_misses:
        print(f"\n❌ JSON class names NOT found in CSV QUERIES:")
        for name in sorted(query_misses):
            print(f"  - {name}")
    
    print(f"\n✅ JSON class names found in CSV SCIENTIFIC_NAMES:")
    for name in sorted(scientific_matches):
        print(f"  - {name}")
    
    if scientific_misses:
        print(f"\n❌ JSON class names NOT found in CSV SCIENTIFIC_NAMES:")
        for name in sorted(scientific_misses):
            print(f"  - {name}")
    
    print(f"\n✅ JSON class names found in CSV COMMON_NAMES:")
    for name in sorted(common_matches):
        print(f"  - {name}")
    
    if common_misses:
        print(f"\n❌ JSON class names NOT found in CSV COMMON_NAMES:")
        for name in sorted(common_misses):
            print(f"  - {name}")
    
    # Conclusion
    print(f"\n🏁 CONCLUSION:")
    if len(query_matches) > len(scientific_matches) and len(query_matches) > len(common_matches):
        print("✅ JSON class names are most likely using CSV QUERY field!")
        print(f"   - {len(query_matches)} matches with queries")
        print(f"   - {len(scientific_matches)} matches with scientific names")
        print(f"   - {len(common_matches)} matches with common names")
    elif len(scientific_matches) > len(query_matches) and len(scientific_matches) > len(common_matches):
        print("✅ JSON class names are most likely using CSV SCIENTIFIC_NAME field!")
    elif len(common_matches) > len(query_matches) and len(common_matches) > len(scientific_matches):
        print("✅ JSON class names are most likely using CSV COMMON_NAME field!")
    else:
        print("🤔 Results are inconclusive - need manual inspection")
    
    # Show specific examples for verification
    print(f"\n🔍 VERIFICATION EXAMPLES:")
    csv_df = pd.read_csv("/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/scripts/051625/lila-taxonomy-mapping_release.csv")
    csv_df_filtered = csv_df[csv_df['dataset_name'] == target_dataset_name]
    
    sample_json_names = list(all_json_class_names)[:5]
    for json_name in sample_json_names:
        print(f"\nJSON class_name: '{json_name}'")
        
        # Find in CSV (filtered by dataset)
        query_row = csv_df_filtered[csv_df_filtered['query'] == json_name]
        scientific_row = csv_df_filtered[csv_df_filtered['scientific_name'] == json_name]
        common_row = csv_df_filtered[csv_df_filtered['common_name'] == json_name]
        
        if not query_row.empty:
            row = query_row.iloc[0]
            print(f"  Found as QUERY: '{row['query']}' -> scientific: '{row['scientific_name']}' -> common: '{row['common_name']}'")
        
        if not scientific_row.empty:
            row = scientific_row.iloc[0]
            print(f"  Found as SCIENTIFIC: '{row['scientific_name']}' -> query: '{row['query']}' -> common: '{row['common_name']}'")
        
        if not common_row.empty:
            row = common_row.iloc[0]
            print(f"  Found as COMMON: '{row['common_name']}' -> query: '{row['query']}' -> scientific: '{row['scientific_name']}'")
        
        if query_row.empty and scientific_row.empty and common_row.empty:
            print(f"  ❌ NOT FOUND in CSV for dataset '{target_dataset_name}'!")

if __name__ == "__main__":
    main()
