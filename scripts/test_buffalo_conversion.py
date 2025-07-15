#!/usr/bin/env python3
"""
Test script to check what 'buffalo' converts to in the mapping
"""

import pandas as pd

def test_buffalo_conversion():
    print("🔍 Testing buffalo conversion...")
    
    # Load the CSV mapping file
    csv_file = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/scripts/051625/lila-taxonomy-mapping_release.csv"
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ CSV loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Create mapping from query to common_name (same logic as original code)
        mapping = {}
        for _, row in df.iterrows():
            query = row['query']
            common_name = row['common_name']
            
            # Skip if either query or common_name is NaN/None
            if pd.notna(query) and pd.notna(common_name):
                mapping[query] = common_name
        
        print(f"\n📊 Total mappings created: {len(mapping)}")
        
        # Test buffalo specifically
        test_query = "buffalo"
        print(f"\n🎯 Testing query: '{test_query}'")
        
        if test_query in mapping:
            result = mapping[test_query]
            print(f"✅ Found mapping: '{test_query}' -> '{result}'")
        else:
            print(f"❌ No mapping found for '{test_query}'")
        
        # Show all buffalo-related entries in the CSV
        print(f"\n🔍 All buffalo-related entries in CSV:")
        buffalo_entries = df[df['query'].str.contains('buffalo', case=False, na=False)]
        
        if not buffalo_entries.empty:
            for idx, row in buffalo_entries.iterrows():
                print(f"  Dataset: {row['dataset_name']}")
                print(f"  Query: '{row['query']}'")
                print(f"  Common name: '{row['common_name']}'")
                print(f"  Scientific name: '{row['scientific_name']}'")
                print("  ---")
        else:
            print("  No buffalo entries found")
        
        # Test other animals from the config for comparison
        test_animals = ["baboon", "domesticanimal", "gazellethomsons", "hyenaspotted", 
                       "wildebeestblue", "zebraplains"]
        
        print(f"\n🧪 Testing other animals from config:")
        for animal in test_animals:
            if animal in mapping:
                print(f"  '{animal}' -> '{mapping[animal]}'")
            else:
                print(f"  '{animal}' -> NOT FOUND")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_buffalo_conversion()
