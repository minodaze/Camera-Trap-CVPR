#%%
import json
import matplotlib.pyplot as plt
from collections import Counter

def load_json_data(filename):
    """Load JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def extract_images_from_data(data):
    """Extract all image information from data"""
    all_images = []
    
    if isinstance(data, dict):
        # Handle test.json format (contains checkpoints)
        for checkpoint, images in data.items():
            for img in images:
                if 'image_path' in img:  # Ensure there's an image path
                    all_images.append(img)
    elif isinstance(data, list):
        # Handle train-all.json format (direct image list)
        all_images = data
    
    return all_images

def check_overlap_between_datasets(train_images, test_images):
    """Check for overlapping images between train and test datasets using image paths"""
    train_paths = set(img.get('image_path', '') for img in train_images if 'image_path' in img)
    test_paths = set(img.get('image_path', '') for img in test_images if 'image_path' in img)
    overlap = train_paths & test_paths
    return overlap, len(train_paths), len(test_paths)

def get_class_distribution(images):
    """Get class distribution"""
    class_names = [img.get('class_name', 'unknown') for img in images if 'class_name' in img]
    return Counter(class_names)

def plot_histogram(class_counter, title, figsize=(12, 8)):
    """Plot histogram"""
    if not class_counter:
        print(f"No data to plot for {title}")
        return
    
    # Sort by count
    sorted_classes = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_classes)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(classes)), counts)
    plt.xlabel('Class Name')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def print_class_distribution(class_counter, title):
    """Print class distribution"""
    print(f"\n{title}")
    print("=" * 50)
    sorted_classes = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)
    for class_name, count in sorted_classes:
        print(f"{class_name}: {count}")
    print(f"Total classes: {len(class_counter)}")
    print(f"Total images: {sum(class_counter.values())}")

def main():
    # Load data files
    print("Loading data files...")
    train_data = load_json_data('/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/na/na_lebec_CA-22/train-all.json')
    test_data = load_json_data('/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/na/na_lebec_CA-22/test.json')

    # Extract all image information
    train_images = extract_images_from_data(train_data)
    test_images = extract_images_from_data(test_data)
    
    print(f"Train images loaded: {len(train_images)}")
    print(f"Test images loaded: {len(test_images)}")
    
    # Check for overlapping images between train and test
    print("\n" + "="*60)
    print("CHECKING FOR OVERLAPPING IMAGES BETWEEN TRAIN AND TEST")
    print("="*60)
    overlap, train_total, test_total = check_overlap_between_datasets(train_images, test_images)
    if overlap:
        print(f"Found {len(overlap)} overlapping image paths between train and test:")
        for path in list(overlap)[:10]:  # Show only first 10
            print(f"  {path}")
        if len(overlap) > 10:
            print(f"  ... and {len(overlap) - 10} more")
        print(f"\nOverlap rate: {len(overlap)}/{train_total} train images ({len(overlap)/train_total*100:.2f}%)")
        print(f"Overlap rate: {len(overlap)}/{test_total} test images ({len(overlap)/test_total*100:.2f}%)")
    else:
        print("✓ No overlapping images found between train and test sets.")
    
    # Get class distribution
    train_class_dist = get_class_distribution(train_images)
    test_class_dist = get_class_distribution(test_images)
    
    # Combine class distributions
    combined_class_dist = train_class_dist + test_class_dist
    
    # Print training data class distribution
    print_class_distribution(train_class_dist, "TRAINING DATA CLASS DISTRIBUTION")
    
    # Print combined class distribution
    print_class_distribution(combined_class_dist, "COMBINED DATA CLASS DISTRIBUTION")
    
    # Plot training data histogram
    plot_histogram(train_class_dist, "Training Data - Class Distribution")
    
    # Plot combined data histogram
    plot_histogram(combined_class_dist, "Combined Data - Class Distribution")

if __name__ == "__main__":
    main()
# %%
