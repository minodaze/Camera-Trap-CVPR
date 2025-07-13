#!/usr/bin/env python3
"""
Example usage of plot_per_class_from_json.py as an importable module.
This shows how to use it like plot.py - directly in Python scripts.
"""

import plot_per_class_from_json

# Example 1: Plot from a dictionary (like plot.py)
def example_dict_plotting():
    """Example of plotting directly from a metrics dictionary."""
    print("Example 1: Plotting from dictionary")
    
    metrics = {
        'class_names': ['Fraxinus americana', 'Acer saccharum', 'Quercus alba', 'Pinus strobus'],
        'per_class_accuracy': [0.85, 0.72, 0.91, 0.68],
        'samples_per_class': [100, 80, 120, 90],
        'overall_accuracy': 0.790,
        'balanced_accuracy': 0.789,
        'dataset': 'ENO_C05',
        'method': 'bioclip2',
        'checkpoint': 'ckp_1'
    }
    
    # Direct plotting (like plot.py)
    fig = plot_per_class_from_json.plot_from_dict(
        metrics, 
        title="Per-Class Accuracy Example",
        output_path="example_dict_plot.png",
        style='publication',
        show=False  # Set to True to display the plot
    )
    
    print("✓ Plot saved as example_dict_plot.png")
    return fig

# Example 2: Plot from JSON string
def example_json_string_plotting():
    """Example of plotting from a JSON string."""
    print("\nExample 2: Plotting from JSON string")
    
    json_string = '''
    {
        "class_names": ["Species A", "Species B", "Species C"],
        "per_class_accuracy": [0.82, 0.75, 0.88],
        "samples_per_class": [95, 105, 110],
        "overall_accuracy": 0.817,
        "balanced_accuracy": 0.817,
        "dataset": "MAD_MAD05",
        "method": "baseline",
        "checkpoint": "ckp_2"
    }
    '''
    
    fig = plot_per_class_from_json.plot_from_json_string(
        json_string,
        title="JSON String Example",
        output_path="example_json_plot.png",
        style='nature',
        show=False
    )
    
    print("✓ Plot saved as example_json_plot.png")
    return fig

# Example 3: Compare multiple methods
def example_comparison_plotting():
    """Example of comparing multiple methods."""
    print("\nExample 3: Comparing multiple methods")
    
    method1 = {
        'class_names': ['Species A', 'Species B', 'Species C'],
        'per_class_accuracy': [0.82, 0.75, 0.88],
        'samples_per_class': [95, 105, 110],
        'overall_accuracy': 0.817,
        'balanced_accuracy': 0.817
    }
    
    method2 = {
        'class_names': ['Species A', 'Species B', 'Species C'],
        'per_class_accuracy': [0.79, 0.81, 0.85],
        'samples_per_class': [95, 105, 110],
        'overall_accuracy': 0.817,
        'balanced_accuracy': 0.817
    }
    
    method3 = {
        'class_names': ['Species A', 'Species B', 'Species C'],
        'per_class_accuracy': [0.85, 0.70, 0.90],
        'samples_per_class': [95, 105, 110],
        'overall_accuracy': 0.817,
        'balanced_accuracy': 0.817
    }
    
    fig = plot_per_class_from_json.plot_comparison_from_dicts(
        [method1, method2, method3],
        labels=['Baseline', 'BioCLIP', 'Fine-tuned'],
        title="Method Comparison",
        output_path="example_comparison_plot.png",
        plot_type='bar',
        style='publication',
        show=False
    )
    
    print("✓ Comparison plot saved as example_comparison_plot.png")
    return fig

# Example 4: Plot from existing JSON files
def example_json_files_plotting():
    """Example of plotting from JSON files."""
    print("\nExample 4: Plotting from JSON files")
    
    # This assumes you have JSON files already
    json_files = [
        'extracted_metrics/ENO_C05_bioclip2_ckp_1_metrics.json',
        'extracted_metrics/ENO_C05_baseline_ckp_1_metrics.json'
    ]
    
    # Check if files exist
    import os
    existing_files = [f for f in json_files if os.path.exists(f)]
    
    if existing_files:
        fig = plot_per_class_from_json.plot_from_json_files(
            existing_files,
            labels=None,  # Will use filenames as labels
            title="File Comparison",
            output_path="example_files_plot.png",
            plot_type='bar',
            style='publication',
            show=False
        )
        print("✓ Files plot saved as example_files_plot.png")
        return fig
    else:
        print("  No existing JSON files found, skipping this example")
        return None

if __name__ == "__main__":
    print("Running plot_per_class_from_json.py API examples...")
    print("=" * 50)
    
    # Run all examples
    example_dict_plotting()
    example_json_string_plotting()
    example_comparison_plotting()
    example_json_files_plotting()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nYou can now use plot_per_class_from_json in your scripts like:")
    print("import plot_per_class_from_json")
    print("fig = plot_per_class_from_json.plot_from_dict(my_metrics)")
