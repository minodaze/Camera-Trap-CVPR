import os
import colorsys
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from tools.analysis.utils.plot_utils import plot_piechart, plot_multiple_piecharts

class PlotAnalysis:
    def __init__(self, config, dataset, analysis_path):
        """
        Initialize the PlotAnalysis module.

        Args:
            config (dict): Configuration dictionary.
            dataset (Dataset): Prepared dataset object.
            analysis_path (str): Root path for saving analysis results.
        """
        self.config = config.get("plot_analysis", {})
        self.dataset = dataset
        self.analysis_path = analysis_path
        self.camera_colors = self._assign_camera_colors()

    def _assign_camera_colors(self):
        """
        Assign unique colors to each class in each camera for consistency across plots.

        Returns:
            dict: A dictionary mapping camera names to class-color mappings.
        """
        camera_colors = {}
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                all_classes = set()
                for entry in camera_data["data"]:
                    for cls in entry["class"]:
                        all_classes.add(cls["class_name"])  # Use class_name instead of class_id

                # Combine tab20, tab20b, and tab20c colormaps
                colormaps = [plt.cm.get_cmap("tab20"), plt.cm.get_cmap("tab20b"), plt.cm.get_cmap("tab20c")]
                unique_colors = []
                for i in range(len(all_classes)):
                    colormap = colormaps[i // 20 % len(colormaps)]  # Cycle through colormaps
                    unique_colors.append(colormap(i % 20))

                camera_colors[camera_name] = {cls: unique_colors[idx] for idx, cls in enumerate(sorted(all_classes))}
        return camera_colors

    def run(self):
        """
        Run all plot-related analyses based on the configuration.
        """
        if self.config.get("ckp_piechart", True):
            self._plot_ckp_piecharts()
        if self.config.get("count_histogram", True):
            self._plot_count_histograms()

    def _plot_ckp_piecharts(self):
        """
        Generate and save pie charts for checkpoints.
        """
        print("📊 Generating checkpoint pie charts...")
        total_len = len(self.dataset.metadata)
        
        for i, (dataset_name, cameras) in enumerate(self.dataset.metadata.items()):
            for camera_name, camera_data in tqdm(cameras.items(), desc=f"({i} / {total_len}) Processing {dataset_name}'s cameras", leave=False):
                if "ckp" not in camera_data:
                    continue

                # Prepare paths for saving plots
                camera_path = os.path.join(self.analysis_path, dataset_name, camera_name, "plots", "ckp_piechart")

                # Generate pie charts
                self._generate_piechart_group(camera_data, camera_path, dataset_name, camera_name)

    def _generate_piechart_group(self, camera_data, camera_path, dataset_name, camera_name):
        """
        Generate and save all four types of pie charts for a camera.

        Args:
            camera_data (dict): Metadata for the camera.
            camera_path (str): Path to save the plots.
            dataset_name (str): Name of the dataset.
            camera_name (str): Name of the camera.
        """
        # Full-train pie chart
        full_train_data = [entry for ckp in camera_data["ckp"].values() for entry in ckp["train"]]
        class_counts = self._calculate_class_counts(full_train_data)
        # colors = [self.camera_colors[camera_name][cls] for cls in class_counts.keys()]
        plot_piechart(
            class_counts,
            f"Full Train - {camera_name}",
            self.camera_colors[camera_name],
            os.path.join(camera_path, "full-train.png"),
            show_all_legend=True
        )

        # Default train pie charts
        default_piechart_data = []
        for ckp_id, ckp_data in camera_data["ckp"].items():
            class_counts = self._calculate_class_counts(ckp_data["train"])
            default_piechart_data.append((class_counts, f"Ckp {ckp_id}"))
        plot_multiple_piecharts(
            default_piechart_data,
            os.path.join(camera_path, "default.png"),
            f"Default Train - {camera_name}",
            self.camera_colors[camera_name]
        )

        # Test pie charts
        test_piechart_data = []
        for ckp_id, ckp_data in camera_data["ckp"].items():
            class_counts = self._calculate_class_counts(ckp_data["val"])
            test_piechart_data.append((class_counts, f"Ckp {ckp_id}"))
        plot_multiple_piecharts(
            test_piechart_data,
            os.path.join(camera_path, "test.png"),
            f"Test - {camera_name}",
            self.camera_colors[camera_name]
        )

        # Accumulated train pie charts
        accumulated_data = []
        accumulated_piechart_data = []
        for ckp_id, ckp_data in camera_data["ckp"].items():
            accumulated_data.extend(ckp_data["train"])
            class_counts = self._calculate_class_counts(accumulated_data)
            accumulated_piechart_data.append((class_counts, f"Up to Ckp {ckp_id}"))
        plot_multiple_piecharts(
            accumulated_piechart_data,
            os.path.join(camera_path, "acc-train.png"),
            f"Accumulated Train - {camera_name}",
            self.camera_colors[camera_name]
        )

    def _plot_count_histograms(self):
        """
        Generate and save count histograms for train and test data.
        """
        print("📊 Generating count histograms...")
        total_len = len(self.dataset.metadata)
        
        for i, (dataset_name, cameras) in enumerate(self.dataset.metadata.items()):
            for camera_name, camera_data in tqdm(cameras.items(), desc=f"({i} / {total_len}) Processing {dataset_name}'s cameras", leave=False):
                if "data" not in camera_data:
                    continue

                # Prepare paths for saving plots
                camera_path = os.path.join(self.analysis_path, dataset_name, camera_name, "plots", "histograms")
                os.makedirs(camera_path, exist_ok=True)

                # Generate histograms for train and test data
                combined_data = [entry for ckp in camera_data["ckp"].values() for entry in ckp["train"] + ckp["val"]]

                self._generate_histogram(combined_data, os.path.join(camera_path, "class_histogram.png"), f"Class Histogram - {camera_name}", camera_name)

    def _generate_histogram(self, data, output_path, title, camera_name):
        """
        Generate and save a histogram for the given data.

        Args:
            data (list): List of data entries.
            output_path (str): Path to save the histogram.
            title (str): Title of the histogram.
            camera_name (str): Name of the camera (used for color mapping).
        """
        # Calculate class counts
        class_counts = self._calculate_class_counts(data)

        # Sort class counts by value (descending)f
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        classes = [cls for cls, _ in sorted_classes]
        counts = [count for _, count in sorted_classes]

        # Get colors for the sorted classes
        colors = [self.camera_colors[camera_name][cls] for cls in classes]

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.bar(classes, counts, color=colors)
        plt.xlabel("Class Names")
        plt.ylabel("Counts")
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_path, dpi=300)
        plt.close()

    def _calculate_class_counts(self, data):
        """
        Calculate class counts for a given set of data entries.

        Args:
            data (list): List of data entries for a camera.

        Returns:
            dict: Class counts.
        """
        counts = defaultdict(int)
        for entry in data:
            for cls in entry["class"]:
                counts[cls["class_name"]] += 1  # Use class_name instead of class_id
        return counts