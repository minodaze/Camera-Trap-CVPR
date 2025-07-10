from collections import defaultdict
import os
import csv
from datetime import datetime
import math

# Define how each metric should be saved
METRIC_SAVE_CONFIG = {
    "ts_l1_accumulated": "average",  # Save the average value
    "ts_l1_full": "average",        # Save the average value
    "gini_index": "value",          # Save a single value
    "l1_test": "value",             # Save a single value
}

class MetricAnalysis:
    def __init__(self, config, dataset, analysis_path):
        """
        Initialize the MetricAnalysis module.

        Args:
            config (dict): Configuration dictionary.
            dataset (Dataset): Prepared dataset object.
            analysis_path (str): Root path for saving analysis results.
        """
        self.config = config.get("metrics_analysis", {})
        self.dataset = dataset
        self.analysis_path = analysis_path

    def run(self):
        """
        Run all metric-related analyses based on the configuration.
        """
        if self.config.get("ts_l1_accumulated", False):
            self._compute_ts_l1_accumulated()
        if self.config.get("ts_l1_full", False):
            self._compute_ts_l1_full()
        if self.config.get("gini_index", False):
            self._compute_gini_index()
        if self.config.get("l1_test", False):
            self._compute_l1_test()
        self._save_metrics_results()

    def _compute_ts_l1_accumulated(self):
        """
        Compute the accumulated L1 metric (time-series perspective) and store the results in the metadata.
        Train data is accumulated up to the previous checkpoint, and test data is from the current checkpoint.
        """
        print("▶️ Computing time-series L1 accumulated metric...")

        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue  # Skip if no checkpoints are available

                cumulative_train_data = []
                analysis_results = {}
                total_l1_metric = 0
                total_class_count = 0

                sorted_ckp_ids = sorted(camera_data["ckp"].keys())

                for idx, ckp_id in enumerate(sorted_ckp_ids):
                    ckp_data = camera_data["ckp"][ckp_id]

                    if idx == 0:
                        continue  # Skip the first checkpoint

                    prev_ckp_id = sorted_ckp_ids[idx - 1]
                    cumulative_train_data.extend(camera_data["ckp"][prev_ckp_id]["train"])
                    train_data = cumulative_train_data
                    test_data = ckp_data["val"]

                    train_counts = self._calculate_class_counts(train_data)
                    test_counts = self._calculate_class_counts(test_data)

                    train_total = sum(train_counts.values())
                    test_total = sum(test_counts.values())

                    l1_metric = 0

                    for cls, test_count in test_counts.items():
                        q = test_count / test_total if test_total > 0 else 0
                        train_count = train_counts.get(cls, 0)
                        p = train_count / train_total if train_total > 0 else 0

                        if q > p:
                            l1_metric += abs(q - p)
                            total_l1_metric += abs(q - p)
                            total_class_count += 1

                    analysis_results[ckp_id] = {"ts_l1_accumulated": l1_metric}

                global_avg_l1_metric = total_l1_metric / total_class_count if total_class_count > 0 else 0

                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["ts_l1_accumulated"] = {
                    "checkpoints": analysis_results,
                    "average": global_avg_l1_metric,
                }

    def _compute_ts_l1_full(self):
        """
        Compute the accumulated L1 metric using all train checkpoints for every test checkpoint.
        """
        print("▶️ Computing time-series L1 full metric...")

        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue  # Skip if no checkpoints are available

                full_train_data = []
                for ckp_data in camera_data["ckp"].values():
                    full_train_data.extend(ckp_data["train"])

                analysis_results = {}
                total_l1_metric = 0
                total_class_count = 0

                sorted_ckp_ids = sorted(camera_data["ckp"].keys())

                for ckp_id in sorted_ckp_ids:
                    ckp_data = camera_data["ckp"][ckp_id]
                    train_data = full_train_data
                    test_data = ckp_data["val"]

                    train_counts = self._calculate_class_counts(train_data)
                    test_counts = self._calculate_class_counts(test_data)

                    train_total = sum(train_counts.values())
                    test_total = sum(test_counts.values())

                    l1_metric = 0

                    for cls, test_count in test_counts.items():
                        q = test_count / test_total if test_total > 0 else 0
                        train_count = train_counts.get(cls, 0)
                        p = train_count / train_total if train_total > 0 else 0

                        if q > p:
                            l1_metric += abs(q - p)
                            total_l1_metric += abs(q - p)
                            total_class_count += 1

                    analysis_results[ckp_id] = {"ts_l1_full": l1_metric}

                global_avg_l1_metric = total_l1_metric / total_class_count if total_class_count > 0 else 0

                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["ts_l1_full"] = {
                    "checkpoints": analysis_results,
                    "average": global_avg_l1_metric,
                }

    def _compute_gini_index(self):
        """
        Compute the Gini index for the class distribution across all checkpoints (combined train and test data).
        """
        print("▶️ Computing Gini index...")

        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue  # Skip if no checkpoints are available

                # Aggregate class counts across all checkpoints (train + test)
                combined_class_counts = defaultdict(int)
                for ckp_data in camera_data["ckp"].values():
                    for entry in ckp_data["train"] + ckp_data["val"]:
                        for cls in entry["class"]:
                            combined_class_counts[cls["class_id"]] += 1

                # Calculate total samples
                total_samples = sum(combined_class_counts.values())

                # Calculate Gini index
                if total_samples == 0:
                    gini_index = 0
                else:
                    gini_index = 1 - sum((count / total_samples) ** 2 for count in combined_class_counts.values())

                # Store the Gini index in the camera's metadata
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["gini_index"] = {"value": gini_index}

    def _compute_l1_test(self):
        """
        Compute the L1 drift metric across adjacent checkpoints and store the result in the metadata.
        This measures how much class distributions change between adjacent checkpoints.
        Normalization is performed within each checkpoint (local normalization).
        """
        print("▶️ Computing L1 drift metric with local normalization...")

        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue  # Skip if no checkpoints are available

                sorted_ckp_ids = sorted(camera_data["ckp"].keys())
                if len(sorted_ckp_ids) < 2:
                    # L1 drift requires at least two checkpoints
                    continue

                total_l1_drift = 0
                num_pairs = 0

                for i in range(len(sorted_ckp_ids) - 1):
                    ckp_id_1 = sorted_ckp_ids[i]
                    ckp_id_2 = sorted_ckp_ids[i + 1]

                    # Get class counts for the two checkpoints
                    ckp_1_data = camera_data["ckp"][ckp_id_1]["val"]
                    ckp_2_data = camera_data["ckp"][ckp_id_2]["val"]

                    counts_1 = self._calculate_class_counts(ckp_1_data)
                    counts_2 = self._calculate_class_counts(ckp_2_data)

                    # Normalize class frequencies within each checkpoint
                    total_1 = sum(counts_1.values())
                    total_2 = sum(counts_2.values())

                    normalized_1 = {cls: count / total_1 for cls, count in counts_1.items() if total_1 > 0}
                    normalized_2 = {cls: count / total_2 for cls, count in counts_2.items() if total_2 > 0}

                    # Get the union of all classes
                    all_classes = set(normalized_1.keys()).union(set(normalized_2.keys()))

                    # Compute L1 drift for this pair of checkpoints
                    l1_drift = sum(abs(normalized_1.get(cls, 0) - normalized_2.get(cls, 0)) for cls in all_classes)
                    total_l1_drift += l1_drift
                    num_pairs += 1

                # Calculate the average L1 drift across all pairs of adjacent checkpoints
                average_l1_drift = total_l1_drift / num_pairs if num_pairs > 0 else 0

                # Store the result in the camera's metadata
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["l1_test"] = {"value": average_l1_drift}

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
                counts[cls["class_id"]] += 1
        return counts

    def _save_metrics_results(self):
        """
        Save each camera's basic info and metrics to a CSV file.
        """
        print("💾 Saving metrics results to CSV...")
        output_path = os.path.join(self.analysis_path, "metrics_results.csv")

        headers = [
            "Dataset",
            "Camera Name",
            "Total Images",
            "Total Unique Classes",
            "Time Span (Months)",
            "Number of Checkpoints",
        ]

        metric_headers = []
        for metric_name in self.config:
            if self.config[metric_name]:
                metric_headers.append(metric_name)

        headers.extend(metric_headers)

        with open(output_path, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()

            for dataset_name, cameras in self.dataset.metadata.items():
                for camera_name, camera_data in cameras.items():
                    if "ckp" not in camera_data:
                        continue

                    unique_images = set(entry["image_id"] for ckp in camera_data["ckp"].values() for entry in ckp["train"] + ckp["val"])
                    total_images = len(unique_images)

                    unique_classes = set(cls["class_id"] for ckp in camera_data["ckp"].values() for entry in ckp["train"] + ckp["val"] for cls in entry["class"])
                    total_unique_classes = len(unique_classes)

                    timestamps = [datetime.strptime(entry["datetime"], "%Y:%m:%d %H:%M:%S") for ckp in camera_data["ckp"].values() for entry in ckp["train"] + ckp["val"]]
                    if timestamps:
                        min_time = min(timestamps)
                        max_time = max(timestamps)
                        time_span_months = (max_time.year - min_time.year) * 12 + (max_time.month - min_time.month)
                    else:
                        time_span_months = 0

                    num_checkpoints = len(camera_data["ckp"])

                    row = {
                        "Dataset": dataset_name,
                        "Camera Name": camera_name,
                        "Total Images": total_images,
                        "Total Unique Classes": total_unique_classes,
                        "Time Span (Months)": time_span_months,
                        "Number of Checkpoints": num_checkpoints,
                    }

                    for metric_name in metric_headers:
                        metric_value = camera_data.get("analysis", {}).get(metric_name, {}).get(METRIC_SAVE_CONFIG[metric_name], "N/A")
                        if isinstance(metric_value, float):
                            metric_value = round(metric_value, 4)
                        row[metric_name] = metric_value

                    writer.writerow(row)