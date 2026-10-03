import json
import os
from Data_Stream import *
from Oracle import *
from A_REDIN import *
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

matplotlib.use('TkAgg')  # Adjust backend as needed
from data_visualization import *
from more_stats import *
from main_helper_functions import *
from sklearn.metrics import ConfusionMatrixDisplay
from cluster_visualization import ClusterEvolutionPlotter


def load_config(config_path):
    """Load the configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as f:
        return json.load(f)


def validate_config(config):
    """Validate a single run configuration."""
    required_keys = [
        "DATA_SOURCE", "N_REL_CLASSES", "KAPPA", "QS_VAR", "SM_VAR",
        "REL_PROC_VAR", "K_COMP_PTS", "NGHBHOOD_MERGE", "SINGLETON_MERGE",
        "DATA_WINDOW_SIZE", "NUM_POINTS_TO_PROCESS", "GRAPH_BATCH_SIZE",
        "VERBOSE_FLAGS", "MAKE_GRAPHS", "MAKE_EVO_GRAPHS", "RANDOM_SEED_OFFSET"
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Additional validation
    if config["DATA_WINDOW_SIZE"] <= 0:
        raise ValueError("DATA_WINDOW_SIZE must be greater than 0")
    if config["K_COMP_PTS"] < 1:
        raise ValueError("K_COMP_PTS must be 1 or greater")
    if config["NGHBHOOD_MERGE"] and config["K_COMP_PTS"] < 2:
        raise ValueError("NGHBHOOD_MERGE requires K_COMP_PTS >= 2")
    if config["SINGLETON_MERGE"] and config["K_COMP_PTS"] < 2:
        raise ValueError("SINGLETON_MERGE requires K_COMP_PTS >= 2")
    if config["DATA_SOURCE"] not in ["NICE", "MNIST", "MNIST_2D", "EMNIST", "PARKING_LOT"]:
        raise ValueError(f"Invalid DATA_SOURCE: {config['DATA_SOURCE']}")


def save_results(results, results_file):
    """Save results to the specified file."""
    try:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_file}")
    except Exception as e:
        print(f"Error saving results to file: {e}")


def get_existing_result(config, existing_results):
    """Check if a configuration already exists in the results and return the result if it does."""
    for result in existing_results:
        existing_config = result.get("config", {})
        if existing_config == config:
            return result
    return None


def run_ared(config):
    """Run ARED with the given configuration."""
    # Capture start time
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # Extract configuration parameters
    DATA_SOURCE = config["DATA_SOURCE"]
    N_REL_CLASSES = config["N_REL_CLASSES"]
    KAPPA = config["KAPPA"]
    QS_VAR = config["QS_VAR"]
    SM_VAR = config["SM_VAR"]
    REL_PROC_VAR = config["REL_PROC_VAR"]
    K_COMP_PTS = config["K_COMP_PTS"]
    NGHBHOOD_MERGE = config["NGHBHOOD_MERGE"]
    SINGLETON_MERGE = config["SINGLETON_MERGE"]
    DATA_WINDOW_SIZE = config["DATA_WINDOW_SIZE"]
    NUM_POINTS_TO_PROCESS = config["NUM_POINTS_TO_PROCESS"]
    GRAPH_BATCH_SIZE = config["GRAPH_BATCH_SIZE"]
    VERBOSE_FLAGS = config["VERBOSE_FLAGS"]
    MAKE_GRAPHS = config["MAKE_GRAPHS"]
    MAKE_EVO_GRAPHS = config["MAKE_EVO_GRAPHS"]
    RANDOM_SEED_OFFSET = config["RANDOM_SEED_OFFSET"]

    print(f"\nRunning ARED with configuration: DATA_SOURCE={DATA_SOURCE}, N_REL_CLASSES={N_REL_CLASSES}, KAPPA={KAPPA}")

    # Get data
    X_skewed, y_w_rel, sparsity_levels, rel_classes = get_data(
        DATA_SOURCE, N_REL_CLASSES, VERBOSE_FLAGS, RANDOM_SEED_OFFSET
    )

    # Initialize Data Stream, Oracle, and ARED
    data_stream = Data_Stream(X_skewed, y_w_rel)
    oracle = Oracle(X_skewed, y_w_rel)
    ared = ARED(
        oracle, KAPPA, DATA_WINDOW_SIZE, K_COMP_PTS, QS_VAR,
        REL_PROC_VAR, SM_VAR, NGHBHOOD_MERGE, SINGLETON_MERGE, VERBOSE_FLAGS
    )
    start_time, times, num_correct_queries, num_queries, num_clusters, num_labels, pt_dists, \
        num_pts_searched_list, conf_matrices, num_queries_last_batch = set_up_stats(ared)

    if MAKE_EVO_GRAPHS:
        evo_plotter = ClusterEvolutionPlotter()
        build_up_flag_evo_plotter = False

    # Process first point
    ared.process_first_point(data_stream.stream_new_data_point())

    # Determine number of points to process
    dataset_size = len(X_skewed)
    if NUM_POINTS_TO_PROCESS == -1:
        NUM_POINTS_TO_PROCESS = data_stream.get_remaining_num_points()
    else:
        NUM_POINTS_TO_PROCESS = min(NUM_POINTS_TO_PROCESS, dataset_size - 1)

    # Stream and process data
    for i in range(1, NUM_POINTS_TO_PROCESS + 1):
        if MAKE_EVO_GRAPHS:
            if i == 100:
                evo_plotter.add_snapshot(ared, X_skewed, y_w_rel, f"a) First 100 Points (Run: {DATA_SOURCE})")
            elif i > 1000 and not ared.l_buf.build_up_period and not build_up_flag_evo_plotter:
                evo_plotter.add_snapshot(ared, X_skewed, y_w_rel, f"b) Labeled Buffer Full (Run: {DATA_SOURCE})")
                build_up_flag_evo_plotter = True

        # Save and print per batch
        if i % GRAPH_BATCH_SIZE == 0:
            j = i // GRAPH_BATCH_SIZE
            print(f"Processing point {i}...")
            times.append(time.time())
            num_correct_queries.append(ared.num_correct_queries)
            num_queries.append(ared.num_queries)
            num_clusters.append(len(ared.subspace_partition.cluster_dict))
            num_labels.append(len(ared.subspace_partition.set_of_known_labels))
            conf_matrices.append(ared.conf_matrix.copy())
            if 0 in VERBOSE_FLAGS:
                if j > 1:
                    print(f"Last {GRAPH_BATCH_SIZE} points took {times[j] - times[j - 1]:.2f} seconds")
                    print(f"Points queried in this batch: {num_queries[j - 1] - num_queries[j - 2]}")
                    print(f"Number of clusters: {num_clusters[j - 1]}")

            if MAKE_EVO_GRAPHS:
                evo_plotter.plot_clusters_colored_by_label(
                    ared, X_skewed, y_w_rel, title=f"Cluster Visualization by Label (Run: {DATA_SOURCE})"
                )

            if SINGLETON_MERGE:
                ared.singleton_merge()

        pt_dist, num_pts_searched = ared.process_point(data_stream.stream_new_data_point())
        pt_dists.append(pt_dist)
        num_pts_searched_list.append(num_pts_searched)

    if MAKE_EVO_GRAPHS:
        evo_plotter.add_snapshot(ared, X_skewed, y_w_rel, f"c) with Forgetting (Run: {DATA_SOURCE})")
        evo_plotter.plot()
        evo_plotter.plot_dataset(X_skewed, y_w_rel)

    print(f"ARED DONE for {DATA_SOURCE}")

    # Calculate and plot statistics
    PLOT_FLAG = MAKE_GRAPHS
    single_rel_recall_list, query_precision_list, rel_individual_recalls, query_rate_ave_list = \
        calc_rel_recall_query_precision(sparsity_levels, conf_matrices, rel_classes, ared, num_correct_queries, \
                                        num_queries, PLOT_FLAG, GRAPH_BATCH_SIZE, NUM_POINTS_TO_PROCESS)

    print("Confusion Matrix:")
    print(ared.conf_matrix)
    with np.printoptions(threshold=sys.maxsize):
        print(ared.conf_matrix)

    if MAKE_GRAPHS:
        try:
            batch_num_pts = list(range(GRAPH_BATCH_SIZE, NUM_POINTS_TO_PROCESS + 1, GRAPH_BATCH_SIZE))
            plt.figure(figsize=(10, 5))
            plt.plot(batch_num_pts, num_clusters)
            plt.grid()
            plt.title(f"Number of Clusters over Time (Run: {DATA_SOURCE})")
            plt.xlabel("Number of Points Processed")
            plt.ylabel("Number of Clusters")
            plt.legend(["Number of clusters"])
            plt.savefig(f"clusters_{DATA_SOURCE}_kappa{KAPPA}_nrel{N_REL_CLASSES}.png")
            plt.close()

        except Exception as e:
            print(f"Error plotting graphs: {e}")

    time_elapsed = time.time() - start_time
    print(f"Run for {DATA_SOURCE} took {time_elapsed:.2f} seconds")
    print(f"ARED COMPLETE for {DATA_SOURCE}")

    # Capture completion time
    completion_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

    return single_rel_recall_list, query_precision_list, rel_individual_recalls, query_rate_ave_list, start_time_str, completion_time_str


if __name__ == '__main__':
    # Define the hyperparameter grid
    kappa_values = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, \
        0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.25, 1.5, 2]  # Example values for KAPPA; adjust as needed
    window_sizes = [100, 500, 1000, 2500, 5000, 8000, 10000, 20000, 30000]  # Example values for DATA_WINDOW_SIZE; adjust as needed
    data_sources = ["PARKING_LOT", "EMNIST"]  # All possible datasets

    # Base configuration (fixed parameters)
    base_config = {
        "N_REL_CLASSES": 3,  # Example; adjust as needed
        "QS_VAR": 0,  # Example; adjust as needed
        "SM_VAR": 0,  # Example; adjust as needed
        "REL_PROC_VAR": 0,  # Example; adjust as needed
        "K_COMP_PTS": 2,
        "NGHBHOOD_MERGE": True,
        "SINGLETON_MERGE": True,
        "NUM_POINTS_TO_PROCESS": -1,
        "GRAPH_BATCH_SIZE": 100,
        "VERBOSE_FLAGS": [0],
        "MAKE_GRAPHS": False,
        "MAKE_EVO_GRAPHS": False,
        "RANDOM_SEED_OFFSET": 0
    }

    # Precompute dataset sizes
    dataset_sizes = {}
    for data_source in data_sources:
        X_skewed, _, _, _ = get_data(
            data_source,
            base_config["N_REL_CLASSES"],
            base_config["VERBOSE_FLAGS"],
            base_config["RANDOM_SEED_OFFSET"]
        )
        dataset_sizes[data_source] = len(X_skewed)
        print(f"Dataset size for {data_source}: {dataset_sizes[data_source]}")

    # Initialize timed out pairs per data source
    timed_out_pairs = {ds: [] for ds in data_sources}

    # Generate configurations for the grid search
    configs = []
    for data_source in data_sources:
        for kappa in kappa_values:
            for window_size in window_sizes:
                config = base_config.copy()
                config["DATA_SOURCE"] = data_source
                config["KAPPA"] = kappa
                config["DATA_WINDOW_SIZE"] = window_size
                configs.append(config)

    # Load existing results if the file exists
    results_file = "config_files/results.json"
    results = []
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error loading existing results file: {e}")

    for idx, config in enumerate(configs):
        ds = config["DATA_SOURCE"]
        kappa = config["KAPPA"]
        window_size = config["DATA_WINDOW_SIZE"]

        print(f"\nChecking run {idx + 1}/{len(configs)} with config: {config}")

        # Skip if data window size larger than dataset
        if window_size > dataset_sizes[ds]:
            print(f"Skipping run {idx + 1} because DATA_WINDOW_SIZE {window_size} > dataset size {dataset_sizes[ds]}")
            continue

        # Check if exceeds any timed out pair
        skip_due_to_timeout = False
        for t_kappa, t_window in timed_out_pairs[ds]:
            if kappa > t_kappa and window_size > t_window:
                skip_due_to_timeout = True
                break
        if skip_due_to_timeout:
            print(f"Skipping run {idx + 1} because it exceeds a timed out configuration in both kappa and window size")
            continue

        # Skip if configuration already exists in results, but check if it timed out
        existing_result = get_existing_result(config, results)
        if existing_result:
            if 'start_time' in existing_result and 'completion_time' in existing_result:
                try:
                    start = datetime.strptime(existing_result['start_time'], "%Y-%m-%d %H:%M:%S")
                    complete = datetime.strptime(existing_result['completion_time'], "%Y-%m-%d %H:%M:%S")
                    elapsed = (complete - start).total_seconds()
                    if elapsed > 7200:
                        timed_out_pairs[ds].append((kappa, window_size))
                        print(f"Existing run for {ds} with kappa={kappa}, window={window_size} timed out ({elapsed:.2f}s), added to timed_out_pairs")
                except ValueError:
                    print(f"Error parsing times for existing run {idx + 1}")
            print(f"Configuration already exists in results, skipping run {idx + 1}")
            continue

        try:
            validate_config(config)
            run_start_time = time.time()
            single_rel_recall_list, query_precision_list, rel_individual_recalls, query_rate_ave_list, start_time_str, completion_time_str = run_ared(
                config)
            time_elapsed = time.time() - run_start_time

            # Check if run took longer than 2 hours (7200 seconds)
            if time_elapsed > 7200:
                timed_out_pairs[ds].append((kappa, window_size))
                print(f"Run {idx + 1} took {time_elapsed:.2f} seconds (> 2 hours), added to timed out pairs for {ds}")

            # Compute averages
            single_rel_recall = np.mean(single_rel_recall_list)
            avg_query_precision = np.mean(query_precision_list)
            avg_rel_individual_recalls = np.mean(rel_individual_recalls)
            avg_query_rate_ave = np.mean(query_rate_ave_list)

            result = {
                "config": config,
                "single_rel_recall": float(single_rel_recall),
                "avg_query_precision": float(avg_query_precision),
                "avg_rel_individual_recalls": float(avg_rel_individual_recalls),
                "avg_query_rate_ave": float(avg_query_rate_ave),
                "start_time": start_time_str,
                "completion_time": completion_time_str,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            results.append(result)

            # Save results after each run
            save_results(results, results_file)
        except Exception as e:
            print(f"Error in run {idx + 1}: {e}")
            continue

    plt.show()  # Display all plots at the end