'''
DATA_SOURCE: Dataset to run ARED on
|- NICE: Nice dataset - synthetic data with well separated Gaussian clusters
|- MNIST: MNIST dataset
|- EMNIST: EMNIST dataset
|- PARKING_LOT
|- PARKING_LOT_DAGMM
|- PARKING_LOT_DINO
|- MVtechAD
|- MVtechAD_DINO
|- VisA
|- VisA_DINO

and ...

N_REL_CLASSES: Specified number of relevant classes
|- This is the number of classes (starting from sparsest) that are to be considered relevant
|- MNIST settings: 4 relevant classes ~1.4% of data as relevant
|- EMNIST settings: 3 relevant classes ~1% of data as relevant
|- NICE settings: 4 relevant classes ~1.4% of data as relevant
|- PARKING_LOT
|- PARKING_LOT_DAGMM
|- PARKING_LOT_DINO
|- MVtechAD
|- MVtechAD_DINO
|- VisA
|- VisA_DINO

and ...

KAPPA: Paranoia Parameter

'''
#
# DATA_SOURCE = "MNIST" # NOTE: currently multiplied by 10x to get ~130,000 samples
# KAPPA = 0.75 # MNIST
# N_REL_CLASSES = 4

# DATA_SOURCE = "MNIST_2D"
# N_REL_CLASSES = 3

# DATA_SOURCE = "EMNIST"
# N_REL_CLASSES = 10
# KAPPA = 0.1

DATA_SOURCE = "EMNIST_DINO"
N_REL_CLASSES = 10
KAPPA = 0.1

# DATA_SOURCE = "NICE"
# KAPPA = 1 # NICE
# N_REL_CLASSES = 4


# DATA_SOURCE = "PARKING_LOT_BASE"
# KAPPA = 0.75 # PL - Base
# DATA_SOURCE = "PARKING_LOT_DAGMM"
# KAPPA = 7 # PL - DAGMM

# DATA_SOURCE = "PARKING_LOT_DINO"
# KAPPA = 0.5# PL - DINO
# N_REL_CLASSES = 5

# DATA_SOURCE = "MVtechAD"
# KAPPA = 1
# N_REL_CLASSES = 6 # unused

# DATA_SOURCE = "MVtechAD_DINO"
# KAPPA = 1
# # N_REL_CLASSES = 6 # unused

# DATA_SOURCE = "VisA"
# KAPPA = 1
# N_REL_CLASSES = 6 # unused

# DATA_SOURCE = "VisA_DINO"
# KAPPA = 1
# N_REL_CLASSES = 6 # unused

'''
QS_VAR: Query Strategy Variants
|- 0: Diameter check
|- 1: Approx. Ave Single Linkage Average 
'''
QS_VAR = 1

'''
DATA_AUG_VAR: Auto data augmentation variants
|- 0, (0,): No data augmentation, empty shape
|- 1, (n, n): x4 90 degree rotation with unflattened shape. Data must be a square matrix 
'''
DATA_AUG_VAR = (0, (256,256))

'''
K_COMP_PTS: Number of points to compare to when looking for relevance
|- 1: Standard ARED
|- 2 or more: k ARED
@WARNING: must be 1 or greater
'''
K_COMP_PTS = 2

'''
NGHBHOOD_MERGE: Neighborhood Merge Variants
|- if selected, if a split is going to occur, then check if 2nd closest neighbor cluster
|- has same label as newly queried point, and merge with that cluster instead
|- NOTE: K_COMP_PTS must be >= 2 for this
|- NOTE: currently this is only implemented for top 2
|- False: No neighborhood merge
|- True: Neighborhood merge
'''
NGHBHOOD_MERGE = False

'''
SINGLETON_MERGE: Neighborhood Merge Variants
|- if selected, periodically (atm every GRAPH_BATCH_SIZE) merge singleton clusters w/ K_COMP_PTS nearest neighbor clusters
|- NOTE: K_COMP_PTS must be >= 2 for this
|- False: No singleton merge
|- True: singleton merge
'''
SINGLETON_MERGE = False

# ------------------------------------------------------------------
# NEW: threshold for “small” clusters that will be forcibly merged
# ------------------------------------------------------------------
SMALL_CLUSTER_THRESHOLD = 3      # clusters with < 3 points are merged

'''
Smart_Forgetting_Var (flag, threshold)
|- 0: no smart forgetting logic
|- 1: dumbest smart forgetting (never forget relevant points)
|- 2: dumb smart forgetting (Do not forget a point if cluster it is in is less than X percentage of the data)
|--- Threshold
|- 3: dumb smart forgetting (Do not forget a point if class is in is less than X percentage of the data)
|--- Threshold
'''
SMART_FORGETTING_VAR = (3, 0.01)

'''
window_size: size of the data_window window saved by ARED
|- int: larger window size means it remembers more data
|- WARNING: value must be larger than 0
'''
DATA_WINDOW_SIZE = 1000 # ultimately needs to be driven by anomaly ratio
if DATA_AUG_VAR[0] == 1: # Since data augmentation stores each point 4 times, increase window by 4.
    DATA_WINDOW_SIZE = DATA_WINDOW_SIZE * 4

'''
NUM_POINTS_TO_PROCESS: Number of points in dataset to process
|- -1: process all the data
|-  0 to inf: process up to that number if data is available
'''
NUM_POINTS_TO_PROCESS = -1

'''
GRAPH_BATCH_SIZE: number of points in batch for stats purposes.
'''
GRAPH_BATCH_SIZE = 100

'''
VERBOSE_FLAGS: Array of control flags to make ARED loud or quiet
|- Array, containing verbose flags for different types of messages
|- Example: VERBOSE_FLAGS = [0, 1, 2]
|- Put these numbers in the array to change which parts of ARED are very loud
|- 0: Prints dataset info and every 1000th loop of data processing
|- 1: Prints when new clusters are created and where clusters are inserted
|- 2: Prints split information, which cluster id and o_pt movements 
|- 3: Prints the l_buf cluster_id_array and abs_index_array and data windows assigned_cluster_id buffer
|- 4: Prints the forgotten_abs_index and forgotten_point_cluster_id during subspace_partition_maintenance 
|- 5: Prints information about cluster merging 
|- 6: Prints information about points recycled due to smart forgetting
'''
VERBOSE_FLAGS = [0] #example setting [1, 2] for two verbose level control flags

'''
MAKE_GRAPHS
|- True: make graphs
|- False: do not make graphs
'''
MAKE_GRAPHS = True
MAKE_EVO_GRAPHS = False
MAKE_GRAPHS_ACCURACY = True

'''
RANDOM_SEED_OFFSET
|- any integer
|- a way to vary seed sequence
'''
RANDOM_SEED_OFFSET = 25

# Imports ===================================

from Data_Stream import *
from Oracle import *
from A_REDIN import *
import sys
from Stats import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' or 'wxAgg' depending on your system
from data_visualization import *
from more_stats import *
from main_helper_functions import *
from sklearn.metrics import ConfusionMatrixDisplay

from cluster_visualization import ClusterEvolutionPlotter

if __name__ == '__main__':

        # Get data ===================================================
        X_skewed, y_w_rel, sparsity_levels,rel_classes = get_data(DATA_SOURCE,N_REL_CLASSES, VERBOSE_FLAGS, RANDOM_SEED_OFFSET)

        # Initialize Data Stream, Oracle and ARED ===================================
        data_stream = Data_Stream(X_skewed, y_w_rel)
        #data_stream.shuffle_data()
        oracle = Oracle(X_skewed, y_w_rel)
        ared = ARED(oracle, KAPPA, DATA_WINDOW_SIZE, K_COMP_PTS, QS_VAR, DATA_AUG_VAR, NGHBHOOD_MERGE, SINGLETON_MERGE, SMART_FORGETTING_VAR, VERBOSE_FLAGS)
        start_time, times, num_correct_queries, num_queries, num_clusters, num_labels, pt_dists, num_pts_searched_list, conf_matrices, \
            cumulative_relevants = set_up_stats(ared)
        buffer_fill_percents = []
        anom_only_queries = []  # only anomalous (not near a relevant cluster)
        rel_only_queries = []  # only relevant-near (not anomalous)
        both_a_and_r_queries = []  # triggered by both_a_and_r_queries conditions at once
        relevant_class_accuracy_snapshots = []  # list of dicts: {class_label: running_acc} per batch

        if MAKE_EVO_GRAPHS:
            evo_plotter = ClusterEvolutionPlotter()
            build_up_flag_evo_plotter = False # Flag used to only add the full l_buf to the subgraph once

        # Stream and Process data =========================================
        ared.process_first_point(data_stream.stream_new_data_point())

        dataset_size = len(X_skewed)
        if NUM_POINTS_TO_PROCESS == -1:
            NUM_POINTS_TO_PROCESS = data_stream.get_remaining_num_points()
        else:
            NUM_POINTS_TO_PROCESS = min(NUM_POINTS_TO_PROCESS, dataset_size-1)

        for i in range(1, NUM_POINTS_TO_PROCESS+1): # the +1 gives us an extra point, but makes the batch
            # arithmetic work out to include last batch by having last sample # end in 0

            if MAKE_EVO_GRAPHS:
                if i == 100:
                    evo_plotter.add_snapshot(ared, X_skewed, y_w_rel, "a) First 100 Points")
                elif i > 1000 and not ared.l_buf.build_up_period and not build_up_flag_evo_plotter:
                    evo_plotter.add_snapshot(ared, X_skewed, y_w_rel, "b) Labeled Buffer Full")
                    build_up_flag_evo_plotter = True

            # save and print per batch ---------------------------------------------------------------------
            if i % GRAPH_BATCH_SIZE == 0:
                j = i//GRAPH_BATCH_SIZE # count of number of batches
                print(i)
                times.append(time.time())
                # update CUMULATIVE stats...
                num_correct_queries.append(ared.num_correct_queries)
                num_queries.append(ared.num_queries)
                num_clusters.append(len(ared.subspace_partition.cluster_dict))
                num_labels.append(len(ared.subspace_partition.set_of_known_labels))
                conf_matrices.append(ared.conf_matrix.copy())
                fill_pct = ared.l_buf.data_circular_buffer.count / ared.l_buf.data_circular_buffer.size * 100
                buffer_fill_percents.append(fill_pct)


                cumulative_relevants.append(ared.cumulative_relevant_seen)
                anom_only_queries.append(ared.anom_only_queries)
                rel_only_queries.append(ared.rel_only_queries)
                both_a_and_r_queries.append(ared.both_a_and_r_queries)

                print(f"fill % of buffer: {fill_pct:.2f}%")
                if 0 in VERBOSE_FLAGS:
                    if j > 1:
                        print(f"Processing point {i}... (last {GRAPH_BATCH_SIZE} points took {times[j]- times[j-1]:.2f} seconds)")
                        print(f"# relevants in this batch: {cumulative_relevants[j-1] - cumulative_relevants[j-2]}")
                        print(f"  → anomalous queries only this batch: {anom_only_queries[j-1] - anom_only_queries[j-2]}")
                        print(f"  → relevant queries only this batch : {rel_only_queries[j-1] - rel_only_queries[j-2]}")
                        print(f"  → both trigger queries in this batch : {both_a_and_r_queries[j-1] - both_a_and_r_queries[j-2]}")
                        print(f"  → total queries this batch : {num_queries[j-1] - num_queries[j-2]}")
                        print(f"# correct queries in this batch: {num_correct_queries[j-1] - num_correct_queries[j-2]}")
                        print(f"Number of clusters: {num_clusters[j-1]}")  # Add cluster count
                        print(f"Number of labels: {num_labels[j-1]}")
                        print(f"fill % of buffer: {fill_pct:.2f}%")

                if MAKE_EVO_GRAPHS:
                    evo_plotter.plot_clusters_colored_by_label(ared, X_skewed, y_w_rel, title="Cluster Visualization by Label")

                if MAKE_GRAPHS_ACCURACY:
                    rel_acc_snapshot = {}
                    current_cm = ared.conf_matrix
                    for rel_class_str in rel_classes:
                        ci = ared.oracle.int_str_label_bidict[rel_class_str]
                        tp = current_cm[ci, ci]
                        total = np.sum(current_cm[ci, :])
                        rel_acc_snapshot[rel_class_str] = tp / total if total > 0 else 0.0
                    relevant_class_accuracy_snapshots.append(rel_acc_snapshot)

                if SINGLETON_MERGE:
                    ared.SMALL_CLUSTER_THRESHOLD = SMALL_CLUSTER_THRESHOLD  # expose the constant
                    ared.small_cluster_merge()

            # end save and print -------------------------------------------------------------

            pt_dist, num_pts_searched = ared.process_point(data_stream.stream_new_data_point())
            pt_dists.append(pt_dist)
            num_pts_searched_list.append(num_pts_searched)

        if MAKE_EVO_GRAPHS:
            evo_plotter.add_snapshot(ared, X_skewed, y_w_rel, "c) with Forgetting")
            evo_plotter.plot()
            evo_plotter.plot_dataset(X_skewed, y_w_rel)


        relevant_per_batch = np.diff(cumulative_relevants, prepend=0)
        num_correct_queries_per_batch = np.diff(num_correct_queries, prepend=0)
        num_queries_per_batch = np.diff(num_queries, prepend=0)

        # with np.printoptions(threshold=sys.maxsize):
        #     print(ared.conf_matrix)

        if MAKE_GRAPHS:
            calc_rel_recall_query_precision(sparsity_levels, conf_matrices, rel_classes, ared,
                                            num_correct_queries, num_queries,
                                            MAKE_GRAPHS, GRAPH_BATCH_SIZE, NUM_POINTS_TO_PROCESS,
                                            anom_only_queries, rel_only_queries, both_a_and_r_queries,
                                            cumulative_relevants)

        from data_visualization import plot_clusters_and_queries_over_time

        plot_clusters_and_queries_over_time(
            num_clusters=num_clusters,
            num_queries=num_queries,
            graph_batch_size=GRAPH_BATCH_SIZE,
            title=f"A/RED: Clusters & Queries - {DATA_SOURCE} (κ={KAPPA})",
            save_path=f"clusters_queries_{DATA_SOURCE}_kappa{KAPPA}.png"
        )

        if MAKE_GRAPHS_ACCURACY:
            plot_relevant_class_running_accuracy(
                relevant_class_accuracy_snapshots,
                rel_classes,
                GRAPH_BATCH_SIZE
            )

        current_time = time.time()
        time_elapsed = current_time - start_time
        print(f"Run took {time_elapsed:.2f} seconds")
        print("ARED COMPLETE")

        current_time = time.time()
        time_elapsed = current_time - start_time
        print(f"Run took {time_elapsed:.2f} seconds")
        print("ARED COMPLETE")

        # ------------------------------------------------------------------
        # FINAL PERFORMANCE SUMMARY (Precision, Recall, F1 + Relevance %)
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("FINAL PERFORMANCE SUMMARY")
        print("=" * 80)

        total_points_processed = (NUM_POINTS_TO_PROCESS if NUM_POINTS_TO_PROCESS > 0 else len(X_skewed))
        total_relevants_seen = cumulative_relevants[-1] if cumulative_relevants else 0
        relevance_percentage_seen = (total_relevants_seen / total_points_processed * 100
                                     if total_points_processed > 0 else 0.0)

        # Overall Query Precision
        total_queries_final = num_queries[-1] if num_queries else 0
        total_correct_final = num_correct_queries[-1] if num_correct_queries else 0
        overall_query_precision = (total_correct_final / total_queries_final
                                   if total_queries_final > 0 else 0.0)

        # Overall Relevant Recall + discovered classes
        final_conf_matrix = conf_matrices[-1] if conf_matrices else np.zeros((oracle.num_classes, oracle.num_classes))
        final_recall, _ = calculate_single_rel_recall(final_conf_matrix, rel_classes, ared)

        # F1-score
        if overall_query_precision + final_recall > 0:
            overall_f1 = 2 * (overall_query_precision * final_recall) / (overall_query_precision + final_recall)
        else:
            overall_f1 = 0.0

        # Discovered relevant classes
        discovered_rel_classes = []
        for rel_class_str in rel_classes:
            i = ared.oracle.int_str_label_bidict[rel_class_str]
            if final_conf_matrix[i, i] > 0:
                discovered_rel_classes.append(rel_class_str)

        num_discovered = len(discovered_rel_classes)
        num_target = len(rel_classes)

        print(f"DATA_SOURCE {DATA_SOURCE},KAPPA = {KAPPA:0.2f},N_REL_CLASSES = {N_REL_CLASSES}")
        print(f"Total points processed           : {total_points_processed}")
        print(f"Total relevant points seen       : {total_relevants_seen}")
        print(f"Relevant points percentage       : {relevance_percentage_seen:.3f}%")
        print(f"Target relevant classes          : {num_target} → {rel_classes}")
        print(f"Discovered relevant classes      : {num_discovered}/{num_target} → {discovered_rel_classes}")
        print(f"Total queries made               : {total_queries_final}")
        print(f"Total correct queries            : {total_correct_final}")
        print("")
        print(
            f"Overall Query Precision          : {overall_query_precision:.4f} ({overall_query_precision * 100:.2f}%)")
        print(f"Overall Relevant Recall          : {final_recall:.4f} ({final_recall * 100:.2f}%)")
        print(f"Overall F1-Score                 : {overall_f1:.4f} ({overall_f1 * 100:.2f}%)")

        # --- Class & Total Accuracy ---
        correct_class_counter = ared.correct_class_counter
        y_w_rel_processed = y_w_rel[:total_points_processed]

        total_correct_acc = sum(correct_class_counter.values())
        total_queried_acc = sum(1 for label, _ in y_w_rel_processed if label in correct_class_counter)
        total_accuracy = total_correct_acc / total_queried_acc if total_queried_acc > 0 else 0.0
        print("")

        print(f"Total Accuracy                   : {total_accuracy:.4f} ({total_accuracy * 100:.2f}%)")
        print(f"Per-Class Accuracy:")
        for cls in sorted(correct_class_counter.keys()):
            correct = correct_class_counter[cls]
            total = sum(1 for label, _ in y_w_rel_processed if label == cls)
            acc = correct / total if total > 0 else 0.0
            print(f"    Class {cls}: {correct}/{total} → {acc:.4f} ({acc * 100:.2f}%)")

        print("")

        forgotten_classes = ared.l_buf.forgotten_class_counter
        if len(forgotten_classes.keys()) > 0:
            print(f"Classes forgotten")
            print("Class label: Count")
        else:
            print("No classes forgotten")
        for key in forgotten_classes.keys():
            print(f"\t{key}: {forgotten_classes[key]}")

        print("=" * 80)
        #print(oracle.int_str_label_bidict)