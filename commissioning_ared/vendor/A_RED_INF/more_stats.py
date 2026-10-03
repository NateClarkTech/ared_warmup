import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def calculate_single_rel_recall(confusion_matrix,rel_classes,ared):
    """
    NOTE: this is tricky because we the confusion matrix doesn't record which points were queried
    However, we know the off-diagonal elements were not queried, so recall still = TP / (TP + FN)
    Args:
        confusion_matrix: nxn numpy array where rows are true labels and columns are predicted labels
    Returns:
        scalar: relevant recall
    """
    # Ensure input is a numpy array
    cm = np.array(confusion_matrix)
    n_classes = cm.shape[0]

    num_rel_pts_queried = 0.0
    num_rel_pts_streamed = 0.0

    for k,c in enumerate(rel_classes):
        i = ared.oracle.int_str_label_bidict[c]

        # True positives: diagonal element
        num_rel_pts_queried += cm[i, i]
        num_rel_pts_streamed += np.sum(cm[i, :])

    if num_rel_pts_queried == 0:
        rel_recall = 0.0
    else:
        rel_recall = num_rel_pts_queried/num_rel_pts_streamed

    return rel_recall, num_rel_pts_streamed

def calculate_precision_recall_all_classes(confusion_matrix):
    """
    Calculate precision and recall for each class from an nxn confusion matrix.

    Args:
        confusion_matrix: nxn numpy array where rows are true labels and columns are predicted labels

    Returns:
        tuple: (precision, recall) where each is an nx1 numpy array
    """
    # Ensure input is a numpy array
    cm = np.array(confusion_matrix)
    n_classes = cm.shape[0]

    # Initialize arrays for precision and recall
    precision = np.zeros((n_classes,))
    recall = np.zeros((n_classes,))

    for i in range(n_classes):
        # True positives: diagonal element
        true_positives = cm[i, i]

        # False positives: sum of column i, excluding true positives
        false_positives = np.sum(cm[:, i]) - true_positives

        # False negatives: sum of row i, excluding true positives
        false_negatives = np.sum(cm[i, :]) - true_positives

        # Precision = TP / (TP + FP)
        precision[i] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

        # Recall = TP / (TP + FN)
        recall[i] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    return precision, recall

def calc_rel_recall_query_precision(sparsity_levels, conf_matrices, rel_classes, ared,
                                    num_correct_queries, num_queries,
                                    plot_flag, GRAPH_BATCH_SIZE, NUM_POINTS_TO_PROCESS,
                                    anom_only_queries, rel_only_queries, both_a_and_r_queries, cumulative_relevants):
    """
    Old-style single plot: Query Precision + Relevant Recall on left axis,
    Query Rate + Relevant Rate on right axis.
    Adapted to accept the new function signature (extra breakdown args unused here).
    """

    import numpy as np
    import matplotlib.pyplot as plt

    conf_array = np.array(conf_matrices)
    num_correct = np.array(num_correct_queries)
    num_quer    = np.array(num_queries)

    # ------------------------------------------------------------------
    # Per-batch metrics
    # ------------------------------------------------------------------
    query_precision_list  = []
    query_rate_list       = []
    rel_rate_list         = []
    single_rel_recall_list = []

    # First batch
    cm0 = conf_matrices[0]
    q0  = num_quer[0]
    c0  = num_correct[0]

    query_precision_list.append(c0 / q0 if q0 > 0 else 0.0)
    query_rate_list.append(q0 / GRAPH_BATCH_SIZE)

    recall0, streamed0 = calculate_single_rel_recall(cm0, rel_classes, ared)
    single_rel_recall_list.append(recall0)
    rel_rate_list.append(streamed0 / GRAPH_BATCH_SIZE)

    # Subsequent batches
    for b in range(1, len(conf_matrices)):
        batch_cm  = conf_array[b] - conf_array[b - 1]
        queries_b = num_quer[b]    - num_quer[b - 1]
        correct_b = num_correct[b] - num_correct[b - 1]

        query_precision_list.append(correct_b / queries_b if queries_b > 0 else 0.0)
        query_rate_list.append(queries_b / GRAPH_BATCH_SIZE)

        recall_b, streamed_b = calculate_single_rel_recall(batch_cm, rel_classes, ared)
        single_rel_recall_list.append(recall_b)
        rel_rate_list.append(streamed_b / GRAPH_BATCH_SIZE)

    # ------------------------------------------------------------------
    # x-axis
    # ------------------------------------------------------------------
    num_batches  = len(conf_matrices)
    batch_num_pts = np.arange(GRAPH_BATCH_SIZE,
                              GRAPH_BATCH_SIZE * num_batches + 1,
                              GRAPH_BATCH_SIZE)

    # Trim to actual number of batches (guard against length mismatch)
    query_precision_list   = query_precision_list[:num_batches]
    query_rate_list        = query_rate_list[:num_batches]
    rel_rate_list          = rel_rate_list[:num_batches]
    single_rel_recall_list = single_rel_recall_list[:num_batches]

    # ------------------------------------------------------------------
    # PLOTTING  —  original single-figure style
    # ------------------------------------------------------------------
    if plot_flag:
        fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

        # Left axis: precision / recall (blue tones)
        ax1.set_xlabel("Processed Points", fontsize=12)
        ax1.set_ylabel("Precision / Recall", color='tab:blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)

        ln1 = ax1.plot(batch_num_pts, query_precision_list,
                       'o-', color='tab:blue', linewidth=2, markersize=6,
                       label="Query Precision")
        ln2 = ax1.plot(batch_num_pts, single_rel_recall_list,
                       's-', color='tab:cyan', linewidth=2, markersize=6,
                       label="Relevant Recall")

        # Right axis: rates (red tones)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Rate (per point)", color='tab:red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='tab:red')

        max_rate = max(max(query_rate_list), max(rel_rate_list)) if query_rate_list else 0.1
        ax2.set_ylim(0, max_rate * 1.25)

        ln3 = ax2.plot(batch_num_pts, query_rate_list,
                       '^-', color='tab:red', linewidth=2, markersize=6,
                       label="Query Rate")
        ln4 = ax2.plot(batch_num_pts, rel_rate_list,
                       'd--', color='tab:orange', linewidth=2, markersize=6,
                       label="Relevant Rate")

        # Combined legend inside plot (upper right, matching original)
        lns  = ln1 + ln2 + ln3 + ln4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="upper right", fontsize=10,
                   frameon=True, fancybox=False, edgecolor='gray')

        plt.title("A/RED Performance per Batch", fontsize=13, pad=12)
        fig.tight_layout()
        fig.savefig('./Figures/ared_precision_recall_figures.pdf',
                    dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.show()

    # ------------------------------------------------------------------
    # Return values  —  matches new code's return signature
    # ------------------------------------------------------------------
    return (single_rel_recall_list,
            query_precision_list,
            query_rate_list,
            rel_rate_list,
            [],   # precision_ratio_list — not computed in this style
            [])   # recall_ratio_list    — not computed in this style

def plot_relevant_class_running_accuracy(snapshots, rel_classes, GRAPH_BATCH_SIZE):
    """
    Plots the running (cumulative) accuracy for each relevant class over the data stream.

    Args:
        snapshots: list of dicts [{class_label: cumulative_accuracy}, ...], one per batch
        rel_classes: list of relevant class label strings
        GRAPH_BATCH_SIZE: int, points per batch (used for x-axis)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    num_batches = len(snapshots)
    batch_num_pts = np.arange(GRAPH_BATCH_SIZE, GRAPH_BATCH_SIZE * num_batches + 1, GRAPH_BATCH_SIZE)

    fig, ax = plt.subplots(figsize=(10, 5))

    for rel_class in rel_classes:
        accs = [snap.get(rel_class, 0.0) for snap in snapshots]
        ax.plot(batch_num_pts, accs, marker='o', linewidth=2, markersize=5, label=f"Class {rel_class}")

    ax.set_xlabel("Processed Points", fontsize=14)
    ax.set_ylabel("Running Accuracy (Cumulative TP / Total Seen)", fontsize=13)
    ax.set_title("Running Accuracy of Relevant Classes Over Data Stream", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower right', fontsize=11, frameon=True)
    ax.tick_params(axis='both', labelsize=12)

    fig.tight_layout()
    fig.savefig('./Figures/ared_relevant_class_accuracy.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Sample 3x3 confusion matrix
    sample_cm = np.array([
        [10, 2, 1],
        [1, 15, 3],
        [2, 1, 12]
    ])

    precision, recall = calculate_precision_recall_all_classes(sample_cm)

    print("Precision per class:")
    print(precision)
    print("\nRecall per class:")
    print(recall)