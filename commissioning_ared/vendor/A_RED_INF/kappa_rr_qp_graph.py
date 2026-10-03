"""
kappa_sweep_pr_curve.py

Runs ARED on MVtechAD_DINO for each kappa in a given list, collects
overall query precision, relevant recall, and number of discovered
relevant classes, then plots a Query Precision vs Relevant Recall
curve with F1 contours and boxed kappa labels — matching the style
of Figure 2 from the first paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.use('TkAgg')

from Data_Stream import *
from Oracle import *
from A_REDIN import *
from more_stats import calculate_single_rel_recall
from main_helper_functions import *

# ---------------------------------------------------------------
# CONFIG  —  edit these to match your standard main.py settings
# ---------------------------------------------------------------
DATA_SOURCE         = "MVtechAD_DINO"
N_REL_CLASSES       = 6          # unused by MVtechAD_DINO but required by get_data
QS_VAR              = 1
DATA_AUG_VAR        = (0, (256, 256))
K_COMP_PTS          = 2
NGHBHOOD_MERGE      = True
SINGLETON_MERGE     = True
SMALL_CLUSTER_THRESHOLD = 3
SMART_FORGETTING_VAR = (3, 0.01)
DATA_WINDOW_SIZE    = 1000
NUM_POINTS_TO_PROCESS = -1
GRAPH_BATCH_SIZE    = 100
VERBOSE_FLAGS       = []          # silent during sweep
RANDOM_SEED_OFFSET  = 25

KAPPA_LIST = [0.25, 0.5, 0.625, 0.75, 0.825, 1.0, 1.125, 1.5]


# ---------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------
def run_single_kappa(kappa):
    print(f"\n{'='*60}")
    print(f"  Running kappa = {kappa}")
    print(f"{'='*60}")

    X_skewed, y_w_rel, sparsity_levels, rel_classes = get_data(
        DATA_SOURCE, N_REL_CLASSES, VERBOSE_FLAGS, RANDOM_SEED_OFFSET
    )

    data_stream = Data_Stream(X_skewed, y_w_rel)
    oracle      = Oracle(X_skewed, y_w_rel)
    ared        = ARED(
        oracle, kappa, DATA_WINDOW_SIZE, K_COMP_PTS, QS_VAR,
        DATA_AUG_VAR, NGHBHOOD_MERGE, SINGLETON_MERGE,
        SMART_FORGETTING_VAR, VERBOSE_FLAGS
    )

    start_time, times, num_correct_queries, num_queries, num_clusters, \
        num_labels, pt_dists, num_pts_searched_list, conf_matrices, \
        cumulative_relevants = set_up_stats(ared)

    ared.process_first_point(data_stream.stream_new_data_point())

    n_pts = data_stream.get_remaining_num_points() \
        if NUM_POINTS_TO_PROCESS == -1 \
        else min(NUM_POINTS_TO_PROCESS, len(X_skewed) - 1)

    for i in range(1, n_pts + 1):
        if i % GRAPH_BATCH_SIZE == 0:
            times.append(time.time())
            num_correct_queries.append(ared.num_correct_queries)
            num_queries.append(ared.num_queries)
            num_clusters.append(len(ared.subspace_partition.cluster_dict))
            num_labels.append(len(ared.subspace_partition.set_of_known_labels))
            conf_matrices.append(ared.conf_matrix.copy())
            cumulative_relevants.append(ared.cumulative_relevant_seen)

            if SINGLETON_MERGE:
                ared.SMALL_CLUSTER_THRESHOLD = SMALL_CLUSTER_THRESHOLD
                ared.small_cluster_merge()

            if i % 1000 == 0:
                print(f"  {i} / {n_pts} points processed...")

        pt_dist, num_pts_searched = ared.process_point(
            data_stream.stream_new_data_point()
        )
        pt_dists.append(pt_dist)
        num_pts_searched_list.append(num_pts_searched)

    # --- collect final stats ---
    final_cm = conf_matrices[-1] if conf_matrices else \
        np.zeros((oracle.num_classes, oracle.num_classes))

    total_queries  = num_queries[-1]  if num_queries  else 0
    total_correct  = num_correct_queries[-1] if num_correct_queries else 0
    query_precision = total_correct / total_queries if total_queries > 0 else 0.0

    rel_recall, _ = calculate_single_rel_recall(final_cm, rel_classes, ared)

    # discovered = at least one correct query for that class
    num_discovered = sum(
        1 for c in rel_classes
        if final_cm[ared.oracle.int_str_label_bidict[c],
                    ared.oracle.int_str_label_bidict[c]] > 0
    )
    num_target = len(rel_classes)

    print(f"  kappa={kappa:.2f}  precision={query_precision:.3f}  "
          f"recall={rel_recall:.3f}  discovered={num_discovered}/{num_target}")

    return query_precision, rel_recall, num_discovered, num_target


# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------
def plot_pr_curve(kappas, precisions, recalls, discovered_counts,
                  num_target, title=None, save_path=None):

    fig, ax = plt.subplots(figsize=(7, 6))

    # --- F1 contours ---
    recall_grid  = np.linspace(0.01, 1.0, 300)
    precision_grid = np.linspace(0.01, 1.0, 300)
    R, P = np.meshgrid(recall_grid, precision_grid)
    F1   = 2 * P * R / (P + R)

    f1_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cs = ax.contour(R, P, F1, levels=f1_levels,
                    colors='gray', linestyles='dashed', linewidths=0.8, alpha=0.6)
    ax.clabel(cs, fmt={v: f'f1={v}' for v in f1_levels},
              fontsize=7, inline=True, inline_spacing=2)

    # --- PR curve ---
    color = '#8B0057'   # dark magenta, matching the "A/RED Baseline" line in the paper
    ax.plot(recalls, precisions, '-', color=color, linewidth=2, zorder=3)
    ax.scatter(recalls, precisions, color=color, s=60, zorder=4)

    # --- boxed labels (kappa value above each point) ---
    # --- boxed labels (kappa value above each point) ---
    used_positions = []
    point_offsets = []
    for r, p in zip(recalls, precisions):
        offset = (0, 14)
        for ur, up, uo in used_positions:
            if abs(r - ur) < 0.05 and abs(p - up) < 0.05:
                offset = (0, 28) if uo == (0, 14) else (0, 14)
        used_positions.append((r, p, offset))
        point_offsets.append(offset)

    for kappa, r, p, disc, offset in zip(kappas, recalls, precisions, discovered_counts, point_offsets):
        ax.annotate(
            f"κ={kappa}\n{disc}/{num_target}",
            xy=(r, p),
            xytext=offset,
            textcoords='offset points',
            fontsize=8,
            fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                      edgecolor=color, linewidth=1.5),
            zorder=5
        )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Relevant Recall', fontsize=13)
    ax.set_ylabel('Query Precision', fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.2)

    plot_title = title or f"Query Precision vs Relevant Recall\n{DATA_SOURCE}"
    ax.set_title(plot_title, fontsize=13)

    # legend entry showing what the box number means
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color, linewidth=2,
               label=f'A/RED — {DATA_SOURCE}\n(box = discovered rel. classes / {num_target})')
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='lower left',
              frameon=True, edgecolor='black')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == '__main__':
    kappas          = KAPPA_LIST
    precisions      = []
    recalls         = []
    discovered_list = []
    num_target      = None

    for kappa in kappas:
        prec, rec, disc, n_target = run_single_kappa(kappa)
        precisions.append(prec)
        recalls.append(rec)
        discovered_list.append(disc)
        num_target = n_target

    print("\n--- Sweep complete ---")
    for k, p, r, d in zip(kappas, precisions, recalls, discovered_list):
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        print(f"  kappa={k:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  disc={d}/{num_target}")

    plot_pr_curve(
        kappas, precisions, recalls, discovered_list,
        num_target=num_target,
        save_path='./Figures/kappa_sweep_pr_curve.pdf'
    )