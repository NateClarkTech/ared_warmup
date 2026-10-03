#!/usr/bin/env python3
"""
Improved Merging Ablation Script - Plots all 4 combinations reliably
"""

import matplotlib.pyplot as plt
import numpy as np
from main_helper_functions import get_data
from A_REDIN import ARED
from Oracle import Oracle
from Data_Stream import Data_Stream

# ====================== CONFIG ======================
# DATA_SOURCE = "MVtechAD_DINO"
# KAPPA = 1.0
# N_REL_CLASSES = 6

# DATA_SOURCE = "MNIST" # NOTE: currently multiplied by 10x to get ~130,000 samples
# KAPPA = 0.75 # MNIST
# N_REL_CLASSES = 4

DATA_SOURCE = "EMNIST_DINO"
N_REL_CLASSES = 10
KAPPA = 0.1

RANDOM_SEED_OFFSET = 42
GRAPH_BATCH_SIZE = 100
NUM_POINTS_TO_PROCESS = 25000  # Increased for better differentiation


# ====================================================

def run_with_merging(nghb_merge, singleton_merge, label):
    print(f"\n→ Running: {label}")

    try:
        X, y_w_rel, _, _ = get_data(DATA_SOURCE, N_REL_CLASSES, [0], RANDOM_SEED_OFFSET)

        data_stream = Data_Stream(X, y_w_rel)
        oracle = Oracle(X, y_w_rel)

        ared = ARED(
            oracle=oracle,
            kappa=KAPPA,
            l_buf_size=1000,
            K_COMP_PTS=2,
            QS_VAR=1,
            DATA_AUG_VAR=(0, (256, 256)),
            NGHBHOOD_MERGE=nghb_merge,
            SINGLETON_MERGE=singleton_merge,
            SMART_FORGETTING_VAR=(3, 0.01),
            VERBOSE_FLAGS=[0]
        )

        num_clusters = []
        points_processed = 0

        # First point
        ared.process_first_point(data_stream.stream_new_data_point())
        points_processed += 1
        if points_processed % GRAPH_BATCH_SIZE == 0:
            num_clusters.append(len(ared.subspace_partition.cluster_dict))

        # Main loop
        while points_processed < NUM_POINTS_TO_PROCESS and data_stream.get_remaining_num_points() > 0:
            ared.process_point(data_stream.stream_new_data_point())
            points_processed += 1

            if points_processed % GRAPH_BATCH_SIZE == 0:
                num_clusters.append(len(ared.subspace_partition.cluster_dict))
                print(f"  Processed {points_processed:5d} points | Clusters: {num_clusters[-1]}")

        print(f"  → Finished {label}: Final clusters = {num_clusters[-1] if num_clusters else 0}")
        return num_clusters

    except Exception as e:
        print(f"  ❌ Error in {label}: {e}")
        return None


# ====================== RUN ALL 4 CASES ======================
cases = [
    (False, False, "No Merging"),
    (True, False, "Neighborhood Only"),
    (False, True, "Singleton Only"),
    (True, True, "Both Merging")
]

results = {}
x_points = None

for nghb, singleton, label in cases:
    num_clusters_list = run_with_merging(nghb, singleton, label)
    if num_clusters_list is not None:
        results[label] = num_clusters_list
        if x_points is None:
            x_points = np.arange(len(num_clusters_list)) * GRAPH_BATCH_SIZE

# ====================== PLOTTING (Publication Quality) ======================

if len(results) < 4:
    print(f"\n⚠️  Only {len(results)}/{len(cases)} configurations succeeded.")

plt.figure(figsize=(13, 8))

colors = ['gray', 'tab:blue', 'tab:orange', 'tab:green']
line_styles = ['-', '-', '-', '-']
markers = ['o', 's', '^', 'D']

for (label, data), color, ls, marker in zip(results.items(), colors, line_styles, markers):
    plt.plot(x_points[:len(data)], data,
             color=color,
             linestyle=ls,
             marker=marker,
             linewidth=3.0,      # Increased line width
             markersize=6,       # Larger markers
             label=label)

plt.title(f"Effect of Merging Strategies on Cluster Count\n"
          f"{DATA_SOURCE} | κ={KAPPA} | QS_VAR=1",
          fontsize=16, pad=20)

plt.xlabel("Points Streamed", fontsize=14)
plt.ylabel("Number of Clusters", fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(fontsize=13, loc='upper left', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

# Save high-resolution version for publication
plt.savefig(f"merging_ablation_{DATA_SOURCE}_kappa{KAPPA}.png",
            dpi=400, bbox_inches='tight')
plt.savefig(f"merging_ablation_{DATA_SOURCE}_kappa{KAPPA}.pdf",
            bbox_inches='tight')   # Vector format for LaTeX

plt.show(block=True)

print("\nPublication-quality plot saved (PNG + PDF).")