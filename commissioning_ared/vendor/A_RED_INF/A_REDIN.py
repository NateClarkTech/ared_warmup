from Circular_Buffer import *
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree
import threading
from collections import defaultdict
import heapq
from sklearn.mixture import GaussianMixture
from FiniteBuffer import FiniteBuffer
from sklearn.neighbors import NearestNeighbors

from main import QS_VAR

"""# Subspace Partition"""
# NOTE: ARED_IN doesn't use o-pts at the moment, and ALL o_pt support has been removed
# NOTE: all point data is referred to using absolute indexes including _all the points that have been streamed - _both
# l_pts and o_pts (the Finite Buffer has an internal conversion that keeps track of just l_pts, since o_pts aren't
# streamed into it)

class Subspace_Partition:
    def __init__(self,l_buf):
        #                                                                   (l_pt_idxs)          (o_pt_idxs)
        self.cluster_dict = {} # cluster is expected to be in the format of [label, relevance, [abs_idx_l_pt], [abs_idx_o_pt], diameter]
        self.set_of_known_labels = set()
        # cluster key is the cluster's key in cluster_dict - they start at 0 and just keep incrementing as new ones are needed
        # NOTE: they are _not re-used/recycled
        self.next_cluster_key_num = 0
        self.l_buf = l_buf

    def create_new_cluster(self, label, relevance, l_pt_idxs, o_pt_idxs, QS_VAR):
        self.set_of_known_labels.add(label)
        # can use len(cluster_dict) as cluster_id of new cluster because new cluster goes at end of list
        this_cluster_key_num = self.next_cluster_key_num
        self.cluster_dict[this_cluster_key_num] = Cluster(label, relevance, l_pt_idxs, self.l_buf, this_cluster_key_num, QS_VAR)
        self.next_cluster_key_num += 1
        return this_cluster_key_num

    def remove_l_pt_from_partition(self, pt_abs_idx, pt_cluster_key):
        self.cluster_dict[pt_cluster_key].remove_l_pt_from_cluster(pt_abs_idx)
        if len(self.cluster_dict[pt_cluster_key].l_pt_idxs) == 0:
            # zero'd out points in cluster so need to remove cluster
            del self.cluster_dict[pt_cluster_key]


"""# Cluster"""
class Cluster:
    def __init__(self, label, relevance, l_pt_idxs, l_buf, cluster_key, QS_VAR=0):
        self.label = label
        self.relevance = relevance
        self.l_pt_idxs = l_pt_idxs
        self.comp_distance = 0
        self.cluster_id = cluster_key
        self.last_updated = 0  # <<< FIX >>> For aging (optional future use)

        if len(l_pt_idxs) > 1 and QS_VAR == 0:
            self.update_diameter_all(l_buf)
        elif len(l_pt_idxs) > 1 and QS_VAR == 1:
            self.update_ave_nn_dist_all(l_buf)

    def touch(self):
        import time
        self.last_updated = time.time()  # <<< FIX >>> Track activity

    def add_l_pt(self, abs_idx, l_buf, QS_VAR=0):
        self.l_pt_idxs.append(abs_idx)
        self.touch()  # <<< FIX >>> Mark as active
        if QS_VAR == 0:
            self.update_diameter_single(l_buf)
        elif QS_VAR == 1:
            self.update_ave_nn_dist_single(l_buf)

    def add_l_pt_no_comp_dist_update(self, abs_idx):
        self.l_pt_idxs.append(abs_idx)
        self.touch()  # <<< FIX >>>

    def remove_l_pt_from_cluster(self, pt_abs_idx):
        self.l_pt_idxs.remove(pt_abs_idx)

    def update_comp_distance_single(self, l_buf, QS_VAR=0):
        if QS_VAR == 0:
            self.update_diameter_single(l_buf)
        elif QS_VAR == 1:
            self.update_ave_nn_dist_single(l_buf)

    def merge_comp_distances(self, l_buf, QS_VAR):
        if QS_VAR == 0:
            self.update_diameter_all(l_buf)
        elif QS_VAR == 1:
            self.update_ave_nn_dist_all(l_buf)

    def update_diameter_single(self, l_buf):
        latest_idx = self.l_pt_idxs[-1]
        latest_pts = l_buf.get_pt_data_abs(latest_idx)
        # If no points at the latest index, nothing to do
        if latest_pts is None or len(latest_pts) == 0:
            return
        max_dist = 0.0
        # Loop over all previous indices
        for idx in self.l_pt_idxs[:-1]:
            prev_pts = l_buf.get_pt_data_abs(idx)
            if prev_pts is None:
                continue
            # Compare every latest point with every previous point at this index
            for latest_pt in latest_pts:
                for prev_pt in prev_pts:
                    dist = np.linalg.norm(latest_pt - prev_pt)
                    if dist > max_dist:
                        max_dist = dist
        # Update comp_distance if the new maximum distance is larger
        if max_dist > self.comp_distance:
            self.comp_distance = max_dist

    def update_diameter_all(self, l_buf):
        max_dist = 0.0
        # Get all points from all indices in the cluster
        all_points = []
        for idx in self.l_pt_idxs:
            pts = l_buf.get_pt_data_abs(idx)
            if pts is not None:
                all_points.extend(pts)
        # If fewer than 2 points, diameter is 0
        if len(all_points) < 2:
            self.comp_distance = 0.0
            return
        # Simple double loop over all individual points
        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                dist = np.linalg.norm(all_points[i] - all_points[j])
                if dist > max_dist:
                    max_dist = dist
        self.comp_distance = max_dist

    def update_ave_nn_dist_single(self, l_buf):
        if len(self.l_pt_idxs) < 2:
            self.comp_distance = 0.0
            return
        latest_idx = self.l_pt_idxs[-1]
        latest_pts = l_buf.get_pt_data_abs(latest_idx)
        # If no new points, nothing to update
        if latest_pts is None or len(latest_pts) == 0:
            print("PANIC: Problem with cluster storage")
            return
        # Collect all existing points (from all previous indices)
        existing_pts = []
        for idx in self.l_pt_idxs[:-1]:
            pts = l_buf.get_pt_data_abs(idx)
            if pts is not None:
                existing_pts.extend(pts)
        # If no existing points (shouldn't happen due to len check, but safe)
        if not existing_pts:
            self.comp_distance = 0.0
            print("PANIC: PANIC problem with cluster storage")
            return
        # Find the minimum distance from ANY latest point to ANY existing point
        min_dist = np.inf
        for latest_pt in latest_pts:
            for existing_pt in existing_pts:
                dist = np.linalg.norm(latest_pt - existing_pt)
                if dist < min_dist:
                    min_dist = dist
        # If all distances are infinite or zero points (edge case)
        if min_dist == np.inf:
            min_dist = 0.0
        # Incremental update of the average
        n = len(self.l_pt_idxs)  # Total number of indices (not individual points)
        self.comp_distance = (self.comp_distance * (n - 1) + min_dist) / n

    def update_ave_nn_dist_all(self, l_buf):
        if len(self.l_pt_idxs) < 2:
            self.comp_distance = 0.0
            return
        # Collect all individual points from all indices
        all_points = []
        for idx in self.l_pt_idxs:
            pts = l_buf.get_pt_data_abs(idx)
            if pts is not None:
                all_points.extend(pts)
        num_points = len(all_points)
        if num_points < 2:
            self.comp_distance = 0.0
            return
        # Compute distance from each point to every other point, find nearest for each
        total_nn_dist = 0.0
        for i in range(num_points):
            min_dist = np.inf
            for j in range(num_points):
                if i == j:
                    continue
                dist = np.linalg.norm(all_points[i] - all_points[j])
                if dist < min_dist:
                    min_dist = dist
            total_nn_dist += min_dist
        # Average of the nearest-neighbor distances
        self.comp_distance = total_nn_dist / num_points

    # AI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ------------------------------------------------------------------
    # Inside class Cluster (unchanged except for the two helpers below)
    # ------------------------------------------------------------------
    def _diameter_of_union(self, l_buf, other_idxs):
        """Return the exact diameter of *self + other_idxs* without modifying self."""
        # Collect all individual points from self's indices and the other indices
        all_points = []
        # Points from self
        for i in self.l_pt_idxs:
            pts = l_buf.get_pt_data_abs(i)
            if pts is not None:
                all_points.extend(pts)
        # Points from other_idxs
        for i in other_idxs:
            pts = l_buf.get_pt_data_abs(i)
            if pts is not None:
                all_points.extend(pts)
        # If fewer than 2 points, diameter is 0
        if len(all_points) < 2:
            return 0.0
        max_d = 0.0
        for a in range(len(all_points)):
            for b in range(a + 1, len(all_points)):
                d = np.linalg.norm(all_points[a] - all_points[b])
                if d > max_d:
                    max_d = d
        return max_d

    def _avg_nn_of_union(self, l_buf, other_idxs):
        """Return the exact avg-NN distance of *self + other_idxs* without modifying self."""
        # Collect all individual points from self's indices and the other indices
        all_points = []
        # Points from self
        for i in self.l_pt_idxs:
            pts = l_buf.get_pt_data_abs(i)
            if pts is not None:
                all_points.extend(pts)
        # Points from other_idxs
        for i in other_idxs:
            pts = l_buf.get_pt_data_abs(i)
            if pts is not None:
                all_points.extend(pts)
        num_points = len(all_points)
        if num_points < 2:
            return 0.0
        # Compute the nearest-neighbor distance for each point (brute force)
        total_nn_dist = 0.0
        for i in range(num_points):
            min_dist = np.inf
            for j in range(num_points):
                if i == j:
                    continue
                dist = np.linalg.norm(all_points[i] - all_points[j])
                if dist < min_dist:
                    min_dist = dist
            total_nn_dist += min_dist
        # Return the average of all nearest-neighbor distances
        return total_nn_dist / num_points

    def __repr__(self):
        return (
            f"Cluster(id={self.cluster_id}, "
            f"label={self.label!r}, "
            f"relevance={self.relevance}, "
            f"n_pts={len(self.l_pt_idxs)}, "
            f"l_pt_idxs={self.l_pt_idxs}, "
            f"comp_distance={self.comp_distance:.4f})"
        )

    def __str__(self):
        lines = [
            f"┌─ Cluster {self.cluster_id} ───────────────────────",
            f"│  label        : {self.label}",
            f"│  relevance    : {self.relevance}",
            f"│  comp_distance: {self.comp_distance:.4f}",
            f"│  n_pts        : {len(self.l_pt_idxs)}",
            f"│  l_pt_idxs   : {self.l_pt_idxs}",
            f"└────────────────────────────────────────",
        ]
        return "\n".join(lines)

"""# AREDIN"""

class ARED:

    def __init__(self, oracle, kappa, l_buf_size, K_COMP_PTS, QS_VAR, DATA_AUG_VAR, NGHBHOOD_MERGE, SINGLETON_MERGE, SMART_FORGETTING_VAR, VERBOSE_FLAGS):
        self.kappa = kappa
        self.K_COMP_PTS = K_COMP_PTS
        self.l_buf = FiniteBuffer(l_buf_size, .8, 2, DATA_AUG_VAR)
        self.subspace_partition = Subspace_Partition(self.l_buf)
        self.oracle = oracle
        self.num_correct_queries = 0
        self.num_queries = 0
        self.anom_only_queries = 0  # only anomalous
        self.rel_only_queries = 0  # only relevant-near
        self.both_a_and_r_queries = 0  # triggered by both_a_and_r_queries
        self.num_pts_streamed = 0
        self.abs_index = -1
        self.cumulative_relevant_seen = 0

        self.correct_class_counter = {}

        # VARIATION CONTROL FLAGS
        self.QS_VAR = QS_VAR # {0: diameter, 1: Ave Single Link Dist in Cluster
        self.DATA_AUG_VAR = DATA_AUG_VAR # {0: no augmentation, 1: x4 data rotation.}
        self.NGHBHOOD_MERGE = NGHBHOOD_MERGE
        self.SINGLETON_MERGE = SINGLETON_MERGE
        self.SMART_FORGETTING_VAR = SMART_FORGETTING_VAR
        self.verbose_flags = VERBOSE_FLAGS
        self.conf_matrix = np.zeros((oracle.num_classes, oracle.num_classes), dtype=int)

    def small_cluster_merge(self):
        """
        Called every GRAPH_BATCH_SIZE points.
        Finds every cluster with len(l_pt_idxs) < SMALL_CLUSTER_THRESHOLD
        and merges it into the *closest* cluster that has the same label.
        """
        if not hasattr(self, 'SMALL_CLUSTER_THRESHOLD'):
            # fallback – if the constant is missing in main.py
            SMALL_CLUSTER_THRESHOLD = 3
        else:
            SMALL_CLUSTER_THRESHOLD = self.SMALL_CLUSTER_THRESHOLD

        # ------------------------------------------------------------------
        # 1. Collect all clusters that are too small
        # ------------------------------------------------------------------
        small_keys = [
            key for key, cl in self.subspace_partition.cluster_dict.items()
            if len(cl.l_pt_idxs) < SMALL_CLUSTER_THRESHOLD
        ]

        if not small_keys:
            if 5 in self.verbose_flags:
                print("small_cluster_merge: no clusters below threshold")
            return

        if 5 in self.verbose_flags:
            print(f"small_cluster_merge: {len(small_keys)} clusters below threshold {SMALL_CLUSTER_THRESHOLD}")

        # ------------------------------------------------------------------
        # 2. For each small cluster, compute its centroid (only live points)
        # ------------------------------------------------------------------
        centroids = {}
        for key in small_keys:
            cl = self.subspace_partition.cluster_dict[key]
            live_pts = []
            for i in cl.l_pt_idxs:
                pts = self.l_buf.get_pt_data_abs(i)
                if pts is not None:
                    live_pts.extend(pts)
            if not live_pts:
                # the whole cluster has been forgotten → just delete it
                del self.subspace_partition.cluster_dict[key]
                continue
            centroids[key] = np.mean(live_pts, axis=0)

        # ------------------------------------------------------------------
        # 3. For every other *candidate* cluster (any size, same label) compute its centroid
        # ------------------------------------------------------------------
        candidate_centroids = {}
        for key, cl in self.subspace_partition.cluster_dict.items():
            if key in small_keys:
                continue
            live_pts = []
            for i in cl.l_pt_idxs:
                pts = self.l_buf.get_pt_data_abs(i)
                if pts is not None:
                    live_pts.extend(pts)
            if live_pts:
                candidate_centroids[key] = np.mean(live_pts, axis=0)

        # ------------------------------------------------------------------
        # 4. For each small cluster, find the nearest same-label candidate
        # ------------------------------------------------------------------
        merges = []  # list of (keep_key, merge_key) to perform
        processed = set()

        for small_key in small_keys:
            if small_key not in self.subspace_partition.cluster_dict:
                continue
            small_label = self.subspace_partition.cluster_dict[small_key].label
            small_center = centroids.get(small_key)
            if small_center is None:
                continue

            best_dist = np.inf
            best_keep = None

            for cand_key, cand_center in candidate_centroids.items():
                if cand_key in processed:
                    continue
                cand_label = self.subspace_partition.cluster_dict[cand_key].label
                if cand_label != small_label:
                    continue
                d = np.linalg.norm(small_center - cand_center)
                if d < best_dist:
                    best_dist = d
                    best_keep = cand_key

            if best_keep is not None:
                merges.append((best_keep, small_key))
                processed.add(best_keep)

        # ------------------------------------------------------------------
        # 5. Perform the merges (in any order – merge_clusters is idempotent)
        # ------------------------------------------------------------------
        for keep, merge in merges:
            if 5 in self.verbose_flags:
                print(
                    f"small_cluster_merge: merging {merge} (size {len(self.subspace_partition.cluster_dict[merge].l_pt_idxs)}) "
                    f"into {keep} (size {len(self.subspace_partition.cluster_dict[keep].l_pt_idxs)})")
            self.merge_clusters(keep, merge)

    def merge_clusters(self, keep_key, merge_key):
        if keep_key == merge_key:
            return

        if 5 in self.verbose_flags:
            print("Merging clusters:", keep_key, "←", merge_key)

        keep_cl = self.subspace_partition.cluster_dict[keep_key]
        merge_cl = self.subspace_partition.cluster_dict[merge_key]

        old_keep_size = len(keep_cl.l_pt_idxs)  # size before merge
        merge_size = len(merge_cl.l_pt_idxs)

        # 1. Move points from merge_cl → keep_cl
        keep_cl.l_pt_idxs.extend(merge_cl.l_pt_idxs)

        # 2. Update circular buffer: fix cluster_key references
        cb = self.l_buf.cluster_key_circular_buffer
        for i in range(cb.count):
            if cb.get(i) == merge_key:
                cb.set_at(i, keep_key)

        # 3. Delete merged cluster
        del self.subspace_partition.cluster_dict[merge_key]

        # 4. UPDATE comp_distance based on QS_VAR
        if self.QS_VAR == 0:  # Diameter
            # Existing diameter code (already fast enough)
            max_cross = 0.0
            # Collect only the new points from keep_cl (last merge_size points)
            keep_pts = []
            for i in keep_cl.l_pt_idxs[-merge_size:]:
                pts = self.l_buf.get_pt_data_abs(i)
                if pts is not None:
                    keep_pts.extend(pts)

            # Collect all points from merge_cl
            merge_pts = []
            for i in merge_cl.l_pt_idxs:
                pts = self.l_buf.get_pt_data_abs(i)
                if pts is not None:
                    merge_pts.extend(pts)

            # NOTE: Maybe vectorized this calculation?
            for k_pt in keep_pts:
                if k_pt is None: continue
                for m_pt in merge_pts:
                    if m_pt is None: continue
                    d = np.linalg.norm(k_pt - m_pt)
                    if d > max_cross:
                        max_cross = d
            keep_cl.comp_distance = max(keep_cl.comp_distance, merge_cl.comp_distance, max_cross)

        else:  # QS_VAR == 1 → Average Nearest-Neighbor Distance (Single-Linkage Style)
            # Goal: update keep_cl.comp_distance exactly using only new pairwise links

            if old_keep_size < 1:
                # Keep cluster was empty or singleton → just take merge cluster's avg
                keep_cl.comp_distance = merge_cl.comp_distance
            elif merge_size == 0:
                pass  # nothing to do
            else:
                # Extract live points from both_a_and_r_queries clusters
                # Collect all points from the kept old indices
                keep_pts_list = []
                for i in keep_cl.l_pt_idxs[:-merge_size]:  # old points only
                    pts = self.l_buf.get_pt_data_abs(i)
                    if pts is not None:
                        keep_pts_list.extend(pts)

                keep_pts = np.array(keep_pts_list)

                # Collect all points from the merge cluster indices
                merge_pts_list = []
                for i in merge_cl.l_pt_idxs:
                    pts = self.l_buf.get_pt_data_abs(i)
                    if pts is not None:
                        merge_pts_list.extend(pts)

                merge_pts = np.array(merge_pts_list)

                if len(merge_pts) == 0:
                    pass
                elif len(keep_pts) == 0:
                    # Only merged points survive
                    keep_cl.comp_distance = merge_cl.comp_distance
                else:
                    # Compute min distance from each merged point to any point in keep_cl
                    # Shape: (merge_size, keep_size)
                    dist_matrix = np.linalg.norm(merge_pts[:, np.newaxis, :] - keep_pts[np.newaxis, :, :], axis=2)
                    min_dists_to_keep = dist_matrix.min(axis=1)  # one per merged point

                    # These are the new nearest-neighbor links introduced by the merge
                    avg_new_link_dist = min_dists_to_keep.mean()

                    # Weighted update of running average
                    old_avg = keep_cl.comp_distance
                    old_n = old_keep_size
                    new_n = old_n + len(merge_pts)

                    # New global average NN distance
                    keep_cl.comp_distance = (old_avg * old_n + avg_new_link_dist * len(merge_pts)) / new_n

    def determine_comparison_cluster(self, data_point):
        comparison_point_info = None
        relevant_point_info = None
        #                                 0            1                2     3      4     5    6
        # Get k closest points in l_buf [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]
        k_closest_pts, num_pts_searched = self.l_buf.find_closest_pts(data_point, self.K_COMP_PTS)

        if self.NGHBHOOD_MERGE and len(k_closest_pts) > 1 and k_closest_pts[0][3] == k_closest_pts[1][3] and k_closest_pts[0][0] != k_closest_pts[1][0]:
        # i.e. if we're doing neighborhood merging and we've already processed more than one point overall,
        # and the 2 closest points have the same label,
        # and they're not in the same cluster - merge the 2 closest clusters!
            if k_closest_pts[0][0] < k_closest_pts[1][0]:
                keep_idx, merge_idx = 0, 1
            else:
                keep_idx, merge_idx = 1, 0

            keep_key = k_closest_pts[keep_idx][0]
            merge_key = k_closest_pts[merge_idx][0]

            self.merge_clusters(keep_key, merge_key)

            # update the merged point’s key so k_closest_pts is correct
            pt_info = list(k_closest_pts[merge_idx])
            pt_info[0] = keep_key
            k_closest_pts[merge_idx] = tuple(pt_info)

        comparison_point_info = k_closest_pts[0]

        # Check for relevance in k the closest points
        for pt in k_closest_pts:
            if pt[5]: # if pt[5] - relevance label - is true
                relevant_point_info = pt
                break

        # No relevant cluster in top-k, return closest overall
        return comparison_point_info, relevant_point_info, num_pts_searched, k_closest_pts
        # for k_closest_pts indices...
        #    0            1                 2     3      4     5          6
        # [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]


    def anomalous(self, data_point, cluster_key, distance):
        cluster = self.subspace_partition.cluster_dict[cluster_key]

        # Point is anomalous if its distance is greater than the cluster's diameter
        return distance * self.kappa > cluster.comp_distance

    def query(self, abs_data_index):
        (label, relevance) = self.oracle.answer_query(abs_data_index)

        return label, relevance

    def update_structs_w_new_pt(self, abs_idx, data_point, cluster_key, label, relevance):
        # do maintenance by adding pt to l_buf, forgetting from subspace_partition if necessary

        # update l_buf to have the new point
        #                      0    1      2          3               4          5
        # forgotten_pt_info = [key, label, relevance, internal_index, abs_index, data]
        forgotten_pt_info = self.l_buf.insert_pt(data_point, cluster_key, label, relevance, abs_idx)
        if forgotten_pt_info:

            # Forgetting no
            if self.SMART_FORGETTING_VAR[0] == 0:
                forgotten_pt_cluster_key = forgotten_pt_info[0]
                forgotten_pt_abs_idx = forgotten_pt_info[4]
                self.subspace_partition.remove_l_pt_from_partition(forgotten_pt_abs_idx, forgotten_pt_cluster_key)

            # Dumbest smart forgetting
            if self.SMART_FORGETTING_VAR[0] == 1:
                maintenance_complete = False

                # TODO: Make this loop safe from looping forever
                while not maintenance_complete: # and counter != self.l_buf.buffer_size:

                    forgotten_pt_cluster_key = forgotten_pt_info[0]
                    forgotten_pt_abs_idx = forgotten_pt_info[4]
                    cluster_relevance = forgotten_pt_info[2]
                    if cluster_relevance: # if cluster is relevant do not forget it.
                        cluster_key = forgotten_pt_cluster_key
                        cluster = self.subspace_partition.cluster_dict[cluster_key]
                        self.abs_index += 1
                        abs_idx = self.abs_index
                        cluster_label = cluster.label
                        data_point = forgotten_pt_info[5]
                        cluster.add_l_pt_no_comp_dist_update(abs_idx)
                        self.subspace_partition.remove_l_pt_from_partition(forgotten_pt_abs_idx, forgotten_pt_cluster_key)
                        #                      0    1      2          3               4          5
                        # forgotten_pt_info = [key, label, relevance, internal_index, abs_index, data]
                        forgotten_pt_info = self.l_buf.insert_pt(data_point, cluster_key, cluster_label, relevance, abs_idx)

                        if 6 in self.verbose_flags:
                            print(f"Point f{forgotten_pt_abs_idx} was not forgotten and was recycled as point f{abs_idx}")

                    else:
                        self.subspace_partition.remove_l_pt_from_partition(forgotten_pt_abs_idx, forgotten_pt_cluster_key)
                        maintenance_complete = True

            # Dumb smart forgetting
            if self.SMART_FORGETTING_VAR[0] == 2:
                maintenance_complete = False

                # TODO: Make this loop safe from looping forever
                while not maintenance_complete:  # and counter != self.l_buf.buffer_size:
                    forgotten_pt_cluster_key = forgotten_pt_info[0]
                    forgotten_pt_abs_idx = forgotten_pt_info[4]
                    cluster_size = self.subspace_partition.cluster_dict[forgotten_pt_cluster_key].l_pt_idxs
                    # If the size of a cluster is less than the percentage threshold of for smart forgetting
                    if len(cluster_size) < self.l_buf.buffer_size * self.SMART_FORGETTING_VAR[1]:
                        cluster_key = forgotten_pt_cluster_key
                        cluster = self.subspace_partition.cluster_dict[cluster_key]
                        self.abs_index += 1
                        abs_idx = self.abs_index
                        cluster_label = cluster.label
                        data_point = forgotten_pt_info[5]
                        cluster.add_l_pt_no_comp_dist_update(abs_idx)
                        self.subspace_partition.remove_l_pt_from_partition(forgotten_pt_abs_idx, forgotten_pt_cluster_key)
                        #                      0    1      2          3               4          5
                        # forgotten_pt_info = [key, label, relevance, internal_index, abs_index, data]
                        forgotten_pt_info = self.l_buf.insert_pt(data_point, cluster_key, cluster_label, relevance, abs_idx)

                        if 6 in self.verbose_flags:
                            print(f"Point f{forgotten_pt_abs_idx} was not forgotten and was recycled as point f{abs_idx}")
                    else:
                        self.subspace_partition.remove_l_pt_from_partition(forgotten_pt_abs_idx, forgotten_pt_cluster_key)
                        maintenance_complete = True

            # threshold based smart forgetting
            if self.SMART_FORGETTING_VAR[0] == 3:
                maintenance_complete = False

                # TODO: Make this loop safe from looping forever
                while not maintenance_complete:  # and counter != self.l_buf.buffer_size:
                    forgotten_pt_cluster_key = forgotten_pt_info[0]
                    forgotten_pt_abs_idx = forgotten_pt_info[4]
                    class_count = self.l_buf.class_count_in_buffer[forgotten_pt_info[1]]
                    # If the buffer has less than x percentage of a give class, do not forget it
                    if class_count < self.l_buf.buffer_size * self.SMART_FORGETTING_VAR[1]:
                        cluster_key = forgotten_pt_cluster_key
                        cluster = self.subspace_partition.cluster_dict[cluster_key]
                        self.abs_index += 1
                        abs_idx = self.abs_index
                        cluster_label = cluster.label
                        data_point = forgotten_pt_info[5]
                        cluster.add_l_pt_no_comp_dist_update(abs_idx)
                        self.subspace_partition.remove_l_pt_from_partition(forgotten_pt_abs_idx, forgotten_pt_cluster_key)
                        #                      0    1      2          3               4          5
                        # forgotten_pt_info = [key, label, relevance, internal_index, abs_index, data]
                        forgotten_pt_info = self.l_buf.insert_pt(data_point, cluster_key, cluster_label, relevance, abs_idx)

                        if 6 in self.verbose_flags:
                            print(f"Point f{forgotten_pt_abs_idx} was not forgotten and was recycled as point f{abs_idx}")
                    else:
                        self.subspace_partition.remove_l_pt_from_partition(forgotten_pt_abs_idx, forgotten_pt_cluster_key)
                        maintenance_complete = True

    def add_l_pt_to_existing_cl(self, abs_idx, data_point, cluster_key):
        # Done (by Ro)
        # run when we add a new labeled data point to a known cluster
        # this adds to both_a_and_r_queries l_buf and appropriate cluster in subspace partition
        # it also performs subspace partition maintenance if necessary

        if 1 in self.verbose_flags:
            print("add_l_pt:", abs_idx, cluster_key)

        this_cluster = self.subspace_partition.cluster_dict[cluster_key]

        # add point to cluster, so diameter gets updated properly
        this_cluster.add_l_pt_no_comp_dist_update(abs_idx)

        # do maintenance by adding pt to l_buf, forgetting from subspace_partition if necessary
        self.update_structs_w_new_pt(abs_idx, data_point, cluster_key,this_cluster.label, this_cluster.relevance)

        this_cluster.update_comp_distance_single(self.l_buf, self.QS_VAR)


    def split(self, data_point, data_point_idx, new_cluster_label, new_cluster_relevance, old_cluster_id):
        # Done (by Ro Not-Goodly)
        # NOTE: there are no o_pt_idxs in this implementation, so "splitting" consists only of creating a new cluster

        # make new cluster with 1 new l_pt
        this_cluster_key_num = self.subspace_partition.create_new_cluster(new_cluster_label, new_cluster_relevance, [data_point_idx], [], self.QS_VAR)

        # do maintenance by adding pt to l_buf, forgetting from subspace_partition if necessary
        self.update_structs_w_new_pt(data_point_idx, data_point, this_cluster_key_num, new_cluster_label, new_cluster_relevance)

    def process_first_point(self, data_point):
        # START QUERY
        self.num_pts_streamed += 1
        self.abs_index += 1
        data_point_abs_idx = self.abs_index
        self.num_queries += 1
        self.anom_only_queries += 1

        label, relevance = self.query(self.num_pts_streamed - 1)
        if relevance:  # the point itself is relevant
            self.cumulative_relevant_seen += 1
        # END QUERY

        # UPDATE CLUSTER Dictionary
        # Create new cluster
        cluster_key = self.subspace_partition.create_new_cluster(label, relevance, [data_point_abs_idx], [], self.QS_VAR)
        self.l_buf.insert_pt(data_point, cluster_key, label, relevance, data_point_abs_idx)
        # no maintenance required because this is the first point (so buffer's not full)
        # UPDATE CLUSTER LIST

        if 1 in self.verbose_flags:
            print("new cluster:", 0, [0])


        # update confusion matrix
        int_label = self.oracle.int_str_label_bidict[label]
        self.conf_matrix[int_label,int_label] += 1

    def process_point(self, data_point):
        self.num_pts_streamed += 1
        self.abs_index += 1
        data_point_abs_idx = self.abs_index

        # START DETERMINE COMPARISON CLUSTER
        #  0              1                2         3      4     5          6
        #  cluster_key,   pt_internal_idx, dist,     label, data, rel,       true_abs_idxz
        comp_cl_data, rel_cl_data, num_pts_searched, k_closest_pts = self.determine_comparison_cluster(data_point)
        comp_cluster_key, pt_internal_idx, distance, label, data, relevance, true_abs_idx = comp_cl_data
        comp_cluster_label = self.subspace_partition.cluster_dict[comp_cluster_key].label

        true_label, _ = self.oracle.answer_query(self.num_pts_streamed - 1)
        if comp_cluster_label == true_label:
            if true_label not in self.correct_class_counter:
                self.correct_class_counter[true_label] = 0
            self.correct_class_counter[true_label] += 1

        is_anomalous = self.anomalous(data_point, comp_cluster_key, distance)
        comp_cl_is_relevant = (rel_cl_data is not None)

        if comp_cl_is_relevant or is_anomalous:
            # ----- mutually exclusive counting -----
            if is_anomalous and comp_cl_is_relevant:
                self.both_a_and_r_queries += 1  # both_a_and_r_queries reasons fired
            elif is_anomalous:
                self.anom_only_queries += 1
            else:  # comp_cl_is_relevant only
                self.rel_only_queries += 1
            self.num_queries += 1

            # Query!
            new_pt_label, new_pt_relevant = self.query(self.num_pts_streamed - 1)

            # query is "correct" if we a) discovered a new class, or b) are querying a relevant class
            if label not in self.subspace_partition.set_of_known_labels or new_pt_relevant:
                self.num_correct_queries += 1

            label_is_same = (new_pt_label == comp_cluster_label)
            if label_is_same:  # if not a new label
                self.add_l_pt_to_existing_cl(data_point_abs_idx, data_point, comp_cluster_key)
                # this adds pt to l_buf and updates appropriate cluster in subspace partition
            else: # new pt label different from comp. cluster
                if self.K_COMP_PTS > 1 and self.NGHBHOOD_MERGE and len(k_closest_pts) > 1:
                    #                     0            1                2     3      4     5    6
                    # k closest points [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]
                    second_cluster_label = k_closest_pts[1][3]
                    if new_pt_label == second_cluster_label:
                        second_cluster_key = k_closest_pts[1][0]
                        self.add_l_pt_to_existing_cl(data_point_abs_idx, data_point, second_cluster_key)
                        # this adds pt to l_buf and updates appropriate cluster in subspace partition
                        # print("NEIGHBORHOOD MERGE!")
                    else:
                        self.split(data_point, data_point_abs_idx, new_pt_label, new_pt_relevant, comp_cluster_key)
                else:
                    self.split(data_point, data_point_abs_idx, new_pt_label, new_pt_relevant, comp_cluster_key)
            # update confusion matrix
            new_pt_label_int = self.oracle.int_str_label_bidict[new_pt_label]
            self.conf_matrix[new_pt_label_int,new_pt_label_int] += 1
        else:
            # update confusion matrix
            # this is o_pt case, so ARED doesn't actually know label, so we have to "peek" at it
            actual_new_pt_label = self.oracle.y[self.num_pts_streamed - 1][0]
            actual_new_pt_label_int = self.oracle.int_str_label_bidict[actual_new_pt_label]
            comp_cluster_label_int = self.oracle.int_str_label_bidict[comp_cluster_label]
            self.conf_matrix[actual_new_pt_label_int,comp_cluster_label_int] += 1

        actual_new_pt_rel = self.oracle.y[self.num_pts_streamed - 1][1]
        if actual_new_pt_rel:  # the point itself is relevant
            self.cumulative_relevant_seen += 1
        # DEBUG ONLY -----------------------
        return distance, num_pts_searched