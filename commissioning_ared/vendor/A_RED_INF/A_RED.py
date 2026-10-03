from Circular_Buffer import *
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree
import threading
from collections import defaultdict
import heapq
from sklearn.mixture import GaussianMixture

from main import QS_VAR

"""# Data Window"""

class Data_Window:
    def __init__(self, data_window_size = 1000):
        self.abs_idx_max = -1 # abs_idx_max is the absolute index most recent point inserted
        self.abs_idx_min = 0
        self.data_window_size = data_window_size
        self.assigned_cluster_id_window = Circular_Buffer(data_window_size)
        self.is_point_labeled_window = Circular_Buffer(data_window_size)
        self.data_in_window = Circular_Buffer(data_window_size)
        self.last_removed_cluster_id = None # Cluster ID of most recently forgotten point, the abs_idx of that point is abs_idx_min - 1

    def insert_data(self, data_point):
        self.data_in_window.append(data_point)
        self.abs_idx_max += 1
        self.abs_idx_min = max(0, self.abs_idx_max - self.data_window_size + 1)

        self.last_removed_cluster_id = self.assigned_cluster_id_window.append(None)

        self.is_point_labeled_window.append(False)

    def get_data_point(self, abs_index):
        if not (self.abs_idx_min <= abs_index <= self.abs_idx_max):
            raise IndexError(f"abs_index {abs_index} is out of the window range "
                             f"[{self.abs_idx_min}, {self.abs_idx_max})")

        dw_index = abs_index - self.abs_idx_min
        return self.data_in_window.get(dw_index)

    def update_cluster_id_at(self, abs_index, new_id):
        if not (self.abs_idx_min <= abs_index <= self.abs_idx_max):
            raise IndexError(f"abs_index {abs_index} is out of the window range "
                             f"[{self.abs_idx_min}, {self.abs_idx_max})")

        dw_index = abs_index - self.abs_idx_min
        self.assigned_cluster_id_window.set_at(dw_index, new_id)

    def updated_labeled_window(self, abs_index):
      if not (self.abs_idx_min <= abs_index <= self.abs_idx_max):
            raise IndexError(f"abs_index {abs_index} is out of the window range "
                             f"[{self.abs_idx_min}, {self.abs_idx_max})")

      dw_index = abs_index - self.abs_idx_min
      self.is_point_labeled_window.set_at(dw_index, True)

"""# Labeled Data"""

class Labeled_Data:
    def __init__(self, data_window):
        self.abs_idx_to_ld_idx_dict = {}
        self.abs_idx_array = []
        self.data_array = []
        self.cluster_id_array = []
        self.label_array = []
        self.relevance_array = []

        self.abs_idx_to_ld_idx_dict = dict()

        self.data_window = data_window

        self.l_pts_ball_tree = None
        self.ball_tree_max_idx = -1

        self._lock = threading.Lock()
        self._ball_tree_thread = None

    def add_point(self, abs_idx, data_point, cluster_id, label, relevance):
        with self._lock:
            self.abs_idx_to_ld_idx_dict[abs_idx] = len(self.data_array)
            self.abs_idx_array.append(abs_idx)
            self.data_array.append(data_point)
            self.cluster_id_array.append(cluster_id)
            self.label_array.append(label)
            self.relevance_array.append(relevance)


            # ALWAYS (mostly) BE BALLING (always start building a new ball tree when we are not building one.)
            if (self._ball_tree_thread is None and len(self.data_array) > 30) or (self._ball_tree_thread is not None and not self._ball_tree_thread.is_alive()):
                self._ball_tree_thread = threading.Thread(target=self._build_ball_tree)
                self._ball_tree_thread.start()

    def _build_ball_tree(self):
        with self._lock:
            data = np.array(self.data_array)
            self.l_pts_ball_tree = BallTree(data, leaf_size=1)
            self.ball_tree_max_idx = len(self.data_array) - 1

    def get_data(self, abs_idx):
        return self.data_array[self.abs_idx_to_ld_idx_dict[abs_idx]]

    def get_index_of_abs_idx(self, abs_idx):
        return self.abs_idx_to_ld_idx_dict[abs_idx]

    def query_top_k_clusters(self, query_point, k):
        cluster_to_min = {}  # cluster_id -> (min_dist, data_idx)

        if self.l_pts_ball_tree is not None:
            num_bt_points = self.ball_tree_max_idx + 1
            query_k = min(k, num_bt_points)  # buffer to ensure enough clusters # NOTE CHANGE LATER

            dists, indices = self.l_pts_ball_tree.query(
                [query_point], k=query_k, return_distance=True, dualtree=True, breadth_first=False, sort_results=True
            )

            for dist, idx in zip(dists[0], indices[0]):
                cluster_id = self.cluster_id_array[idx]
                if cluster_id not in cluster_to_min:
                    cluster_to_min[cluster_id] = (dist, idx)

        # Brute-force remaining points
        for i in range(self.ball_tree_max_idx + 1, len(self.data_array)):

            dist = np.linalg.norm(np.array(query_point) - np.array(self.data_array[i]))
            cluster_id = self.cluster_id_array[i]
            if cluster_id not in cluster_to_min or dist < cluster_to_min[cluster_id][0]:
                cluster_to_min[cluster_id] = (dist, i)

        # Get k clusters with smallest distances
        closest_k = heapq.nsmallest(k, cluster_to_min.items(), key=lambda x: x[1][0])

        # Return in format: list of (cluster_id, min_dist, data_idx)
        return [(cluster_id, min_dist, data_idx) for cluster_id, (min_dist, data_idx) in closest_k]

"""# Subspace Partition"""

class Subspace_Partition:
    def __init__(self):        #                                                                   (l_pt_idxs)          (o_pt_idxs)
        self.cluster_list = [] # cluster is expected to be in the format of [label, relevance, [abs_idx_l_pt], [abs_idx_o_pt], diameter]
        self.set_of_known_labels = set()
        # cluster id is the cluster's index in cluster_dict

    def create_new_cluster(self, label, relevance, l_pts, o_pts, labeled_data, QS_VAR):
        self.set_of_known_labels.add(label)
        # can use len(cluster_dict) as cluster_id of new cluster because new cluster goes at end of list
        self.cluster_list.append(Cluster(label, relevance, l_pts, o_pts, labeled_data, len(self.cluster_list), QS_VAR))

"""# Cluster"""
class Cluster:
    def __init__(self, label, relevance, l_pts, o_pts, labeled_data, cluster_id, QS_VAR = 0):
        self.label = label
        self.relevance = relevance
        self.l_pts = l_pts
        self.o_pts = o_pts
        self.comp_distance = 0  # QR_VAR=0: Diameter, QS_VAR=1: approx_nn_distance

        # cluster id is this cluster's position in Subspace_Partition.cluster_dict
        self.cluster_id = cluster_id
        if len(l_pts) > 1 and QS_VAR == 0:
            self.update_diameter(labeled_data)

        elif len(l_pts) > 1 and QS_VAR == 1:
            self.update_ave_nn_dist(labeled_data)

        elif QS_VAR == 2:
            self.update_ave_nn_dist_w_o_pts(labeled_data)

    def add_l_pt(self, abs_idx, labeled_data, QS_VAR = 0):
        self.l_pts.append(abs_idx)

        if QS_VAR == 0:
            self.update_diameter(labeled_data)
        elif QS_VAR == 1:
            self.update_ave_nn_dist(labeled_data)
        elif QS_VAR == 2:
            self.update_ave_nn_dist_w_o_pts(labeled_data)

    def add_l_pt_no_comp_dist_update(self, abs_idx):
        self.l_pts.append(abs_idx)

    def merge_comp_distances(self, labeled_data, comp_distance_to_add = 0, num_points_from_merged_cluster = 0, QS_VAR = 0):
        if QS_VAR == 0:
            self.update_diameter(labeled_data)
        if QS_VAR == 1:
            self.comp_distance = (
                                         self.comp_distance * num_points_from_merged_cluster
                                         + comp_distance_to_add * len(self.l_pts)
                                 ) / (len(self.l_pts) + num_points_from_merged_cluster)

    def add_o_pt(self, abs_idx):
        self.o_pts.append(abs_idx)

    def update_diameter(self, labeled_data):
        largest_distance = 0
        for i in range(len(self.l_pts)):
            for j in range(i):
                data_l_pt_i = labeled_data.get_data(self.l_pts[i])
                data_l_pt_j = labeled_data.get_data(self.l_pts[j])
                distance = np.linalg.norm(data_l_pt_i - data_l_pt_j)
                if largest_distance < distance:
                    largest_distance = distance
        self.comp_distance = largest_distance

    def update_ave_nn_dist(self, labeled_data):
        if len(self.l_pts) < 2:
            self.comp_distance = 0.0
            return

        # Retrieve data for all labeled points in the cluster
        data_points = np.array([labeled_data.get_data(abs_idx) for abs_idx in self.l_pts])

        # Compute the average nearest neighbor distance
        self.comp_distance = self.average_nearest_neighbor_distance(data_points)


    # for QS_VAR == 2
    # AI helper function
    def average_nearest_neighbor_distance(self, X):
        if len(X) < 2:
            raise ValueError("At least two points are required to compute nearest neighbor distances.")
        # Build KDTree for efficient nearest neighbor search
        tree = KDTree(X)
        # Query for the two nearest neighbors (including itself)
        distances, _ = tree.query(X, k=2)

        # Extract the distance to the nearest neighbor (exclude self)
        nearest_distances = distances[:, 1]

        # Compute and return the average
        return np.mean(nearest_distances)

    def update_ave_nn_dist_w_o_pts(self, labeled_data):
        data_window = labeled_data.data_window
        # make 2D array of all point data vectors
        newest_l_pt_ind = self.l_pts[-1]
        newest_l_pt_data = labeled_data.get_data(newest_l_pt_ind)
        l_and_o_pt_data = []
        for l_pt_ind in self.l_pts[:-1]:
            l_and_o_pt_data.append(labeled_data.get_data(l_pt_ind))
        for o_pt_ind in self.o_pts[:-1]:
            l_and_o_pt_data.append(data_window.get_data_point(o_pt_ind))
        l_and_o_pt_data.append(newest_l_pt_data)
        l_and_o_pt_data = np.array(l_and_o_pt_data)
        self.comp_distance = self.average_nearest_neighbor_distance(l_and_o_pt_data)

"""# ARED"""

class ARED:
    # CHANGE BIGLY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
    def __init__(self, oracle, kappa=1.0, data_window_size=1000, k_comparison_clusters = 1, QS_VAR = 0, REL_PROC_VAR = 0, SM_VAR=0, VERBOSE_FLAGS = []):
        self.kappa = kappa
        self.data_window = Data_Window(data_window_size)
        self.k_comparison_clusters = k_comparison_clusters
        self.labeled_data = Labeled_Data(self.data_window)
        self.subspace_partition = Subspace_Partition()
        self.oracle = oracle
        self.num_queries = 0 # NEW!  - NEED TO KEEP TRACK OF THESE IN HERE
        self.num_pts_streamed = 0 # NEW!  - NEED TO KEEP TRACK OF THESE IN HERE

        # VARIATION CONTROL FLAGS
        self.QS_VAR = QS_VAR # {0: diameter, 1: Ave Single Link Dist in Cluster
        self.REL_PROC_VAR = REL_PROC_VAR
        self.SM_VAR = SM_VAR
        self.verbose_flags = VERBOSE_FLAGS


    def process_first_point(self, data_point):
        # CHANGE BIGLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Insert data point into data_window
        self.data_window.insert_data(data_point)
        data_point_abs_idx = self.data_window.abs_idx_max

        # START QUERY
        label, relevance = self.query(data_point_abs_idx)
        # END QUERY

        cluster_id = 0

        # Update data_window.assigned_cluster_id_window
        self.data_window.update_cluster_id_at(0, 0)

        # Create new cluster
        self.labeled_data.add_point(data_point_abs_idx, data_point, cluster_id, label, relevance) #cluster_id = 0
        self.subspace_partition.create_new_cluster(label, relevance, [data_point_abs_idx], [], self.labeled_data, self.QS_VAR)

        if 1 in self.verbose_flags:
            print("new cluster:", 0, [0])

    def merge_clusters(self, cluster_a, cluster_b):
        if cluster_a.cluster_id == cluster_b.cluster_id:
            return cluster_a.cluster_id

        # find the cluster with the smaller id
        cluster_with_smaller_id = cluster_a
        cluster_with_larger_id = cluster_b
        if cluster_b.cluster_id < cluster_a.cluster_id:
            cluster_with_larger_id = cluster_a
            cluster_with_smaller_id = cluster_b

        if 5 in self.verbose_flags:
            print("Merging clusters:", cluster_with_smaller_id.cluster_id, cluster_with_larger_id.cluster_id)

        for abs_l_pt_idx_in_larger_id_cluster in cluster_with_larger_id.l_pts:
            #update data window
            if abs_l_pt_idx_in_larger_id_cluster > self.data_window.abs_idx_min:
                self.data_window.update_cluster_id_at(abs_l_pt_idx_in_larger_id_cluster, cluster_with_smaller_id.cluster_id)

            idx = self.labeled_data.get_index_of_abs_idx(abs_l_pt_idx_in_larger_id_cluster)
            self.labeled_data.cluster_id_array[idx] = cluster_with_smaller_id.cluster_id

            cluster_with_smaller_id.add_l_pt_no_comp_dist_update(abs_l_pt_idx_in_larger_id_cluster)

        for abs_o_pt_idx_in_larger_id_cluster in cluster_with_larger_id.o_pts:
            self.data_window.update_cluster_id_at(abs_o_pt_idx_in_larger_id_cluster, cluster_with_smaller_id.cluster_id)
            cluster_with_smaller_id.add_o_pt(abs_o_pt_idx_in_larger_id_cluster)

        if QS_VAR == 0:
            #merge_comp_distances(self, l_buf, comp_distance_to_add = 0, num_points_from_merged_cluster = 0, QS_VAR = 0):
            cluster_with_smaller_id.merge_comp_distances(self.labeled_data)

        elif QS_VAR == 1:
            cluster_with_smaller_id.merge_comp_distances(self.labeled_data, cluster_with_larger_id.comp_distance, len(cluster_with_larger_id.l_pts), self.QS_VAR)

        # replace old cluster with a cluster with an empty shell
        self.subspace_partition.cluster_list[cluster_with_larger_id.cluster_id] = Cluster(None, False, [], [], self.labeled_data, cluster_with_larger_id.cluster_id)

        return cluster_with_smaller_id.cluster_id

    def determine_comparison_cluster(self, data_point):
        comparison_cluster_id = None

        # Get top-k closest clusters (cluster_id, distance, idx of point in self.l_buf)
        top_k = self.labeled_data.query_top_k_clusters(data_point, self.k_comparison_clusters)

        comparison_cluster_id = top_k[0][0]

        if len(top_k) > 1 and self.subspace_partition.cluster_list[top_k[0][0]].label == self.subspace_partition.cluster_list[top_k[1][0]].label:
            comparison_cluster_id = self.merge_clusters(self.subspace_partition.cluster_list[top_k[0][0]], self.subspace_partition.cluster_list[top_k[1][0]])

        #print(top_k)

        # Check for relevance in top-k
        relevant_clusters = [
            (cluster_id, dist)
            for cluster_id, dist, lda_idx in top_k
                if self.subspace_partition.cluster_list[cluster_id].relevance > 0
        ]

        # Prefer a relevant cluster if found
        if relevant_clusters and self.k_comparison_clusters > 1:
            # Return the closest relevant one
            return min(relevant_clusters, key=lambda x: x[1])

        # No relevant cluster in top-k, return closest overall
        return comparison_cluster_id, top_k[0][1]


    def anomalous(self, data_point, cluster_id, distance):
        cluster = self.subspace_partition.cluster_list[cluster_id]

        # Point is anomalous if its distance is greater than the cluster's diameter
        return distance * self.kappa > cluster.comp_distance

    def query(self, abs_data_index):
        self.data_window.updated_labeled_window(abs_data_index)
        # return (label, relevance) from oracle
        return self.oracle.answer_query(abs_data_index)


    # ran when we add a new o_pt to a cluster
    def add_o_pt(self, abs_idx, cluster_id):

        if 1 in self.verbose_flags:
            print("add_o_pt:", abs_idx, cluster_id)

        cluster = self.subspace_partition.cluster_list[cluster_id]
        cluster.add_o_pt(abs_idx)

        # update data_window.assigned_cluster_id_window
        self.data_window.update_cluster_id_at(abs_idx, cluster_id)


    # ran when we add a new labeled data point to a known cluster
    def add_l_pt(self, abs_idx, data_point, cluster_id):

        if 1 in self.verbose_flags:
            print("add_l_pt:", abs_idx, cluster_id)

        # update cluster in subspace partition
        cluster = self.subspace_partition.cluster_list[cluster_id]

        # get label and relevance
        label = cluster.label
        relevance = cluster.relevance

        # update data_window.assigned_cluster_id_window
        self.data_window.update_cluster_id_at(abs_idx, cluster_id)

        # update l_buf to have the new point
        self.labeled_data.add_point(abs_idx, data_point, cluster_id, label, relevance)

        # add point to cluster, so diameter gets updated properly
        cluster.add_l_pt(abs_idx, self.labeled_data, self.QS_VAR)


    def split(self, data_point, data_point_idx, new_cluster_label, new_cluster_relevance, old_cluster_id):
        if self.SM_VAR == 0: # VORONOI Splitting method
            new_cluster_id = len(self.subspace_partition.cluster_list)
            self.labeled_data.add_point(data_point_idx, data_point, new_cluster_id, new_cluster_label, new_cluster_relevance)
            self.data_window.update_cluster_id_at(data_point_idx, new_cluster_id)
            self.subspace_partition.create_new_cluster(new_cluster_label, new_cluster_relevance, [data_point_idx], [], self.labeled_data, self.QS_VAR)

            if 1 in self.verbose_flags:
                print("new cluster:", new_cluster_id, [data_point_idx])

            # array to hold o_pt indexes during the split process
            new_cluster_o_pts_abs_inds = []
            old_cluster_o_pts_abs_inds = []

            # get o_pt indices
            o_pts_abs_inds_to_split = self.subspace_partition.cluster_list[old_cluster_id].o_pts

            if (len(o_pts_abs_inds_to_split) == 0):
                return

            # get l_pt indices
            l_pt_inds = self.subspace_partition.cluster_list[old_cluster_id].l_pts

            # o_pt_index is an abs_idx
            for o_pt_index in o_pts_abs_inds_to_split:
                o_pt = self.data_window.get_data_point(o_pt_index)

                # find the closest labeled point in the exisiting cluster
                distance_to_existing = min([
                    np.linalg.norm(o_pt - self.labeled_data.get_data(l_pt_index))
                    for l_pt_index in l_pt_inds
                ])

                # get the distance to the labeled point in the new cluster
                distance_to_new = np.linalg.norm(o_pt - data_point)

                # put the o_pt in the closest cluster of the two
                if distance_to_existing < distance_to_new:
                    old_cluster_o_pts_abs_inds.append(o_pt_index)
                else:
                    #print(distance_to_new, distance_to_existing, o_pt_index)
                    new_cluster_o_pts_abs_inds.append(o_pt_index)

                    # update the data window so the assigned_label_id_window is correct for window maintenance later
                    self.data_window.update_cluster_id_at(o_pt_index, new_cluster_id)

            if 2 in self.verbose_flags:
                print("Split :")
                print("old_cluster_id w/ o_pt_idxs:", old_cluster_id, old_cluster_o_pts_abs_inds)
                print("new_cluster_id w/ o_pt_idxs:", new_cluster_id, new_cluster_o_pts_abs_inds)

            # put the o_pt_idxs in their correct cluster
            self.subspace_partition.cluster_list[new_cluster_id].o_pts = new_cluster_o_pts_abs_inds # update o_pt_idxs new_cluster
            self.subspace_partition.cluster_list[old_cluster_id].o_pts = old_cluster_o_pts_abs_inds # update o_pt_idxs old_cluster

        # if self.SM_VAR == 1: # 2 GMM splitting method
        #
        #     old_cluster_l_pts_idxes = self.subspace_partition.cluster_dict[old_cluster_id].l_pt_idxs
        #     old_cluster_l_pts = [self.l_buf.get_data(l_pt_idx) for l_pt_idx in old_cluster_l_pts_idxes]
        #     l_pt_idxs = old_cluster_l_pts.append(data_point)
        #
        #     # get o_pt indices
        #     o_pts_abs_inds_to_split = self.subspace_partition.cluster_dict[old_cluster_id].o_pt_idxs
        #     o_pt_idxs = [self.data_window.get_data_point(o_pt_idx) for o_pt_idx in o_pts_abs_inds_to_split]
        #
        #     all_pts = list(set(l_pt_idxs) | set(o_pt_idxs))
        #
        #     # Make 2 GMM using the labeled data points from each cluster
        #     gm = GaussianMixture(n_components=2).fit(all_pts)
        #
        #     # Predict where the unlabeled points belong based on the labeled data
        #     for o_pt_index in o_pt_idxs:
        #         o_pt =


    def relevance_processing(self, new_cluster_id):

        if self.REL_PROC_VAR == 0:
            pass
        elif self.REL_PROC_VAR == 1:
            current_rel_cluster = self.subspace_partition.cluster_list[new_cluster_id]
            additional_cls_to_process = [current_rel_cluster]

            while len(additional_cls_to_process) > 0:
                current_rel_cluster = additional_cls_to_process.pop()
                if len(current_rel_cluster.o_pts) > 0:
                    # here we need to check whether the o_pt_idxs that have been assigned to this cluster are still part of the relevant class
                    current_cluster_label = current_rel_cluster.label
                    # currently this is a new cluster with only 1 l_pt
                    # so sort o_pt_idxs based on distance from that 1 pt, and query in that order, splitting when new label is encountered
                    o_pts_abs_inds = current_rel_cluster.o_pts
                    o_pts_data = []
                    dists = []
                    for i, o_pt_index in enumerate(o_pts_abs_inds):
                        o_pts_data.append(self.data_window.get_data_point(o_pt_index))
                        dists.append(np.linalg.norm(o_pts_data[i] - current_rel_cluster.l_pts[-1])) # gets dist from most recent l_pt in cluster

                    # sort everything by dists
                    sorted_triplets = sorted(zip(dists, o_pts_abs_inds, o_pts_data))

                    for dist, o_pt_abs_ind, o_pt_data in sorted_triplets:
                        new_pt_label, new_pt_relevance = self.query(o_pt_abs_ind)
                        current_rel_cluster.o_pts.remove(o_pt_abs_ind)
                        if new_pt_label == current_cluster_label:
                            # change o_pt to l_pt in current cluster
                            self.data_window.update_cluster_id_at(o_pt_abs_ind, current_rel_cluster.cluster_id)
                            self.data_window.updated_labeled_window(o_pt_abs_ind)
                            self.add_l_pt(o_pt_abs_ind,o_pt_data,current_rel_cluster.cluster_id)

                        else:
                            self.split(o_pt_data,o_pt_abs_ind,new_pt_label,new_pt_relevance, current_rel_cluster.cluster_id)
                            newest_cluster = self.subspace_partition.cluster_list[-1]
                            self.data_window.update_cluster_id_at(o_pt_abs_ind, newest_cluster.cluster_id)
                            self.data_window.updated_labeled_window(o_pt_abs_ind)

                            if newest_cluster.relevance:
                                # add new cluster to additional_cls_to_process
                                additional_cls_to_process.append(newest_cluster)
                            # add split old (originally relevant) cluster to additional_cls_to_process
                            additional_cls_to_process.append(current_rel_cluster)

                            break


    # Removing forgotten o_pt_idxs from the subspace partition
    def subspace_partition_maintenance(self, forgotten_abs_idx, forgotten_point_cluster_id):

        cluster = self.subspace_partition.cluster_list[forgotten_point_cluster_id]

        if 4 in self.verbose_flags:
            print(forgotten_abs_idx, forgotten_point_cluster_id)

        cluster.o_pts.remove(forgotten_abs_idx)

    def process_point(self, data_point):
    # CHANGE BIGLY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if 3 in self.verbose_flags:
            print("labeled id array:", self.labeled_data.cluster_id_array)
            print("labeled abs array:", self.labeled_data.abs_idx_array)
            print("data window assigned id:", self.data_window.assigned_cluster_id_window.get_array())
        # THIS STUFF NEEDS TO GO IN A MAINTENANCE FUNCTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        is_forgotten_point_labeled = self.data_window.is_point_labeled_window.get(0) # the '0' here is the index of the oldest element in data window

        self.data_window.insert_data(data_point)
        data_point_abs_idx = self.data_window.abs_idx_max

        forgotten_abs_idx = self.data_window.abs_idx_min - 1
        forgotten_pt_cluster_id = self.data_window.last_removed_cluster_id

        # if forgotten_pt_cluster_id is NOT None (ie a point has been forgotten) do maintenance
        if forgotten_pt_cluster_id != None and not is_forgotten_point_labeled:
            self.subspace_partition_maintenance(forgotten_abs_idx, forgotten_pt_cluster_id)

        # START DETERMINE COMPARISON CLUSTER

        comp_cluster_id, distance = self.determine_comparison_cluster(data_point)
        comp_cluster_relevant = self.subspace_partition.cluster_list[comp_cluster_id].relevance
        comp_cluster_label = self.subspace_partition.cluster_list[comp_cluster_id].label

        is_anomalous = self.anomalous(data_point, comp_cluster_id, distance)

        if not comp_cluster_relevant and not is_anomalous:
            self.add_o_pt(data_point_abs_idx, comp_cluster_id)

        else:
            # Query!
            new_pt_label, new_pt_relevant = self.query(data_point_abs_idx)
            label_is_same = (new_pt_label == comp_cluster_label)
            if label_is_same:  # if not a new label
                self.add_l_pt(data_point_abs_idx, data_point, comp_cluster_id)
            else:
                self.split(data_point, data_point_abs_idx, new_pt_label, new_pt_relevant, comp_cluster_id)
                if new_pt_relevant:
                    self.relevance_processing(len(self.subspace_partition.cluster_list) - 1)

        # POINT PROCESSED