from Circular_Buffer import Circular_Buffer
from sklearn.neighbors import BallTree
from sklearn.metrics import DistanceMetric
import threading
import numpy as np
import bisect

# READ AND ACT UPON.
# NOTE by Ro: There are 3 types of indexes here, and we should improve the names:
# indexes into circular buffer, ranging from 0 to size of circular buffer - 1
# henceforth "cb_idxs"
# indexes based on total number of l_pts, ever.  These are the "internal abs" at the moment
# henceforth "abs_l_idxs"
# indexes based on total number of l AND o_pts, ever.  These are "true abs" at the moment
# henceforth "abs_l_o_idxs"
# the max internal abs index should correspond to the max circ buff index
# at the moment, ball tree indexes in calls are ball tree indexes, stuff stored in class is l_idxs
# henceforth "bt_idxs"

class BallTreeWithIndexes(BallTree):
    def __init__(self, X, min_index, max_index, leaf_size = 40, metric: str | DistanceMetric = "minkowski"):
        super().__init__(X, leaf_size=leaf_size, metric=metric)
        self.max_index = max_index
        self.min_index = min_index
        self.length = max_index - min_index

class FiniteBuffer:
    def __init__(self, buffer_size: int, ball_tree_ratio: float = 0.8, num_ball_trees: int = 2, DATA_AUG_VAR = (0, (0,))):
        self.data_circular_buffer = Circular_Buffer(buffer_size)
        self.label_circular_buffer = Circular_Buffer(buffer_size)
        self.relevance_circular_buffer = Circular_Buffer(buffer_size)
        self.cluster_key_circular_buffer = Circular_Buffer(buffer_size)
        self.true_abs_idx_circular_buffer = Circular_Buffer(buffer_size)

        self.class_count_in_buffer = {} # entries should be in the form {<Class name: string>: <count: int>}
        self.forgotten_class_counter = {} # dict that counts how many times a class has been completely forgotten from {<Class>}

        self.buffer_size = buffer_size

        self.num_ball_trees = num_ball_trees
        self.ball_tree_interval = int(self.buffer_size * ball_tree_ratio)
        self.non_overlap_interval = np.ceil(self.buffer_size * (1 - ball_tree_ratio) / self.num_ball_trees)
        self.ball_trees = []

        self.num_ball_trees_completed = 0

        self.max_internal_abs_idx = -1
        self.min_internal_abs_idx = 0

        self.build_up_period = True

        # --- Data augmentation flag ---
        self.DATA_AUG_VAR = DATA_AUG_VAR

        if DATA_AUG_VAR[0] == 1:
            assert self.buffer_size % 4 == 0, "If using data augmentation 1 buffer size should be divisible by 4"

        # --- threading-related attributes ---
        self._tree_build_lock = threading.Lock()
        self._building_tree = False  # flag so we don’t start multiple builds

    def insert_pt(self, X, cluster_key, label, relevance, true_abs_idx):

        forgotten_pt_info = None
        build_ball_tree = False

        if label in self.class_count_in_buffer:
            self.class_count_in_buffer[label] += 1
        else:
            self.class_count_in_buffer[label] = 1

        # Check if buffer is fulls
        if self.data_circular_buffer.is_full():
            # forgotten point in form: [key, label, relevance, index, abs_index, data]
            forgotten_pt_info = self._forget_pt()
            self.min_internal_abs_idx += 1
            self.class_count_in_buffer[forgotten_pt_info[1]] -= 1

            # Keep track of times a class was forgotten completely
            if self.class_count_in_buffer[forgotten_pt_info[1]] == 0:
                if forgotten_pt_info[1] not in self.forgotten_class_counter:
                    self.forgotten_class_counter[forgotten_pt_info[1]] = 0
                self.forgotten_class_counter[forgotten_pt_info[1]] += 1

        self.data_circular_buffer.append(X)
        self.label_circular_buffer.append(label)
        self.relevance_circular_buffer.append(relevance)
        self.cluster_key_circular_buffer.append(cluster_key)
        self.true_abs_idx_circular_buffer.append(true_abs_idx)
        self.max_internal_abs_idx += 1

        # if data augmentation method 1 (x4 data samples with 90 degree rotations)
        if self.DATA_AUG_VAR[0] == 1:
            shape = self.DATA_AUG_VAR[1]

            img = X.reshape(shape)

            # Rotate 90°, 180°, 270°
            for k in range(1, 4):
                rot_img = np.rot90(img, k=k)
                rot_flat = rot_img.reshape(-1)

                self.data_circular_buffer.append(rot_flat)
                self.label_circular_buffer.append(label)
                self.relevance_circular_buffer.append(relevance)
                self.cluster_key_circular_buffer.append(cluster_key)
                self.true_abs_idx_circular_buffer.append(true_abs_idx)
                self.max_internal_abs_idx += 1

                if self.data_circular_buffer.is_full():
                    self.min_internal_abs_idx += 1

        if len(self.ball_trees) != 0 and self.ball_trees[0].min_index < self.min_internal_abs_idx:
            self.ball_trees.pop(0)

            # if we have a btree     and # ball trees < max # btree                 and max_abs_idx is greater than max index of newest btree + non overlap interval
        if len(self.ball_trees) != 0 and len(self.ball_trees) < self.num_ball_trees and self.max_internal_abs_idx >= self.ball_trees[-1].max_index + self.non_overlap_interval and not self.build_up_period:
            build_ball_tree = True
             # If we have no btrees    and we have as many points as the btree interval
        elif len(self.ball_trees) == 0 and self.ball_tree_interval <= self.max_internal_abs_idx:
            build_ball_tree = True

        elif self.build_up_period and (len(self.ball_trees) == 0 or self.ball_trees[0].max_index < self.max_internal_abs_idx):
            build_ball_tree = True

        # if ball tree is invalid, forget it and start building new tree
        if build_ball_tree and not self._building_tree:
            self._building_tree = True
            threading.Thread(target=self._build_new_tree, daemon=True).start()
        return forgotten_pt_info # (key, relevance, index, internal_abs_index, true_abs_idx)

    '''
    returns the information about the oldest remembered streamed point
    '''
    def _forget_pt(self):
        forgotten_pt_data = self.data_circular_buffer.get(0)
        forgotten_pt_cluster_key = self.cluster_key_circular_buffer.get(0)
        forgotten_pt_relevance = self.relevance_circular_buffer.get(0)
        forgotten_pt_index = self.min_internal_abs_idx
        forgotten_pt_true_abs_idx = self.true_abs_idx_circular_buffer.get(0)
        forgotten_pt_label = self.label_circular_buffer.get(0)
        # forgotten point in form: [key, label, relevance, index, abs_index, data]
        return forgotten_pt_cluster_key, forgotten_pt_label, forgotten_pt_relevance, forgotten_pt_index, forgotten_pt_true_abs_idx, forgotten_pt_data

    def find_closest_pts(self, X, k):
        closest_pts = []
        internal_idxes = []

        if len(self.ball_trees) != 0:

            # === Snapshot the trees and bounds while holding the build lock ===
            with self._tree_build_lock:
                # copy the list reference so later appends are invisible
                trees_snapshot = list(self.ball_trees)
                min_l_idx_covered_by_btree = trees_snapshot[0].min_index
                max_l_idx_covered_by_btree = trees_snapshot[-1].max_index

            # From here on we never look at self.ball_trees again
            min_cb_index_covered_by_btree = (
                    min_l_idx_covered_by_btree - self.min_internal_abs_idx
            )
            max_cb_index_covered_by_btree = (
                    max_l_idx_covered_by_btree - self.min_internal_abs_idx
            )

            # --- brute-force tail end ---
            for cb_i in range(0, min_cb_index_covered_by_btree):
                dist = np.linalg.norm(X - self.data_circular_buffer.get(cb_i))
                distances = [d[2] for d in closest_pts]
                l_internal_idx = cb_i + self.min_internal_abs_idx
                pos = bisect.bisect_left(distances, dist)
                if pos < k:
                    #   0            1                2     3      4     5    6
                    # [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]
                    closest_pts.insert(
                        pos,
                        (
                            self.cluster_key_circular_buffer.get(cb_i),
                            l_internal_idx,
                            dist,
                            self.label_circular_buffer.get(cb_i),
                            self.data_circular_buffer.get(cb_i),
                            self.relevance_circular_buffer.get(cb_i),
                            self.true_abs_idx_circular_buffer.get(cb_i),
                        ),
                    )

                    if len(closest_pts) > k:
                        closest_pts.pop()

            # --- search only the snapshotted trees ---
            X = X.reshape((1, -1))
            for ball_tree in trees_snapshot:
                dists, bt_idxs = ball_tree.query(X, min(k, ball_tree.length))

                for j in range(dists.shape[1]):
                    l_internal_idx = bt_idxs[0][j] + ball_tree.min_index
                    cb_idx = l_internal_idx - self.min_internal_abs_idx
                    dist = dists[0][j]
                    distances = [d[2] for d in closest_pts]
                    pos = bisect.bisect_left(distances, dist)

                    if pos < k and l_internal_idx not in internal_idxes:
                        #   0            1                2     3      4     5    6
                        # [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]
                        closest_pts.insert(
                            pos,
                            (
                                self.cluster_key_circular_buffer.get(cb_idx),
                                l_internal_idx,
                                dist,
                                self.label_circular_buffer.get(cb_idx),
                                self.data_circular_buffer.get(cb_idx),
                                self.relevance_circular_buffer.get(cb_idx),
                                self.true_abs_idx_circular_buffer.get(cb_idx),
                            ),
                        )
                        internal_idxes.insert(pos, cb_idx)

                        if len(closest_pts) > k:
                            closest_pts.pop()

            # brute force head end
            num_head_pts = self.max_internal_abs_idx - max_l_idx_covered_by_btree
            for cb_idx in range(max_cb_index_covered_by_btree, max_cb_index_covered_by_btree + num_head_pts  + 1):
                dist = np.linalg.norm(X - self.data_circular_buffer.get(cb_idx))
                l_internal_idx = cb_idx + self.min_internal_abs_idx
                distances = [d[2] for d in closest_pts]
                pos = bisect.bisect_left(distances, dist)

                if pos < k:
                    #   0            1                2     3      4     5    6
                    # [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]
                    closest_pts.insert(
                        pos,
                        (
                            self.cluster_key_circular_buffer.get(cb_idx),
                            l_internal_idx,
                            dist,
                            self.label_circular_buffer.get(cb_idx),
                            self.data_circular_buffer.get(cb_idx),
                            self.relevance_circular_buffer.get(cb_idx),
                            self.true_abs_idx_circular_buffer.get(cb_idx),
                        ),
                    )
                    if len(closest_pts) > k:
                        closest_pts.pop()

            num_pts_searched = (
                min_cb_index_covered_by_btree,
                trees_snapshot[-1].max_index - trees_snapshot[0].min_index + 1,
                num_head_pts,
            )

        else: # brute force all points
            for i in range(self.max_internal_abs_idx - self.min_internal_abs_idx + 1):
                dist = np.linalg.norm(X - self.data_circular_buffer.get(i))
                distances = [d for _, __, d, ___, ____, _____, ______ in closest_pts]
                l_internal_idx = i + self.min_internal_abs_idx
                pos = bisect.bisect_left(distances, dist)

                if pos < k:
                    #   0            1                2     3      4     5    6
                    # [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(i),
                                                 l_internal_idx,
                                                 dist,
                                                 self.label_circular_buffer.get(i),
                                                 self.data_circular_buffer.get(i),
                                                 self.relevance_circular_buffer.get(i),
                                                 self.true_abs_idx_circular_buffer.get(i)))

                    if len(closest_pts) > k:
                        closest_pts.pop()

            num_pts_searched = (self.max_internal_abs_idx - self.min_internal_abs_idx + 1,0,0)

        return closest_pts, num_pts_searched

    # currently unused, maybe remove
    def get_pt_data(self, internal_abs_idx):
        if self.min_internal_abs_idx <= internal_abs_idx < self.max_internal_abs_idx:
            return self.data_circular_buffer.get(internal_abs_idx - self.min_internal_abs_idx)

        return None

    # returns every datapoint with the abs index.
    def get_pt_data_abs(self, true_abs_idx):
        internal_abs_idx = self.true_abs_idx_circular_buffer.get_array().index(true_abs_idx)
        size_of_buff = len(self.true_abs_idx_circular_buffer)

        data_points = []
        while internal_abs_idx < size_of_buff and true_abs_idx == self.true_abs_idx_circular_buffer.get(internal_abs_idx):
            data_points.append(self.data_circular_buffer.get(internal_abs_idx))
            internal_abs_idx += 1

        return data_points if len(data_points) > 0 else None

    def _build_new_tree(self):
        """
        Take a snapshot of the latest window of points and build
        a BallTree covering absolute indices [abs_min, abs_max).
        Runs in its own thread.
        """
        try:
            # === 1. Atomic snapshot of current state ===
            with self._tree_build_lock:
                min_abs_snapshot = self.min_internal_abs_idx
                max_abs_snapshot = self.max_internal_abs_idx
                full_array = list(self.data_circular_buffer.get_array())

            # === 2. Compute absolute window bounds ===
            window_size = self.ball_tree_interval
            abs_min = max(min_abs_snapshot, max_abs_snapshot - window_size)
            abs_max = max_abs_snapshot  # exclusive upper bound

            # === 3. Convert to relative slice positions ===
            rel_start = abs_min - min_abs_snapshot
            rel_end = abs_max - min_abs_snapshot + 1

            # === 4. Extract the snapshot window ===
            data_snapshot = full_array[rel_start:rel_end]

            if not data_snapshot:
                return

            # === 5. Build the BallTree OFF the lock ===
            new_tree = BallTreeWithIndexes(data_snapshot, min_index=abs_min, max_index=abs_max, leaf_size=2)

            # === 6. Append finished tree under lock ===
            with self._tree_build_lock:
                self.ball_trees.append(new_tree)

                if self.ball_tree_interval <= self.max_internal_abs_idx and self.build_up_period:
                    self.ball_trees.pop(0)
                    self.build_up_period = False

                elif len(self.ball_trees) > 1 and self.build_up_period:
                    self.ball_trees.pop(0)


        finally:
            self._building_tree = False
            if not self.build_up_period:
                self.num_ball_trees_completed += 1

                if self.num_ball_trees_completed < self.num_ball_trees:
                    print("BUILD UP PERIOD: ", len(self.ball_trees), self.max_internal_abs_idx, self.true_abs_idx_circular_buffer.get(self.max_internal_abs_idx))

                if self.num_ball_trees_completed == self.num_ball_trees:
                    print("STEADY STATE REACHED: ", len(self.ball_trees), self.max_internal_abs_idx, self.true_abs_idx_circular_buffer.get(self.max_internal_abs_idx))
