from A_REDIN import ARED
from Circular_Buffer import Circular_Buffer
from DAGMM import DAGMM
import threading
import numpy as np

class ARED_w_DAGMM:
    def __init__(
        self,
        oracle,
        kappa,
        l_buf_size,
        K_COMP_PTS,
        QS_VAR,
        REL_PROC_VAR,
        SM_VAR,
        NGHBHOOD_MERGE,
        SINGLETON_MERGE,
        VERBOSE_FLAGS,
        dagmm_data_buffer_size,
        dagmm_latent_dim=2,          # DaGMM hyper-parameters
        dagmm_n_components=3,
        dagmm_lambda_energy=0.1,
        dagmm_lambda_cov=0.005,
        dagmm_epochs=100,
        dagmm_batch_size=1024,
        dagmm_lr=1e-4,
    ):
        # DAGMM buffers & models
        self.dagmm_data_buffer = Circular_Buffer(dagmm_data_buffer_size)
        self._dagmm_data_buffer_size = dagmm_data_buffer_size
        self.dagmm_stale = None          # currently used model (may be None at start)
        self.dagmm_fresh = None          # newly trained model waiting to be swapped
        self.dagmm_training_thread = None
        self.dagmm_lock = threading.Lock()

        # ARED (works in latent space)
        self.ared = ARED(
            oracle, kappa, l_buf_size, K_COMP_PTS,
            QS_VAR, REL_PROC_VAR, SM_VAR,
            NGHBHOOD_MERGE, SINGLETON_MERGE, VERBOSE_FLAGS
        )

        # Flags & counters
        self.first_point_processed = False
        self.points_buffered_for_first_training = 0
        self.min_points_to_train = dagmm_data_buffer_size // 2   # 50 % of buffer
        self.unseen_fraction_threshold = 0.5                     # trigger retrain when ≥50 % are new

        # DAGMM hyper-parameters
        self.dagmm_params = {
            "latent_dim": dagmm_latent_dim,
            "n_components": dagmm_n_components,
            "lambda_energy": dagmm_lambda_energy,
            "lambda_cov": dagmm_lambda_cov,
            "epochs": dagmm_epochs,
            "batch_size": dagmm_batch_size,
            "lr": dagmm_lr,
        }

    # ------------------------------------------------------------------ #
    # Helper: transform a point (or batch) with the current model
    # ------------------------------------------------------------------ #
    def _transform_with_current_model(self, x):
        with self.dagmm_lock:
            model = self.dagmm_stale or self.dagmm_fresh
        if model is None:
            return x  # no model yet → use raw space
        return model.transform(np.array([x]))[0]

    # ------------------------------------------------------------------ #
    # Background training thread
    # ------------------------------------------------------------------ #
    def _train_dagmm_async(self):
        """Train a fresh DaGMM on the current content of dagmm_data_buffer."""
        data = []
        for i in range(self.dagmm_data_buffer.count):
            pt = self.dagmm_data_buffer.get(i)
            if pt is not None:
                data.append(pt)
        if len(data) < self.min_points_to_train:
            return

        X = np.stack(data)

        new_model = DAGMM(
            128*128
        )
        new_model.fit(
            X,
            epochs=self.dagmm_params["epochs"],
            batch_size=self.dagmm_params["batch_size"],
            lr=self.dagmm_params["lr"],
            verbose=False,
        )

        # Swap fresh model in (atomic)
        with self.dagmm_lock:
            self.dagmm_fresh = new_model

    # ------------------------------------------------------------------ #
    # Swap fresh → stale and (re)project everything in ARED's FiniteBuffer
    # ------------------------------------------------------------------ #
    def _swap_and_reproject(self):
        with self.dagmm_lock:
            if self.dagmm_fresh is None:
                return
            self.dagmm_stale = self.dagmm_fresh
            self.dagmm_fresh = None

        # Re-project **all live points** in ARED's FiniteBuffer into the new latent space
        new_latent_points = []
        indices_to_update = []
        for i in range(self.ared.l_buf.data_buffer.count):
            raw_pt = self.ared.l_buf.data_buffer.get(i)
            if raw_pt is not None:
                latent_pt = self.dagmm_stale.transform(np.array([raw_pt]))[0]
                new_latent_points.append(latent_pt)
                indices_to_update.append(i)

        # Bulk-update the FiniteBuffer (avoids rebuilding KD-Tree on every insert)
        for idx, latent_pt in zip(indices_to_update, new_latent_points):
            self.ared.l_buf.data_buffer.set_at(idx, latent_pt)

        # Re-build the search structure once
        self.ared.l_buf._rebuild_search_structure()

        print(f"[DAGMM] Swapped new model → latent space updated "
              f"({len(indices_to_update)} points re-projected)")

    # ------------------------------------------------------------------ #
    # Main streaming entry point
    # ------------------------------------------------------------------ #
    def process_point(self, data_point):
        """
        Called for every incoming raw (high-dim) point.
        Handles:
          • buffering for DAGMM training
          • asynchronous training & swapping
          • dimensionality reduction (stale model) → feed to ARED
        """
        # 1. Always buffer the raw point for future DAGMM training
        self.dagmm_data_buffer.insert(data_point)

        # 2. First-time training trigger (when we have enough data)
        if self.dagmm_stale is None and self.dagmm_data_buffer.count >= self.min_points_to_train:
            if self.dagmm_training_thread is None or not self.dagmm_training_thread.is_alive():
                self.dagmm_training_thread = threading.Thread(target=self._train_dagmm_async, daemon=True)
                self.dagmm_training_thread.start()

        # 3. Check if we should swap a finished fresh model
        if self.dagmm_fresh is not None:
            self._swap_and_reproject()

        # 4. Decide whether to start a new background training
        #    (when ≥50 % of the buffer consists of points not seen by the current stale model)
        if self.dagmm_stale is not None:
            unseen_count = 0
            for i in range(self.dagmm_data_buffer.count):
                if self.dagmm_data_buffer.get(i) is not None:
                    unseen_count += 1
            if unseen_count / max(self.dagmm_data_buffer.count, 1) >= self.unseen_fraction_threshold:
                if self.dagmm_training_thread is None or not self.dagmm_training_thread.is_alive():
                    self.dagmm_training_thread = threading.Thread(target=self._train_dagmm_async, daemon=True)
                    self.dagmm_training_thread.start()

        # 5. Transform current point with the best available model
        latent_point = self._transform_with_current_model(data_point)

        # 6. Feed to ARED
        if not self.first_point_processed:
            self.ared.process_first_point(latent_point)
            self.first_point_processed = True
        else:
            self.ared.process_point(latent_point)

    # ------------------------------------------------------------------ #
    # Optional explicit training call (e.g. for batch pre-training)
    # ------------------------------------------------------------------ #
    def train_dagmm(self):
        """Force immediate training on current buffer contents (blocking)."""
        self._train_dagmm_async()
        if self.dagmm_fresh is not None:
            self._swap_and_reproject()