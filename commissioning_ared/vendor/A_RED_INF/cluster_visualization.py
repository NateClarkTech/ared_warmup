from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi
import copy

class ClusterEvolutionPlotter:
    def __init__(self):
        self.snapshots = []  # List of extracted snapshots
        self.label_colors = {}  # {label: color}
        self._label_color_map = plt.cm.get_cmap("tab20")
        self._next_label_color_idx = 0

    def add_snapshot(self, ared, X, y, title):
        """Store snapshot data safely."""
        cluster_dict = ared.subspace_partition.cluster_dict
        if not cluster_dict:
            print("Warning: ARED has no clusters at snapshot time.")
            return

        snapshot_clusters = {}
        for cluster_id, cluster in cluster_dict.items():
            if not hasattr(cluster, "l_pt_idxs"):
                continue
            snapshot_clusters[cluster_id] = {
                "l_pt_idxs": copy.deepcopy(cluster.l_pt_idxs),
                "comp_distance": getattr(cluster, "comp_distance", 1.0),
            }

        snapshot_data = {
            "clusters": snapshot_clusters,
            "kappa": getattr(ared, "kappa", 1.0),
            "X": np.copy(X) if isinstance(X, np.ndarray) else copy.deepcopy(X),
            "y": np.copy(y) if isinstance(y, np.ndarray) else copy.deepcopy(y),
            "title": str(title),
        }
        self.snapshots.append(snapshot_data)

    def _get_color_for_label(self, label):
        if label not in self.label_colors:
            color = self._label_color_map(self._next_label_color_idx % 20)
            self.label_colors[label] = color
            self._next_label_color_idx += 1
        return self.label_colors[label]

    def _plot_snapshot(self, ax, snapshot):
        cluster_dict = snapshot["clusters"]
        kappa = snapshot["kappa"]
        X_skewed = snapshot["X"]
        y_w_rel = snapshot["y"]
        title = snapshot["title"]

        if not cluster_dict:
            ax.text(0.5, 0.5, "No clusters", ha='center', va='center')
            return np.zeros((0, 2))

        points, labels, cluster_ids, cluster_comp_dists = [], [], [], []
        for cluster_id, info in cluster_dict.items():
            indices = info["l_pt_idxs"]
            for idx in indices:
                if idx < len(X_skewed):
                    points.append(X_skewed[idx])
                    labels.append(y_w_rel[idx][0])
                    cluster_ids.append(cluster_id)
                    cluster_comp_dists.append(info["comp_distance"])

        if not points:
            ax.text(0.5, 0.5, "No points", ha='center', va='center')
            return np.zeros((0, 2))

        points = np.array(points)
        labels = np.array(labels)
        cluster_ids = np.array(cluster_ids)
        cluster_comp_dists = np.array(cluster_comp_dists)

        # --- t-SNE projection ---
        if points.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(points)-1))
            points_2d = tsne.fit_transform(points)
        else:
            points_2d = points

        # --- Compute centroids per cluster ---
        cluster_centroids = {}
        for cid in np.unique(cluster_ids):
            cluster_points = points_2d[cluster_ids == cid]
            cluster_centroids[cid] = np.mean(cluster_points, axis=0)

        # --- Voronoi using single-linkage (all points as sites) ---
        all_sites = points_2d
        x_min, y_min = all_sites.min(axis=0)
        x_max, y_max = all_sites.max(axis=0)
        data_range = np.ptp(all_sites, axis=0)
        extension_length = 2 * np.max(data_range)
        dummy_points = np.array([
            [x_min - extension_length, y_min - extension_length],
            [x_min - extension_length, y_max + extension_length],
            [x_max + extension_length, y_min - extension_length],
            [x_max + extension_length, y_max + extension_length],
        ])
        vor = Voronoi(np.vstack((all_sites, dummy_points)))

        # Build cluster-specific Voronoi polygons
        cluster_voronoi_poly = {cid: [] for cid in np.unique(cluster_ids)}
        for i, cid in enumerate(cluster_ids):
            region_index = vor.point_region[i]
            vertices = vor.regions[region_index]
            if -1 in vertices or len(vertices) == 0:
                cluster_voronoi_poly[cid].append(None)
            else:
                cluster_voronoi_poly[cid].append(Polygon(vor.vertices[vertices]))

        # --- Base radius relative to dataset scale ---
        base_radius = 0.05 * np.linalg.norm(np.ptp(points_2d, axis=0))

        # --- Draw clusters ---
        for cid in np.unique(cluster_ids):
            mask = cluster_ids == cid
            cluster_points = points_2d[mask]
            cluster_comp_dist = cluster_comp_dists[mask]

            # Color by dominant label
            cluster_labels = labels[mask]
            unique_cluster_labels, counts = np.unique(cluster_labels, return_counts=True)
            dominant_label = unique_cluster_labels[np.argmax(counts)]
            cluster_color = self._get_color_for_label(dominant_label)

            # Merged boundary
            circles = [
                Point(p[0], p[1]).buffer(cd / max(kappa, 1e-6) * base_radius, resolution=64)
                for p, cd in zip(cluster_points, cluster_comp_dist)
            ]
            merged_shape = unary_union(circles)

            # Intersect with all Voronoi polygons for this cluster
            polys = []
            for poly in cluster_voronoi_poly[cid]:
                if poly is not None:
                    polys.append(merged_shape.intersection(poly))
            merged_shape = unary_union(polys)

            polygons = [merged_shape] if merged_shape.geom_type == "Polygon" else getattr(merged_shape, "geoms", [])
            for poly in polygons:
                if not poly.is_empty:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color=cluster_color, alpha=0.25, zorder=0)

            # Dashed lines to centroid
            centroid = cluster_centroids[cid]
            for p in cluster_points:
                ax.plot([p[0], centroid[0]], [p[1], centroid[1]],
                        color=cluster_color, linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

            # Annotate cluster ID
            ax.text(
                centroid[0], centroid[1], f"C{cid}",
                fontsize=11, weight="bold", ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3"),
                zorder=10
            )

        # --- Scatter points colored by label ---
        for lbl in np.unique(labels):
            mask = labels == lbl
            ax.scatter(points_2d[mask, 0], points_2d[mask, 1],
                       color=self._get_color_for_label(lbl),
                       s=25, alpha=0.8, edgecolors='none', zorder=5)

        # --- Voronoi boundaries ---
        if len(cluster_ids) >= 2:
            # Draw AFTER scatter points, with very high zorder
            for r, simplex in enumerate(vor.ridge_vertices):
                i, j = vor.ridge_points[r]
                if i >= len(cluster_ids) or j >= len(cluster_ids):
                    continue
                if cluster_ids[i] == cluster_ids[j]:
                    continue
                simplex = np.asarray(simplex)
                if np.all(simplex >= 0):
                    start, end = vor.vertices[simplex]
                    line = plt.Line2D(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        color='black',
                        linestyle='--',
                        linewidth=1.4,
                        alpha=0.95,
                        zorder=1000,  # draw over everything else
                    )
                    ax.add_artist(line)

        # --- Adjust view ---
        margin = 0.1 * np.max(data_range)
        x_min, x_max = points_2d[:, 0].min() - margin, points_2d[:, 0].max() + margin
        y_min, y_max = points_2d[:, 1].min() - margin, points_2d[:, 1].max() + margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # --- Move title below subplot ---
        ax.set_xlabel(title, fontsize=14, labelpad=15)

        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle=":", alpha=0.4)

        return points_2d

    def plot(self, align_axes=True, save_pdf_path="clusters_over_time.pdf"):
        """Render all stored snapshots as independent time steps."""
        if not self.snapshots:
            print("No snapshots to plot.")
            return

        n = len(self.snapshots)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharex=True, sharey=True)
        if n == 1:
            axes = [axes]

        all_points = []
        for snapshot, ax in zip(self.snapshots, axes):
            pts = self._plot_snapshot(ax, snapshot)
            all_points.append(pts)

        if align_axes:
            all_points = np.vstack([p for p in all_points if len(p)])
            if len(all_points) > 0:
                x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
                y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
                for ax in axes:
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # space for titles below

        # --- Save as PDF if path is provided ---
        if save_pdf_path is not None:
            fig.savefig(save_pdf_path, format='pdf', bbox_inches='tight')
            print(f"Plot saved as PDF: {save_pdf_path}")

        plt.show(block=False)

    def plot_dataset(self, X, y, title="Dataset Visualization", save_pdf_path=None):
        """
        Plot a dataset (X, y) with a Voronoi overlay, using colors from label_colors.
        Assigns new colors for unseen labels.

        Parameters:
        - X: ndarray of shape (n_samples, n_features), the data points.
        - y: ndarray of shape (n_samples,) or (n_samples, 1), or list of lists, the labels.
        - title: str, title for the plot.
        - save_pdf_path: str, optional path to save the plot as a PDF.
        """
        # --- Ensure inputs are numpy arrays ---
        X = np.array(X)

        # --- Handle y's structure ---
        y = np.array(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()  # Convert (n_samples, 1) to (n_samples,)
        elif y.ndim == 2:
            # If y is 2D with multiple columns, assume first column contains labels
            y = np.array([row[0] for row in y])

        # --- Validate input shapes ---
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have mismatched number of samples: X has {X.shape[0]}, y has {y.shape[0]}")

        # --- t-SNE projection if necessary ---
        if X.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
            points_2d = tsne.fit_transform(X)
        else:
            points_2d = X

        # --- Initialize figure ---
        fig, ax = plt.subplots(figsize=(8, 6))

        # --- Voronoi setup (bounded) ---
        all_sites = points_2d
        x_min, y_min = all_sites.min(axis=0)
        x_max, y_max = all_sites.max(axis=0)
        data_range = np.ptp(all_sites, axis=0)
        extension_length = 2 * np.max(data_range)
        dummy_points = np.array([
            [x_min - extension_length, y_min - extension_length],
            [x_min - extension_length, y_max + extension_length],
            [x_max + extension_length, y_min - extension_length],
            [x_max + extension_length, y_max + extension_length],
        ])
        vor = Voronoi(np.vstack((all_sites, dummy_points)))

        # --- Scatter points colored by label ---
        unique_labels = np.unique(y)
        for lbl in unique_labels:
            mask = y == lbl
            if mask.sum() == 0:
                continue
            ax.scatter(points_2d[mask, 0], points_2d[mask, 1],
                       color=self._get_color_for_label(lbl),
                       s=25, alpha=0.8, edgecolors='none', zorder=5, label=str(lbl))

        # --- Draw Voronoi boundaries between different labels ---
        if len(np.unique(y)) >= 2:
            for r, simplex in enumerate(vor.ridge_vertices):
                i, j = vor.ridge_points[r]
                if i >= len(y) or j >= len(y):
                    continue
                if y[i] == y[j]:
                    continue
                simplex = np.asarray(simplex)
                if np.all(simplex >= 0):
                    start, end = vor.vertices[simplex]
                    ax.plot([start[0], end[0]], [start[1], end[1]],
                            color='black', linestyle='--', linewidth=1.4,
                            alpha=0.95, zorder=1000)  # high zorder so borders overlay points

        # --- Adjust view ---
        margin = 0.1 * np.max(data_range)
        ax.set_xlim(points_2d[:, 0].min() - margin, points_2d[:, 0].max() + margin)
        ax.set_ylim(points_2d[:, 1].min() - margin, points_2d[:, 1].max() + margin)
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle=":", alpha=0.4)

        # --- Add legend ---
        #ax.legend(title="Labels", loc="best")

        # --- Title below ---
        ax.set_xlabel(title, fontsize=14, labelpad=15)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        # --- Optional PDF saving ---
        if save_pdf_path is not None:
            fig.savefig(save_pdf_path, format="pdf", bbox_inches="tight")
            print(f"Plot saved as PDF: {save_pdf_path}")

        plt.show(block=False)

    def plot_clusters_colored_by_label(self, ared, X_skewed, y_w_rel, title="Cluster Visualization by Label", save_pdf_path="cluster_visualization"):
        """
        Visualizes clusters using the same style as ClusterEvolutionPlotter subplots:
        - merged circular cluster regions
        - Voronoi competition boundaries
        - dashed lines to centroids
        - consistent color scheme
        """
        cluster_dict = ared.subspace_partition.cluster_dict
        if not cluster_dict:
            print("No clusters to visualize.")
            return

        # --- Collect data from ARED clusters ---
        points, labels, cluster_ids, cluster_comp_dists = [], [], [], []
        for cluster_id, cluster in cluster_dict.items():
            if not hasattr(cluster, "l_pt_idxs"):
                continue
            indices = cluster.l_pt_idxs
            for idx in indices:
                if idx < len(X_skewed):
                    points.append(X_skewed[idx])
                    labels.append(y_w_rel[idx][0])
                    cluster_ids.append(cluster_id)
                    cluster_comp_dists.append(getattr(cluster, "comp_distance", 1.0))

        if not points:
            print("No points to visualize.")
            return

        points = np.array(points)
        labels = np.array(labels)
        cluster_ids = np.array(cluster_ids)
        cluster_comp_dists = np.array(cluster_comp_dists)

        # --- t-SNE projection if necessary ---
        if points.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(points)-1))
            points_2d = tsne.fit_transform(points)
        else:
            points_2d = points

        # --- Initialize figure ---
        fig, ax = plt.subplots(figsize=(8, 6))

        # --- Compute cluster centroids ---
        cluster_centroids = {}
        for cid in np.unique(cluster_ids):
            cluster_points = points_2d[cluster_ids == cid]
            cluster_centroids[cid] = np.mean(cluster_points, axis=0)

        # --- Voronoi setup (bounded) ---
        all_sites = points_2d
        x_min, y_min = all_sites.min(axis=0)
        x_max, y_max = all_sites.max(axis=0)
        data_range = np.ptp(all_sites, axis=0)
        extension_length = 2 * np.max(data_range)
        dummy_points = np.array([
            [x_min - extension_length, y_min - extension_length],
            [x_min - extension_length, y_max + extension_length],
            [x_max + extension_length, y_min - extension_length],
            [x_max + extension_length, y_max + extension_length],
        ])
        vor = Voronoi(np.vstack((all_sites, dummy_points)))

        # --- Build cluster Voronoi polygons ---
        cluster_voronoi_poly = {cid: [] for cid in np.unique(cluster_ids)}
        for i, cid in enumerate(cluster_ids):
            region_index = vor.point_region[i]
            vertices = vor.regions[region_index]
            if -1 in vertices or len(vertices) == 0:
                cluster_voronoi_poly[cid].append(None)
            else:
                cluster_voronoi_poly[cid].append(Polygon(vor.vertices[vertices]))

        # --- Base radius relative to dataset scale ---
        base_radius = 0.05 * np.linalg.norm(data_range)

        # --- Draw clusters ---
        for cid in np.unique(cluster_ids):
            mask = cluster_ids == cid
            cluster_points = points_2d[mask]
            cluster_comp_dist = cluster_comp_dists[mask]
            cluster_labels = labels[mask]

            unique_cluster_labels, counts = np.unique(cluster_labels, return_counts=True)
            dominant_label = unique_cluster_labels[np.argmax(counts)]
            cluster_color = self._get_color_for_label(dominant_label)

            # Create merged circular boundary
            circles = [
                Point(p[0], p[1]).buffer(cd / max(getattr(ared, "kappa", 1.0), 1e-6) * base_radius, resolution=64)
                for p, cd in zip(cluster_points, cluster_comp_dist)
            ]
            merged_shape = unary_union(circles)

            # Clip with Voronoi regions
            polys = []
            for poly in cluster_voronoi_poly[cid]:
                if poly is not None:
                    polys.append(merged_shape.intersection(poly))
            merged_shape = unary_union(polys)

            # Fill merged shape
            polygons = [merged_shape] if merged_shape.geom_type == "Polygon" else getattr(merged_shape, "geoms", [])
            for poly in polygons:
                if not poly.is_empty:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color=cluster_color, alpha=0.25, zorder=0)

            # Dashed lines to centroid
            centroid = cluster_centroids[cid]
            for p in cluster_points:
                ax.plot([p[0], centroid[0]], [p[1], centroid[1]],
                        color=cluster_color, linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

            # Annotate cluster ID
            ax.text(
                centroid[0], centroid[1], f"C{cid}",
                fontsize=11, weight="bold", ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3"),
                zorder=10
            )

        # --- Scatter points ---
        for lbl in np.unique(labels):
            mask = labels == lbl
            ax.scatter(points_2d[mask, 0], points_2d[mask, 1],
                       color=self._get_color_for_label(lbl), s=25, alpha=0.8, edgecolors='none', zorder=5)

        # --- Voronoi boundaries between clusters ---
        if len(cluster_ids) >= 2:
            for r, simplex in enumerate(vor.ridge_vertices):
                i, j = vor.ridge_points[r]
                if i >= len(cluster_ids) or j >= len(cluster_ids):
                    continue
                if cluster_ids[i] == cluster_ids[j]:
                    continue
                simplex = np.asarray(simplex)
                if np.all(simplex >= 0):
                    start, end = vor.vertices[simplex]
                    ax.plot([start[0], end[0]], [start[1], end[1]],
                            color='black', linestyle='--', linewidth=1.4,
                            alpha=0.95, zorder=1000)  # always above scatter

        # --- Adjust view ---
        margin = 0.1 * np.max(data_range)
        ax.set_xlim(points_2d[:, 0].min() - margin, points_2d[:, 0].max() + margin)
        ax.set_ylim(points_2d[:, 1].min() - margin, points_2d[:, 1].max() + margin)
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle=":", alpha=0.4)

        # --- Title below ---
        ax.set_xlabel(title, fontsize=14, labelpad=15)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        # --- Optional PDF saving ---
        if save_pdf_path is not None:
            fig.savefig(save_pdf_path, format="pdf", bbox_inches="tight")
            print(f"Plot saved as PDF: {save_pdf_path}")

        plt.show(block=False)