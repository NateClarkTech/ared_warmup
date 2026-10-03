import numpy as np
from sklearn.decomposition import PCA  # Explicitly import PCA
from sklearn.manifold import TSNE  # Import t-SNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' or 'wxAgg' depending on your system

# Visualize data using t-SNE (subsample for large datasets)
def tSNE_3D_vis(X_skewed,y_w_rel):
    # Visualize data using t-SNE (subsample for large datasets)
    subsample_size = 10000  # Adjust this based on memory and computation time constraints
    if len(X_skewed) > subsample_size:
        indices = np.random.choice(len(X_skewed), subsample_size, replace=False)
        X_subsampled = X_skewed[indices]
        y_w_rel_subsampled = [y_w_rel[i] for i in indices]
    else:
        X_subsampled = X_skewed
        y_w_rel_subsampled = y_w_rel

    tsne = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)  # 3D t-SNE
    X_3d = tsne.fit_transform(X_subsampled)

    # Extract class labels and relevance
    class_labels = [y[0] for y in y_w_rel_subsampled]  # Use class number for coloring
    # Convert class labels to numeric values if they are strings
    class_labels = np.array(class_labels, dtype=float) if isinstance(class_labels[0], str) else np.array(class_labels)
    relevance = [y[1] for y in y_w_rel_subsampled]  # Extract relevance for size and alpha

    # Create 3D scatter plot with varying marker size and transparency
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Split into relevant and irrelevant points
    relevant_indices = [i for i, r in enumerate(relevance) if r]
    irrelevant_indices = [i for i, r in enumerate(relevance) if not r]

    # Plot irrelevant points with smaller size and lower opacity
    if irrelevant_indices:
        ax.scatter([X_3d[i, 0] for i in irrelevant_indices],
                   [X_3d[i, 1] for i in irrelevant_indices],
                   [X_3d[i, 2] for i in irrelevant_indices],
                   c=[class_labels[i] for i in irrelevant_indices],
                   cmap='jet',
                   s=20,  # Base size for irrelevant
                   alpha=0.2)  # Mostly transparent

    # Plot relevant points with larger size and higher opacity
    if relevant_indices:
        ax.scatter([X_3d[i, 0] for i in relevant_indices],
                   [X_3d[i, 1] for i in relevant_indices],
                   [X_3d[i, 2] for i in relevant_indices],
                   c=[class_labels[i] for i in relevant_indices],
                   cmap='jet',
                   s=50,  # Base size + 5 for relevant
                   alpha=1)  # Opaque

    plt.colorbar(ax.collections[0], label='Class Number')  # Use the first collection for colorbar
    ax.set_title(f't-SNE 3D Visualization of EMNIST Data (Subsample Size: {subsample_size})')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.show()


def plot_stacked_area(list1, list2, list3,legend_labels=['List 1', 'List 2', 'List 3']):
    # Convert lists to numpy arrays for easier manipulation
    y1 = np.array(list1)
    y2 = np.array(list2)
    y3 = np.array(list3)
    x = range(len(list1))

    # Create cumulative arrays for stacking
    y2_cum = y1 + y2
    y3_cum = y1 + y2 + y3

    # Create the stacked area plot

    plt.stackplot(x, y1, y2, y3, labels=legend_labels, colors=['#FF9999', '#66B2FF', '#99FF99'])

    # Customize the plot
    plt.title('Cumulative Stacked Area Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.show()


def plot_clusters_and_queries_over_time(num_clusters, num_queries,
                                        graph_batch_size=100,
                                        title="A/RED: Clusters and Queries Over Time",
                                        save_path=None):
    """
    Plot number of clusters and cumulative queries vs. number of points streamed.
    Displays the plot interactively during runtime and optionally saves it.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x_points = np.arange(len(num_clusters)) * graph_batch_size

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Plot number of clusters (left axis)
    color1 = 'tab:blue'
    ax1.set_xlabel('Points Streamed', fontsize=12)
    ax1.set_ylabel('Number of Clusters', color=color1, fontsize=12)
    ax1.plot(x_points, num_clusters, color=color1, marker='o',
             linewidth=2.5, markersize=4, label='Number of Clusters')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Plot cumulative queries (right axis)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Cumulative Queries', color=color2, fontsize=12)
    ax2.plot(x_points, num_queries, color=color2, marker='s',
             linewidth=2.5, markersize=4, label='Cumulative Queries')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(title, fontsize=14, pad=15)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()

    # Always show the plot during runtime
    plt.show(block=True)  # non-blocking so the program can continue

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")


# Example usage
if __name__ == "__main__":
    list1 = [1, 2, 3, 4, 5]
    list2 = [2, 3, 2, 3, 4]
    list3 = [3, 4, 5, 4, 3]
    plot_stacked_area(list1, list2, list3)
