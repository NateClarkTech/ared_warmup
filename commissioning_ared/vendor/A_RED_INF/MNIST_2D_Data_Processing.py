# NOTE: this works just like MNIST, but 1) limits the number of points to 25,000 so we can t-SNE it in
# a reasonable time, and 2) t-SNE's the data down to 2-D
# NOTE: the sparsity_levels end up not being perfectly accurate (but should be close)

import numpy as np
import pickle
import os
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import seaborn as sns
matplotlib.use('TkAgg')

# Create skewed subset
def create_skewed_mnist(X, y, sparsity_levels, seed):
    # Note: sparsity_levels comes in w/ largest value at 0.5, then 0.25, etc. representing ratio of
    # number of points to total
    sparsity_levels = np.array(sparsity_levels)
    np.random.seed(seed)
    digit_order = np.random.permutation(10)
    indices = []
    digit_indices_idx_0 = np.where(y == str(digit_order[0]))[0]
    num_most_populous_class = len(digit_indices_idx_0)
    print('num_most_populous_class:', num_most_populous_class)
    sparsity_levels = np.array((sparsity_levels * 2 * num_most_populous_class)).astype(int)
    # Note: need *2 because want _all of most populous class, then /2 from there...
    print('sparsity_levels:', sparsity_levels)
    for digit, count in zip(digit_order, sparsity_levels):
        digit_indices = np.where(y == str(digit))[0]
        if len(digit_indices) >= count:
            selected = np.random.choice(digit_indices, count, replace=False)
            indices.extend(selected)
        else:
            print(f"Warning: Not enough samples for digit {digit}, using all {len(digit_indices)}")
            indices.extend(digit_indices)
    indices = np.array(indices)
    np.random.shuffle(indices)

    X_skewed = X[indices]
    y_skewed = y[indices]
    return X_skewed, y_skewed, digit_order

def load_and_skew_mnist(sparsity_levels, seed, save_path="mnist_full.pkl"):
    """
    Loads the replicated MNIST dataset (10x duplicated) from a pickle file,
    creates a skewed subset using create_skewed_mnist, and returns both_a_and_r_queries the skewed
    and full replicated datasets.

    Args:
        sparsity_levels: list of 10 floats, ratio of samples per digit relative to most populous.
        seed: random seed for reproducibility.
        save_path: path to the replicated MNIST pickle file (mnist_replicated_10x.pkl).

    Returns:
        X_skewed, y_skewed: filtered and shuffled MNIST subset.
        X, y: full replicated MNIST dataset (700,000 samples).
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Replicated MNIST file not found at {save_path}. "
                               "Please generate it first using the previous script.")

    # Load the replicated MNIST dataset
    print(f"Loading MNIST from {save_path}...")
    with open(save_path, "rb") as file:
        data = pickle.load(file)
    
    # Extract images and labels
    X = data[0]  # Shape: (700000, 28, 28)
    y = data[1]  # Shape: (700000,)
    
    # Reshape images to (n_samples, 784) to match original code's expectation
    X = X.reshape(X.shape[0], -1)  # Shape: (700000, 784)
    
    # Convert labels to strings (original code expects string labels)
    y = y.astype(str)
    
    print(f"Replicated MNIST loaded: {X.shape[0]} samples")
    
    # Create skewed subset
    print(f"Creating skewed subset with sparsity {sparsity_levels}")
    X_skewed, y_skewed, digit_order = create_skewed_mnist(X, y, sparsity_levels, seed)

    print(f"Skewed dataset shape: {X_skewed.shape}")
    return X_skewed, y_skewed, X, y, digit_order

def generate_is_relevant(label_list, relevant_set):
    return [label in relevant_set for label in label_list]

def MNIST_2D_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS, seed):
    sparsity_levels = [(1 / int(2 ** n)) for n in range(1, 11)]

    X_skewed, y_skewed, X_full, y_full, digit_order = load_and_skew_mnist(sparsity_levels, seed)
    n_events = len(y_skewed)

    # Identify least common digits
    digit_counts = Counter(y_skewed)
    relevant_classes = [digit for digit, _ in digit_counts.most_common()[-N_REL_CLASSES:]]

    if 0 in VERBOSE_FLAGS:
        print(f"Running ARED on skewed MNIST dataset with {n_events} events")
        print(f"Least common digits: {relevant_classes} (marked as relevant)")

    # Generate relevance info
    relevance_array = generate_is_relevant(y_skewed, set(relevant_classes))
    y_w_rel = list(zip(y_skewed, relevance_array))

    sparsity_levels = [(digit_order[n], sparsity_levels[n]) for n in range(len(sparsity_levels))]

    tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(X_skewed) - 1))
    X_skewed = tsne.fit_transform(X_skewed)
    return X_skewed, y_w_rel, sparsity_levels, relevant_classes


def main():
    N_REL_CLASSES = 3
    VERBOSE_FLAGS = [0]
    seed = 42

    X_2d, y_w_rel, sparsity_levels, relevant_classes = MNIST_2D_setup_for_main(
        N_REL_CLASSES, VERBOSE_FLAGS, seed
    )

    print("Relevant classes:", relevant_classes)

    y_labels = np.array([int(y) for y, rel in y_w_rel])
    is_rel = np.array([rel for y, rel in y_w_rel])

    non_rel_mask = ~is_rel
    rel_mask = is_rel

    # Use jet colormap discretized into 10 colors
    num_classes = 10
    cmap = plt.get_cmap('Spectral_r', num_classes)
    bounds = np.arange(-0.5, num_classes + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Non-relevant points (smaller, semi-transparent)
    ax.scatter(
        X_2d[non_rel_mask, 0],
        X_2d[non_rel_mask, 1],
        c=y_labels[non_rel_mask],
        cmap=cmap,
        norm=norm,
        s=50,
        edgecolors='none',
        alpha=.45,
        label='Non-relevant classes'
    )

    # Relevant points (larger, red border)
    ax.scatter(
        X_2d[rel_mask, 0],
        X_2d[rel_mask, 1],
        c=y_labels[rel_mask],
        cmap=cmap,
        norm=norm,
        s=100,
        edgecolors='red',
        linewidths=1.2,
        alpha=1,
        label=f"Relevant classes: {relevant_classes}"
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, boundaries=bounds, ticks=np.arange(num_classes))
    cbar.set_ticklabels(np.arange(num_classes))
    cbar.set_label('Digit Class')

    #ax.set_title('2D t-SNE projection of skewed MNIST\nRelevant classes have red borders')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    #ax.legend()
    plt.grid()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()