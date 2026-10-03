import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def generate_synthetic_dataset_with_relevance(n_least_populous, seed=42):
    """
    Generates a 2D dataset with 10 well-separated classes and a relevance vector
    marking the n least populous classes with 1s. Concatenates y and relevance into a 2D array.

    Args:
        n_least_populous: Number of least populous classes to mark as relevant (1s).
        seed: Random seed for reproducibility.

    Returns:
        X: Feature matrix (n_samples, 2).
        y_w_rel: 2D array (n_samples, 2) with class labels in first column and relevance in second.
    """
    # Define the number of classes
    num_classes = 10

    # Define the number of samples for each class
    # Start with 512 for class 0, halving each time
    samples_per_class = [500000 // (2 ** i) for i in range(num_classes)]
    print("Samples per class:", samples_per_class)

    # Generate random centers with good separation
    np.random.seed(seed)
    min_distance = 10.0  # Minimum distance between centers for good separation
    centers = []

    # Generate first center
    centers.append(np.random.uniform(-20, 20, 2))

    # Generate remaining centers, ensuring minimum distance
    for _ in range(num_classes - 1):
        while True:
            new_center = np.random.uniform(-20, 20, 2)
            # Check distance to all existing centers
            too_close = False
            for existing_center in centers:
                distance = np.linalg.norm(new_center - existing_center)
                if distance < min_distance:
                    too_close = True
                    break
            if not too_close:
                centers.append(new_center)
                break
    centers = np.array(centers)

    # Generate the dataset
    X, y = make_blobs(n_samples=samples_per_class, centers=centers, cluster_std=1.0, random_state=seed)
    n_pts = X.shape[0]
    sparsity_levels = [(str(n),samples_per_class[n]/n_pts) for n in range(num_classes)]
    indices = list(range(len(X)))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    y = y.astype(str)

    # Identify the n least populous classes
    # Since samples_per_class is [512, 256, ..., 1], least populous are highest indices
    least_populous_classes = list(range(num_classes - n_least_populous, num_classes))
    least_populous_classes = [str(d) for d in least_populous_classes]
    print(f"Least populous classes (marked as relevant): {least_populous_classes}")

    # Generate relevance vector (1 for least populous classes, 0 otherwise)
    relevance = np.isin(y, least_populous_classes).astype(int)

    # Concatenate y and relevance into a 2D array
    y_w_rel = [(label, bool(rel)) for label, rel in zip(y, relevance)]

    return X, y_w_rel, sparsity_levels, least_populous_classes


# Example usage
if __name__ == "__main__":
    n_least_populous = 2  # Example: mark 2 least populous classes
    X, y_w_rel = generate_synthetic_dataset_with_relevance(n_least_populous)
    print("Dataset shape:", X.shape)
    print("y_w_rel shape:", y_w_rel.shape)
    print("Number of relevant samples (1s):", np.sum(y_w_rel[:, 1]))