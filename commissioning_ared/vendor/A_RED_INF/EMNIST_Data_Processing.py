import numpy as np
import pickle
import os
from collections import Counter

EMNIST_LABEL_TO_ASCII = [
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
    85, 86, 87, 88, 89, 90,
    97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121, 122
]

def label_to_char(label):
    """Convert EMNIST label index (0–61) to corresponding ASCII character."""
    try:
        label_int = int(label)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid label type for EMNIST: {label} (type {type(label)})")

    if 0 <= label_int < len(EMNIST_LABEL_TO_ASCII):
        return chr(EMNIST_LABEL_TO_ASCII[label_int])
    else:
        raise ValueError(f"Invalid EMNIST label index: {label_int}")

def load_emnist(save_path="Datasets/EMNIST/emnist.pkl"):
    """
    Loads the EMNIST dataset from a pickle file.

    Args:
        save_path: Path to the EMNIST pickle file.

    Returns:
        X: EMNIST images, shape (n_samples, 784).
        y: EMNIST labels, shape (n_samples,).
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"EMNIST file not found at {save_path}. "
                                "Please generate it first using the EMNIST loading script.")

    # Load the EMNIST dataset
    print(f"Loading EMNIST from {save_path}...")
    with open(save_path, "rb") as file:
        data = pickle.load(file)

    # Extract images and labels
    X = data['images']  # Shape: (814255, 784)
    y = data['labels']  # Shape: (814255,)

    print(f"EMNIST loaded: {X.shape[0]} samples")
    return X, y


def generate_is_relevant(label_list, relevant_set):
    """
    Generates a boolean array indicating whether each label is in the relevant set.

    Args:
        label_list: List of labels.
        relevant_set: Set of relevant labels.

    Returns:
        List of booleans indicating relevance.
    """
    return [label in relevant_set for label in label_list]


def EMNIST_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS, seed=42, save_path="Datasets/EMNIST/emnist.pkl"):
    """
    Sets up the EMNIST dataset for main processing, loading the full dataset and
    marking the least common classes as relevant.

    Args:
        N_REL_CLASSES: Number of least common classes to mark as relevant.
        VERBOSE_FLAGS: List of flags for controlling verbosity.
        save_path: Path to the EMNIST pickle file.

    Returns:
        X: EMNIST images, shape (n_samples, 784).
        y_w_rel: List of (char_label, relevance) tuples for the dataset.
    """
    # Load the full EMNIST dataset
    X, y = load_emnist(save_path)
    # Convert labels to characters
    y = [label_to_char(label) for label in y]
    n_events = len(y)

    # Identify the N_REL_CLASSES least common classes
    class_counts = Counter(y)
    least_common_classes = [cls for cls, _ in class_counts.most_common()[-N_REL_CLASSES:]]
    lc_class_freqs = [f for _, f in class_counts.most_common()[-N_REL_CLASSES:]]
    sparsity_levels = [(k,v/n_events) for k,v in class_counts.items()]
    num_relevant_points = sum(lc_class_freqs)
    num_points_total = X.shape[0]
    print(f"Number of classes: {len(set(y))}, number of relevant classes: {N_REL_CLASSES}")
    print(f"% of relevant class points = {100*num_relevant_points/num_points_total:.2f}")

    if 0 in VERBOSE_FLAGS:
        print(f"Running ARED on EMNIST dataset with {n_events} events")
        print(f"Least common classes: {least_common_classes} (marked as relevant)")


    # Generate relevance info
    relevance_array = generate_is_relevant(y, set(least_common_classes))



    # Combine into (char_label, relevance) tuples
    y_w_rel = list(zip(y, relevance_array))

    return X, y_w_rel,sparsity_levels, least_common_classes

if __name__ == "__main__":
    N_REL_CLASSES = 2
    VERBOSE_FLAGS = [0]
    X, y_w_rel, least_common_classes, rel_classes = EMNIST_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS)
    print(f"EMNIST dataset shape: {X.shape}")
    print(f"Sample labels with relevance: {y_w_rel[:5]}")
