#!/usr/bin/env python3
import os
import pickle
import cv2
import numpy as np


def resize_parking_lot_features(
    input_path: str = "./Parking_Lot_Data/features.pkl",
    output_path: str = "./Parking_Lot_Data/resized_features_224.pkl",
) -> None:

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Loading original features...")
    with open(input_path, "rb") as f:
        features = pickle.load(f)

    print(f"Original shape: {features.shape}")

    # Expect shape: (n, 128, 128, 3)
    if features.ndim != 4 or features.shape[1:] != (128, 128, 3):
        raise ValueError(f"Unexpected shape {features.shape}, expected (n,128,128,3)")

    n_samples = features.shape[0]

    print("Resizing images...")

    resized = np.stack([
        cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        for img in features
    ])

    print(f"Resized shape: {resized.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(resized, f)

    print(f"Resized features saved to: {output_path}")


if __name__ == "__main__":
    resize_parking_lot_features()
