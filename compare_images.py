#!/usr/bin/env python3

"""
compare_images.py

Compare two images using DINOv2's pretrained visual features.

Methods:
    1. Cosine Similarity
    2. Euclidean Distance (L2)
    3. Manhattan Distance (L1)
    4. Pearson Correlation
    5. Mahalanobis Distance (optional)

Usage:
    python3 compare_images.py images/corn_field_1.jpg images/corn_field_2.jpg
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# ==========================================
# CONFIGURATION
# ==========================================

# Pretrained DINOv2 Small model (384 features)
MODEL_NAME = "facebook/dinov2-small"

# Use NVIDIA GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Regularization for Mahalanobis covariance estimation
COVARIANCE_REGULARIZATION = 0.01


# ==========================================
# LOAD DINOV2
# ==========================================

print(f"Using device: {device}")
print("Loading DINOv2...")

# Load image processor and pretrained model
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Move model to GPU/CPU and enable evaluation mode
model = model.to(device)
model.eval()

print("DINOv2 loaded successfully!")


# ==========================================
# EXTRACT FEATURES
# ==========================================

# Extract the 384-dimensional CLS feature vector from an image
def get_features(image_path):

    # Load image and convert to RGB
    image = Image.open(image_path).convert("RGB")

    # Prepare image for DINOv2
    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    # Run inference without calculating gradients
    with torch.inference_mode():
        outputs = model(**inputs)

    # Extract CLS token representing the entire image
    features = outputs.last_hidden_state[:, 0, :]

    return features


# ==========================================
# COSINE SIMILARITY
# ==========================================

# Compare the direction of two feature vectors (higher = more similar)
def calculate_cosine(features1, features2):

    similarity = F.cosine_similarity(
        features1,
        features2,
        dim=1
    ).item()

    return similarity


# ==========================================
# EUCLIDEAN DISTANCE (L2)
# ==========================================

# Calculate straight-line distance between feature vectors (lower = more similar)
def calculate_euclidean(features1, features2):

    distance = torch.dist(
        features1,
        features2,
        p=2
    ).item()

    return distance


# ==========================================
# MANHATTAN DISTANCE (L1)
# ==========================================

# Calculate sum of absolute feature differences (lower = more similar)
def calculate_manhattan(features1, features2):

    distance = torch.dist(
        features1,
        features2,
        p=1
    ).item()

    return distance


# ==========================================
# PEARSON CORRELATION
# ==========================================

# Measure correlation between feature vectors (higher = more similar)
def calculate_pearson(features1, features2):

    # Flatten feature tensors
    a = features1.flatten()
    b = features2.flatten()

    # Avoid division by zero when a vector has no variance
    if torch.std(a) == 0 or torch.std(b) == 0:
        return float("nan")

    # Calculate Pearson correlation
    correlation = torch.corrcoef(
        torch.stack([a, b])
    )[0, 1].item()

    return correlation


# ==========================================
# LOAD REFERENCE IMAGES
# ==========================================

# Extract DINOv2 features from reference images for Mahalanobis distance
def load_reference_features(reference_dir):

    reference_dir = Path(reference_dir)

    # Verify reference folder exists
    if not reference_dir.is_dir():
        raise ValueError(
            f"Reference directory does not exist: {reference_dir}"
        )

    # Find all supported image files
    image_paths = sorted(
        path for path in reference_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    # At least two images are needed to estimate covariance
    if len(image_paths) < 2:
        raise ValueError(
            "Mahalanobis distance requires at least 2 reference images."
        )

    print(f"\nLoading {len(image_paths)} reference images...")

    reference_features = []

    # Extract features from each reference image
    for index, image_path in enumerate(image_paths, start=1):

        print(
            f"[{index}/{len(image_paths)}] "
            f"Processing: {image_path.name}"
        )

        features = get_features(image_path)

        # Move features to CPU
        reference_features.append(
            features.squeeze(0).cpu()
        )

    # Combine into a matrix with shape [N, 384]
    return torch.stack(reference_features)


# ==========================================
# MAHALANOBIS DISTANCE
# ==========================================

# Calculate distance while accounting for correlations between features
def calculate_mahalanobis(features1, features2, reference_features):

    # Convert image features to CPU and double precision
    a = features1.squeeze(0).cpu().double()
    b = features2.squeeze(0).cpu().double()

    # Convert reference features to double precision
    X = reference_features.double()

    # Calculate mean and center reference features
    mean = X.mean(dim=0)
    centered = X - mean

    # Estimate covariance matrix
    covariance = (centered.T @ centered) / (X.shape[0] - 1)

    # Shrink covariance toward diagonal for numerical stability
    diagonal = torch.diag(torch.diag(covariance))

    covariance = (
        0.9 * covariance
        + 0.1 * diagonal
    )

    # Add regularization to avoid singular covariance matrices
    covariance += (
        COVARIANCE_REGULARIZATION
        * torch.eye(covariance.shape[0], dtype=torch.float64)
    )

    # Difference between the two image feature vectors
    difference = a - b

    # Solve the covariance-weighted distance calculation
    solution = torch.linalg.solve(
        covariance,
        difference
    )

    # Calculate Mahalanobis distance
    distance_squared = difference @ solution

    distance = torch.sqrt(
        torch.clamp(distance_squared, min=0)
    ).item()

    return distance


# ==========================================
# COMPARE IMAGES
# ==========================================

# Extract features and calculate similarity/distance metrics
def compare_images(image1, image2, reference_dir=None):

    print(f"\nImage 1: {image1}")
    print(f"Image 2: {image2}")

    # Extract DINOv2 features from both images
    features1 = get_features(image1)
    features2 = get_features(image2)

    # Calculate similarity metrics
    cosine = calculate_cosine(features1, features2)
    pearson = calculate_pearson(features1, features2)

    # Calculate distance metrics
    euclidean = calculate_euclidean(features1, features2)
    manhattan = calculate_manhattan(features1, features2)

    # Calculate Mahalanobis only when reference images are provided
    mahalanobis = None

    if reference_dir is not None:

        reference_features = load_reference_features(reference_dir)

        mahalanobis = calculate_mahalanobis(
            features1,
            features2,
            reference_features
        )

    # Display results
    print("\n========== DINOv2 RESULTS ==========")
    print(f"Feature dimensions: {features1.shape[1]}")

    print("\nSimilarity metrics (higher = more similar):")
    print(f"  Cosine Similarity:   {cosine:.4f}")
    print(f"  Pearson Correlation: {pearson:.4f}")

    print("\nDistance metrics (lower = more similar):")
    print(f"  Euclidean (L2):      {euclidean:.4f}")
    print(f"  Manhattan (L1):      {manhattan:.4f}")

    if mahalanobis is not None:
        print(f"  Mahalanobis:         {mahalanobis:.4f}")
    else:
        print("  Mahalanobis:         Not calculated (no references)")

    print("====================================")

    # Return results for future reporting or analysis
    return {
        "cosine_similarity": cosine,
        "euclidean_distance": euclidean,
        "manhattan_distance": manhattan,
        "pearson_correlation": pearson,
        "mahalanobis_distance": mahalanobis
    }


# ==========================================
# MAIN
# ==========================================

# Parse command-line arguments and run image comparison
def main():

    parser = argparse.ArgumentParser(
        description="Compare two images using DINOv2 features."
    )

    # Required image paths
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")

    # Optional reference folder for Mahalanobis distance
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Folder containing reference images for Mahalanobis distance"
    )

    args = parser.parse_args()

    # Run comparison
    compare_images(
        args.image1,
        args.image2,
        args.reference_dir
    )


if __name__ == "__main__":
    main()