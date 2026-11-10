#!/usr/bin/env python3

"""
train_classifier.py

Train an image classifier using DINOv2's pretrained visual features.

Classes:
    1. Corn
    2. Soil
    3. Grass
    4. Rock

Methods:
    1. K-Nearest Neighbors (Cosine Similarity)
    2. Linear Classifier (Cross-Entropy Loss)

Evaluation:
    1. Accuracy
    2. Precision
    3. Recall
    4. F1 Score
    5. Confusion Matrix

Usage:
    python3 train_classifier.py
    python3 train_classifier.py --epochs 50
    python3 train_classifier.py --dataset dataset
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# CONFIGURATION
# ==========================================

# Pretrained DINOv2 Small model (384 features)
MODEL_NAME = "facebook/dinov2-small"

# Use NVIDIA GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Classification categories
CLASSES = ["corn", "soil", "grass", "rock"]

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Model configuration
FEATURE_DIM = 384
NUM_CLASSES = len(CLASSES)

# Training configuration
EPOCHS = 30
LEARNING_RATE = 0.001

# Number of neighbors for k-NN classification
K_NEIGHBORS = 3

# Saved classifier location
MODEL_PATH = "models/terrain_classifier.pth"


# ==========================================
# LOAD DINOV2
# ==========================================

# Load pretrained DINOv2 for feature extraction
def load_dinov2():

    print(f"Using device: {device}")
    print("Loading DINOv2...")

    # Load image processor and pretrained model
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    # Move model to GPU/CPU and enable evaluation mode
    model = model.to(device)
    model.eval()

    print("DINOv2 loaded successfully!")

    return processor, model


# ==========================================
# EXTRACT FEATURES
# ==========================================

# Extract the 384-dimensional CLS feature vector from an image
def get_features(image_path, processor, model):

    # Load image and convert to RGB
    with Image.open(image_path) as image:
        image = image.convert("RGB")

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

    # Normalize feature vector for cosine similarity
    features = F.normalize(features, p=2, dim=1)

    return features


# ==========================================
# LOAD DATASET
# ==========================================

# Extract image features and labels from each category folder
def load_dataset(dataset_dir, processor, model):

    dataset_dir = Path(dataset_dir)

    # Verify dataset folder exists
    if not dataset_dir.is_dir():
        raise ValueError(
            f"Dataset directory does not exist: {dataset_dir}"
        )

    features_list = []
    labels_list = []

    # Process each classification category
    for label, class_name in enumerate(CLASSES):

        class_dir = dataset_dir / class_name

        # Verify category folder exists
        if not class_dir.is_dir():
            raise ValueError(
                f"Missing class directory: {class_dir}"
            )

        # Find all supported image files
        image_paths = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"\nClass: {class_name}")
        print(f"Images: {len(image_paths)}")

        # Extract features from each image
        for index, image_path in enumerate(image_paths, start=1):

            print(
                f"[{index}/{len(image_paths)}] "
                f"Processing: {image_path.name}"
            )

            features = get_features(
                image_path,
                processor,
                model
            )

            # Move features to CPU
            features_list.append(
                features.squeeze(0).cpu()
            )

            labels_list.append(label)

    # Verify images were loaded
    if not features_list:
        raise ValueError(
            f"No supported images found in: {dataset_dir}"
        )

    # Combine features into matrix [N, 384]
    features = torch.stack(features_list)

    # Convert labels to tensor
    labels = torch.tensor(
        labels_list,
        dtype=torch.long
    )

    return features, labels


# ==========================================
# K-NEAREST NEIGHBORS
# ==========================================

# Classify images using cosine distance between DINOv2 features
def evaluate_knn(train_features, train_labels,
                 test_features, test_labels):

    print("\n========== K-NN CLASSIFIER ==========")

    # Convert feature tensors to NumPy arrays
    X_train = train_features.numpy()
    y_train = train_labels.numpy()

    X_test = test_features.numpy()
    y_test = test_labels.numpy()

    # Create k-NN classifier using cosine distance
    knn = KNeighborsClassifier(
        n_neighbors=min(K_NEIGHBORS, len(X_train)),
        metric="cosine"
    )

    # Store labeled training features
    knn.fit(X_train, y_train)

    # Predict test image categories
    predictions = knn.predict(X_test)

    # Calculate evaluation metrics
    evaluate_results(y_test, predictions)

    return knn


# ==========================================
# LINEAR CLASSIFIER
# ==========================================

# Create a trainable linear layer for four categories
def create_classifier():

    # Convert 384 DINOv2 features into 4 class scores
    classifier = nn.Linear(
        FEATURE_DIM,
        NUM_CLASSES
    )

    return classifier.to(device)


# ==========================================
# TRAIN LINEAR CLASSIFIER
# ==========================================

# Train the classification layer using labeled DINOv2 features
def train_classifier(train_features, train_labels,
                     val_features, val_labels, epochs):

    print("\n========== TRAINING CLASSIFIER ==========")

    # Create classification layer
    classifier = create_classifier()

    # Cross-entropy loss for multiclass classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer updates classifier weights
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE
    )

    # Move training data to GPU/CPU
    X_train = train_features.to(device)
    y_train = train_labels.to(device)

    X_val = val_features.to(device)
    y_val = val_labels.to(device)

    best_accuracy = -1.0
    best_state = None

    # Train classifier for specified epochs
    for epoch in range(epochs):

        classifier.train()

        # Reset gradients
        optimizer.zero_grad()

        # Generate class scores
        outputs = classifier(X_train)

        # Calculate classification loss
        loss = criterion(outputs, y_train)

        # Calculate gradients and update weights
        loss.backward()
        optimizer.step()

        # Evaluate classifier using validation images
        classifier.eval()

        with torch.inference_mode():

            val_outputs = classifier(X_val)

            predictions = torch.argmax(
                val_outputs,
                dim=1
            )

            accuracy = (
                (predictions == y_val)
                .float()
                .mean()
                .item()
            )

        # Save best classifier weights in memory
        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_state = {
                name: value.detach().cpu().clone()
                for name, value in classifier.state_dict().items()
            }

        # Display training progress
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.4f} "
            f"Val Accuracy: {accuracy * 100:.2f}%"
        )

    # Restore best validation model
    if best_state is not None:

        classifier.load_state_dict(best_state)

    else:

        raise RuntimeError(
            "Training failed: No valid classifier state was saved."
        )

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")

    return classifier


# ==========================================
# EVALUATE RESULTS
# ==========================================

# Calculate accuracy, precision, recall, F1 score, and confusion matrix
def evaluate_results(true_labels, predictions):

    # Calculate overall classification accuracy
    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # Display precision, recall, and F1 score
    print("\nClassification Report:")

    print(
        classification_report(
            true_labels,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASSES,
            zero_division=0
        )
    )

    # Display confusion matrix
    print("Confusion Matrix:")

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=list(range(NUM_CLASSES))
    )

    print(matrix)


# ==========================================
# TEST LINEAR CLASSIFIER
# ==========================================

# Evaluate trained classifier using unseen test images
def test_classifier(classifier, test_features, test_labels):

    print("\n========== LINEAR CLASSIFIER ==========")

    classifier.eval()

    # Move test data to GPU/CPU
    X_test = test_features.to(device)

    # Run inference without calculating gradients
    with torch.inference_mode():

        outputs = classifier(X_test)

        # Select class with highest score
        predictions = torch.argmax(
            outputs,
            dim=1
        )

    # Move predictions to CPU
    predictions = predictions.cpu().numpy()

    true_labels = test_labels.numpy()

    # Calculate evaluation metrics
    evaluate_results(true_labels, predictions)


# ==========================================
# SAVE CLASSIFIER
# ==========================================

# Save trained classifier weights and class names
def save_classifier(classifier, model_path):

    model_path = Path(model_path)

    # Create models folder if needed
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save classifier and model configuration
    torch.save(
        {
            "model_state_dict": classifier.cpu().state_dict(),
            "classes": CLASSES,
            "feature_dim": FEATURE_DIM,
            "model_name": MODEL_NAME,
            "normalized_features": True
        },
        model_path
    )

    print(f"\nClassifier saved: {model_path}")


# ==========================================
# MAIN
# ==========================================

# Parse command-line arguments and train image classifier
def main():

    parser = argparse.ArgumentParser(
        description="Train image classifier using DINOv2 features."
    )

    # Dataset folder containing train, val, and test
    parser.add_argument(
        "--dataset",
        default="dataset",
        help="Path to dataset directory"
    )

    # Number of training epochs
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs"
    )

    # Saved classifier location
    parser.add_argument(
        "--output",
        default=MODEL_PATH,
        help="Path to save trained classifier"
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    # Load pretrained DINOv2
    processor, model = load_dinov2()

    dataset_dir = Path(args.dataset)

    # Extract features from training images
    print("\n========== TRAINING DATA ==========")

    train_features, train_labels = load_dataset(
        dataset_dir / "train",
        processor,
        model
    )

    # Extract features from validation images
    print("\n========== VALIDATION DATA ==========")

    val_features, val_labels = load_dataset(
        dataset_dir / "val",
        processor,
        model
    )

    # Extract features from testing images
    print("\n========== TEST DATA ==========")

    test_features, test_labels = load_dataset(
        dataset_dir / "test",
        processor,
        model
    )

    # Verify every class has training, validation, and test images
    for split_name, labels in [
        ("train", train_labels),
        ("val", val_labels),
        ("test", test_labels)
    ]:

        counts = torch.bincount(
            labels,
            minlength=NUM_CLASSES
        )

        if torch.any(counts == 0):
            raise ValueError(
                f"{split_name} must contain images for all four classes."
            )

    # Evaluate cosine similarity-based k-NN on test images
    evaluate_knn(
        train_features,
        train_labels,
        test_features,
        test_labels
    )

    # Train linear classifier
    classifier = train_classifier(
        train_features,
        train_labels,
        val_features,
        val_labels,
        args.epochs
    )

    # Evaluate trained classifier
    test_classifier(
        classifier,
        test_features,
        test_labels
    )

    # Save trained classifier
    save_classifier(
        classifier,
        args.output
    )


if __name__ == "__main__":
    main()