#!/usr/bin/env python3

"""
train_classifier.py

Train a terrain classifier using manually collected images and DINOv2 features.

Dataset structure:
    dataset/
        corn/       # Images of corn plants or cornfields
        soil/       # Images of soil
        grass/      # Images of grass
        rock/       # Images of rocks

Each class should have at least 3 images; 10 or more is recommended.
The script automatically splits each class into training, validation, and
testing images. For 10 images per class, the split is 8 / 1 / 1.

Methods:
    1. k-Nearest Neighbors using cosine distance (no neural network training)
    2. Linear classifier trained on frozen DINOv2 image features

Evaluation:
    Accuracy, precision, recall, F1 score, and confusion matrix

Usage:
    python3 train_classifier.py
    python3 train_classifier.py --dataset dataset --epochs 50
    python3 train_classifier.py --output models/terrain_classifier.pth
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoImageProcessor, AutoModel


# ==========================================
# CONFIGURATION
# ==========================================

# Pretrained DINOv2 Small returns a 384-dimensional image feature vector.
MODEL_NAME = "facebook/dinov2-small"

# Names must match the four directories inside dataset/.
CLASSES = ["corn", "soil", "grass", "rock"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# For each class, 80% of images go to training, with at least one reserved
# for validation and at least one for testing.
TRAIN_RATIO = 0.80
RANDOM_SEED = 42

# Linear classifier settings.
EPOCHS = 30
LEARNING_RATE = 0.001

# Number of neighbors for cosine distance-based k-NN.
K_NEIGHBORS = 3

# Saved PyTorch model checkpoint.
MODEL_PATH = "models/terrain_classifier.pth"

# Use an NVIDIA GPU if one is available; otherwise run on the CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# DATASET VALIDATION AND SPLITTING
# ==========================================

def find_images(dataset_dir):
    """Find and verify images in each of the four class folders."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    images_by_class = {}

    for class_name in CLASSES:
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        image_paths = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        valid_paths = []
        for path in image_paths:
            try:
                # verify() identifies corrupt/non-image files before training.
                with Image.open(path) as image:
                    image.verify()
                valid_paths.append(path)
            except (OSError, ValueError, UnidentifiedImageError) as error:
                print(f"Warning: skipping {path}: {error}")

        if len(valid_paths) < 3:
            raise ValueError(
                f"{class_name} has {len(valid_paths)} valid images; "
                "at least 3 are needed (10 recommended)."
            )

        images_by_class[class_name] = valid_paths
        print(f"{class_name:>6}: {len(valid_paths)} valid images")

    return images_by_class


def split_dataset(images_by_class, seed):
    """Create reproducible, non-overlapping train/val/test splits."""
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for label, class_name in enumerate(CLASSES):
        paths = images_by_class[class_name].copy()
        rng.shuffle(paths)

        # Reserve one image for validation and testing, even for small classes.
        num_train = min(int(len(paths) * TRAIN_RATIO), len(paths) - 2)
        num_train = max(1, num_train)
        remaining = len(paths) - num_train
        num_val = max(1, remaining // 2)

        train_paths = paths[:num_train]
        val_paths = paths[num_train:num_train + num_val]
        test_paths = paths[num_train + num_val:]

        for split_name, subset in (
            ("train", train_paths),
            ("val", val_paths),
            ("test", test_paths),
        ):
            splits[split_name].extend((path, label) for path in subset)

        print(
            f"{class_name:>6}: "
            f"train={len(train_paths)}, "
            f"val={len(val_paths)}, "
            f"test={len(test_paths)}"
        )

    return splits


# ==========================================
# LOAD PRETRAINED DINOV2
# ==========================================

def load_dinov2():
    """Load the frozen DINOv2 image encoder and its image processor."""
    print(f"\nUsing device: {DEVICE}")
    print(f"Loading pretrained model: {MODEL_NAME}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # Freeze DINOv2 weights. Only the linear classifier will be trained.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return processor, model


# ==========================================
# EXTRACT DINOV2 FEATURES
# ==========================================

def extract_features(items, processor, model, split_name):
    """Convert image paths into normalized DINOv2 CLS feature vectors."""
    feature_list = []
    label_list = []

    print(f"\n========== {split_name.upper()} IMAGES ==========")
    for index, (image_path, label) in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {CLASSES[label]}: {image_path.name}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

        # Move the processor's tensors to the same device as DINOv2.
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            cls_features = outputs.last_hidden_state[:, 0, :]
            cls_features = F.normalize(cls_features, p=2, dim=1)

        feature_list.append(cls_features.squeeze(0).cpu())
        label_list.append(label)

    return (
        torch.stack(feature_list),
        torch.tensor(label_list, dtype=torch.long),
    )


# ==========================================
# EVALUATION METRICS
# ==========================================

def evaluate_results(true_labels, predictions):
    """Print accuracy and class-level statistics for the test images."""
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        true_labels,
        predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
    ))

    print("Confusion Matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(
        true_labels,
        predictions,
        labels=list(range(len(CLASSES))),
    ))


# ==========================================
# K-NEAREST NEIGHBORS
# ==========================================

def evaluate_knn(train_features, train_labels, test_features, test_labels):
    """Compare test features to labeled training features using cosine distance."""
    print("\n========== K-NN CLASSIFIER ==========")

    # The embeddings are already normalized for cosine comparisons.
    knn = KNeighborsClassifier(
        n_neighbors=min(K_NEIGHBORS, len(train_labels)),
        metric="cosine",
    )
    knn.fit(train_features.numpy(), train_labels.numpy())
    predictions = knn.predict(test_features.numpy())
    evaluate_results(test_labels.numpy(), predictions)


# ==========================================
# TRAIN LINEAR CLASSIFIER
# ==========================================

def train_classifier(train_features, train_labels,
                     val_features, val_labels, epochs):
    """Train a small linear classifier on top of frozen DINOv2 features."""
    print("\n========== TRAINING LINEAR CLASSIFIER ==========")

    # Determine feature size from the loaded model rather than hardcoding 384.
    feature_dim = train_features.shape[1]
    classifier = nn.Linear(feature_dim, len(CLASSES)).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    X_train, y_train = train_features.to(DEVICE), train_labels.to(DEVICE)
    X_val, y_val = val_features.to(DEVICE), val_labels.to(DEVICE)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        classifier.train()
        optimizer.zero_grad()

        outputs = classifier(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.inference_mode():
            val_outputs = classifier(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_predictions = val_outputs.argmax(dim=1)
            val_accuracy = (val_predictions == y_val).float().mean().item()

        # Keep the checkpoint with the lowest validation loss.
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in classifier.state_dict().items()
            }

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {loss.item():.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Accuracy: {val_accuracy * 100:.2f}%"
        )

    # This explicit check also avoids the Pylance optional-state error.
    if best_state is None:
        raise RuntimeError("No valid model state was saved during training.")

    classifier.load_state_dict(best_state)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    return classifier


# ==========================================
# TEST LINEAR CLASSIFIER
# ==========================================

def test_classifier(classifier, test_features, test_labels):
    """Evaluate the selected linear classifier on held-out test images."""
    print("\n========== LINEAR CLASSIFIER TEST ==========")
    classifier.eval()

    with torch.inference_mode():
        outputs = classifier(test_features.to(DEVICE))
        predictions = outputs.argmax(dim=1).cpu().numpy()

    evaluate_results(test_labels.numpy(), predictions)


# ==========================================
# SAVE CLASSIFIER
# ==========================================

def save_classifier(classifier, output_path):
    """Save the linear classifier and metadata needed for future predictions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": classifier.cpu().state_dict(),
        "classes": CLASSES,
        "feature_dim": classifier.in_features,
        "model_name": MODEL_NAME,
        "normalized_features": True,
    }, output_path)
    print(f"\nSaved classifier: {output_path}")


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a DINOv2 classifier from four image folders."
    )
    parser.add_argument("--dataset", default="dataset", help="Image dataset folder")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--output", default=MODEL_PATH, help="Saved model path")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    # Use the same random seed for splitting and classifier initialization.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Check image folders before downloading/loading the pretrained model.
    print("========== DATASET CHECK ==========")
    images_by_class = find_images(args.dataset)
    print("\n========== AUTOMATIC DATASET SPLIT ==========")
    splits = split_dataset(images_by_class, args.seed)

    processor, model = load_dinov2()

    # Extract features once; no DINOv2 backpropagation is performed.
    train_features, train_labels = extract_features(
        splits["train"], processor, model, "train"
    )
    val_features, val_labels = extract_features(
        splits["val"], processor, model, "validation"
    )
    test_features, test_labels = extract_features(
        splits["test"], processor, model, "test"
    )

    evaluate_knn(train_features, train_labels, test_features, test_labels)

    classifier = train_classifier(
        train_features, train_labels, val_features, val_labels, args.epochs
    )
    test_classifier(classifier, test_features, test_labels)
    save_classifier(classifier, args.output)


if __name__ == "__main__":
    main()