#!/usr/bin/env python3

"""
predict_image.py

Predict terrain categories for all images in test_images/ using a trained
linear classifier on top of frozen DINOv2 features.

Supported classes are read from models/terrain_classifier.pth (typically corn,
soil, grass, rock). The image's pixels are used for predictions; its filename
is only used to infer an expected label for optional accuracy reporting.

Examples:
    python3 predict_image.py
    python3 predict_image.py --folder test_images
    python3 predict_image.py --image test_images/corn_test.jpg
    python3 predict_image.py --folder test_images --output results/predictions.csv

Filename labels:
    corn_test.jpg       -> expected: corn
    grass_test2.png     -> expected: grass
    rock-03.jpeg        -> expected: rock
    unknown_photo.jpg   -> expected: unknown (excluded from accuracy)

Outputs:
    - One prediction row per image (expected, predicted, top score, margin)
    - Scores for each category for every image
    - Accuracy, per-class accuracy, and confusion matrix for labeled images
    - Optional CSV with all prediction scores

Note: Softmax percentages are model scores, NOT calibrated probabilities.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel


# ==========================================
# CONFIGURATION
# ==========================================

CLASSIFIER_PATH = "models/terrain_classifier.pth"
TEST_IMAGES_DIR = "test_images"
CSV_OUTPUT_PATH = "results/predictions.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# LOAD TRAINED CLASSIFIER
# ==========================================

def load_classifier(checkpoint_path):
    """
    Restore classifier weights and settings saved by train_classifier.py.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run python3 train_classifier.py first."
        )

    # Load only checkpoints from trusted sources.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    classes = list(checkpoint["classes"])
    feature_dim = int(checkpoint["feature_dim"])
    model_name = str(checkpoint["model_name"])
    normalized_features = bool(
        checkpoint.get("normalized_features", True)
    )

    # Recreate the linear classification layer.
    classifier = nn.Linear(feature_dim, len(classes))

    # Restore learned weights and biases.
    classifier.load_state_dict(
        checkpoint["model_state_dict"]
    )

    classifier = classifier.to(DEVICE)
    classifier.eval()

    print(f"Classifier: {checkpoint_path}")
    print(f"DINOv2 model: {model_name}")
    print(f"Classes: {', '.join(classes)}")
    print(f"Device: {DEVICE}")

    return classifier, classes, model_name, normalized_features


# ==========================================
# LOAD PRETRAINED DINOV2
# ==========================================

def load_dinov2(model_name):
    """
    Load DINOv2 once for all test images.
    """

    print("\nLoading pretrained DINOv2 ...")

    processor = AutoImageProcessor.from_pretrained(model_name)

    model = AutoModel.from_pretrained(
        model_name
    ).to(DEVICE)

    model.eval()

    return processor, model


# ==========================================
# FIND TEST IMAGES AND EXPECTED LABELS
# ==========================================

def find_test_images(folder=None, single_image=None):
    """
    Find all supported images inside a folder or select one image.
    """

    if single_image:
        path = Path(single_image)

        if not path.is_file():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension: {path.suffix}"
            )

        return [path]

    folder_path = Path(folder or TEST_IMAGES_DIR)

    if not folder_path.is_dir():
        raise FileNotFoundError(
            f"Test image folder not found: {folder_path}"
        )

    # Find images, including images inside subfolders.
    image_paths = sorted(
        path for path in folder_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(
            f"No supported images found in {folder_path}"
        )

    return image_paths


def expected_label(image_path, classes):
    """
    Determine the expected category from the image filename.

    Examples:
        corn_test.jpg  -> corn
        grass_02.jpg   -> grass
        rock-03.jpg    -> rock

    Unrecognized filenames return None and are excluded from accuracy.
    """

    stem = image_path.stem.lower()

    for class_name in classes:
        class_lower = class_name.lower()

        if (
            stem == class_lower
            or stem.startswith((
                class_lower + "_",
                class_lower + "-",
                class_lower + " "
            ))
        ):
            return class_name

    return None


# ==========================================
# EXTRACT IMAGE FEATURES
# ==========================================

def extract_features(
    image_path,
    processor,
    model,
    normalized_features
):
    """
    Extract DINOv2 features from an image.

    Uses the same CLS token and normalization as train_classifier.py.
    """

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

        inputs = processor(
            images=rgb_image,
            return_tensors="pt"
        )

    # Move processed image tensors to GPU or CPU.
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(**inputs)

        # Extract the image-level CLS feature vector.
        features = outputs.last_hidden_state[:, 0, :]

        # Match the normalization used during training.
        if normalized_features:
            features = F.normalize(
                features,
                p=2,
                dim=1
            )

    return features


# ==========================================
# PREDICT A SINGLE IMAGE
# ==========================================

def predict_one(
    image_path,
    processor,
    model,
    classifier,
    classes,
    normalized_features
):
    """
    Predict an image category and calculate classification scores.

    Returns:
        - Image path
        - Expected class
        - Predicted class
        - Highest class score
        - Margin between top two scores
        - Scores for every class
    """

    features = extract_features(
        image_path,
        processor,
        model,
        normalized_features
    )

    with torch.inference_mode():

        # Apply the trained linear classifier.
        logits = classifier(features)

        # Convert logits to softmax scores.
        probabilities = F.softmax(
            logits,
            dim=1
        )[0].cpu().tolist()

    # Sort class indices from highest to lowest score.
    ranked = sorted(
        range(len(classes)),
        key=lambda i: probabilities[i],
        reverse=True
    )

    best_idx = ranked[0]

    # Calculate difference between top two scores.
    if len(ranked) > 1:
        margin = (
            probabilities[ranked[0]]
            - probabilities[ranked[1]]
        )
    else:
        margin = 1.0

    return {
        "file": str(image_path),
        "expected": expected_label(image_path, classes),
        "predicted": classes[best_idx],
        "top_score": probabilities[best_idx],
        "margin": margin,
        "scores": dict(zip(classes, probabilities))
    }


# ==========================================
# DISPLAY PREDICTIONS
# ==========================================

def display_results(results, classes):
    """
    Display a prediction summary, class scores, and test statistics.
    """

    print("\n" + "=" * 87)
    print("DINOv2 TERRAIN CLASSIFICATION RESULTS")
    print("=" * 87)

    print(
        f"{'Image':<28} "
        f"{'Expected':<12} "
        f"{'Predicted':<12} "
        f"{'Top score':>10} "
        f"{'Margin':>9} "
        f"{'Result':>8}"
    )

    print("-" * 87)

    # Display one summary row for each image.
    for row in results:

        expected = row["expected"]

        if expected is None:
            status = "N/A"
        elif expected == row["predicted"]:
            status = "PASS"
        else:
            status = "FAIL"

        short_name = Path(row["file"]).name

        if len(short_name) > 27:
            short_name = short_name[:24] + "..."

        print(
            f"{short_name:<28} "
            f"{(expected or 'unknown'):<12} "
            f"{row['predicted']:<12} "
            f"{100 * row['top_score']:>9.1f}% "
            f"{100 * row['margin']:>8.1f}% "
            f"{status:>8}"
        )

    # Show every class score for each image.
    print("\nFULL CLASS SCORES")
    print("(Softmax scores, not calibrated probabilities)")
    print("-" * 87)

    for row in results:

        ordered_scores = sorted(
            row["scores"].items(),
            key=lambda item: item[1],
            reverse=True
        )

        score_text = " | ".join(
            f"{name}: {value * 100:.1f}%"
            for name, value in ordered_scores
        )

        print(
            f"{Path(row['file']).name} "
            f"-> {row['predicted']} "
            f"({score_text})"
        )

    # Only evaluate images with recognizable filename labels.
    labeled = [
        row for row in results
        if row["expected"] is not None
    ]

    print("\n" + "=" * 87)
    print("TEST ANALYSIS")
    print("=" * 87)

    if not labeled:
        print(
            "No recognized labels in filenames. "
            "Predictions displayed without accuracy statistics."
        )
        return

    correct = sum(
        row["expected"] == row["predicted"]
        for row in labeled
    )

    accuracy = 100 * correct / len(labeled)

    print(
        f"Overall accuracy: "
        f"{correct}/{len(labeled)} = {accuracy:.2f}%"
    )

    print(
        f"Unlabeled images (not counted): "
        f"{len(results) - len(labeled)}"
    )

    # Calculate accuracy for each class independently.
    print("\nPer-class accuracy:")

    for class_name in classes:

        subset = [
            row for row in labeled
            if row["expected"] == class_name
        ]

        if not subset:
            print(
                f"  {class_name:<12} "
                "N/A (no labeled images)"
            )
            continue

        class_correct = sum(
            row["predicted"] == class_name
            for row in subset
        )

        class_accuracy = 100 * class_correct / len(subset)

        print(
            f"  {class_name:<12} "
            f"{class_correct}/{len(subset)} "
            f"= {class_accuracy:.1f}%"
        )

    # Create a confusion matrix using expected and predicted labels.
    print("\nConfusion matrix (rows=expected, columns=predicted):")

    label_width = max(
        10,
        *(len(name) + 2 for name in classes)
    )

    print(
        " " * label_width
        + "".join(
            f"{name:>{label_width}}"
            for name in classes
        )
    )

    counts = Counter(
        (row["expected"], row["predicted"])
        for row in labeled
    )

    for actual in classes:

        values = "".join(
            f"{counts[(actual, predicted)]:>{label_width}}"
            for predicted in classes
        )

        print(f"{actual:<{label_width}}" + values)

    print(
        "\nNote: With only a few test images, "
        "accuracy can vary substantially."
    )


# ==========================================
# EXPORT PREDICTIONS TO CSV
# ==========================================

def save_csv(results, classes, output_path):
    """
    Save predictions and all class scores to a CSV file.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "image",
        "expected",
        "predicted",
        "correct",
        "top_score",
        "margin"
    ]

    fieldnames.extend(
        f"score_{class_name}"
        for class_name in classes
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in results:

            expected = row["expected"]

            item = {
                "image": row["file"],
                "expected": expected or "",
                "predicted": row["predicted"],
                "correct": (
                    ""
                    if expected is None
                    else expected == row["predicted"]
                ),
                "top_score": round(
                    row["top_score"], 6
                ),
                "margin": round(
                    row["margin"], 6
                )
            }

            # Add individual score columns.
            item.update({
                f"score_{name}": round(
                    row["scores"][name], 6
                )
                for name in classes
            })

            writer.writerow(item)

    print(f"\nSaved CSV: {output_path}")


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Classify test images using DINOv2 "
            "and a trained terrain classifier."
        )
    )

    parser.add_argument(
        "--folder",
        default=TEST_IMAGES_DIR,
        help="Folder of images (default: test_images)"
    )

    parser.add_argument(
        "--image",
        help="Optional single image; overrides --folder"
    )

    parser.add_argument(
        "--classifier",
        default=CLASSIFIER_PATH,
        help="Saved terrain classifier checkpoint"
    )

    parser.add_argument(
        "--output",
        default=CSV_OUTPUT_PATH,
        help="CSV output path"
    )

    args = parser.parse_args()

    # Step 1: Find all test images.
    paths = find_test_images(
        args.folder,
        args.image
    )

    # Step 2: Load trained classifier.
    classifier, classes, model_name, normalized_features = (
        load_classifier(args.classifier)
    )

    # Step 3: Load pretrained DINOv2 once.
    processor, model = load_dinov2(model_name)

    results = []

    print(
        f"\nFound {len(paths)} image(s). "
        "Starting predictions..."
    )

    # Step 4: Classify every test image.
    for index, image_path in enumerate(paths, start=1):

        try:
            result = predict_one(
                image_path,
                processor,
                model,
                classifier,
                classes,
                normalized_features
            )

        except (OSError, ValueError, UnidentifiedImageError) as error:
            print(
                f"[{index}/{len(paths)}] "
                f"SKIPPED {image_path.name}: {error}"
            )
            continue

        results.append(result)

        print(
            f"[{index}/{len(paths)}] "
            f"{image_path.name} "
            f"-> {result['predicted']}"
        )

    if not results:
        raise RuntimeError(
            "No images were successfully classified. "
            "Check your test images."
        )

    # Step 5: Display results and test analysis.
    display_results(
        results,
        classes
    )

    # Step 6: Save predictions to CSV.
    save_csv(
        results,
        classes,
        args.output
    )


if __name__ == "__main__":
    main()