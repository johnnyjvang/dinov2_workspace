#!/usr/bin/env python3

"""
download_dataset.py

Download images from Wikimedia Commons and filter them using DINOv2.

Classes:
    1. Corn
    2. Soil
    3. Grass
    4. Rock

Process:
    1. Download up to 20 candidate images per class.
    2. Cache downloaded images locally.
    3. Extract DINOv2 features.
    4. Compare every image against all other images.
    5. Calculate within-class and between-class similarity.
    6. Rank images using similarity margins.
    7. Remove ambiguous and duplicate images.
    8. Keep up to 15 images per class.
    9. Split into train, validation, and test datasets.
    10. Save similarity results to CSV.

Usage:
    python3 download_dataset.py

Note:
    Cached images are reused between runs.
    Existing dataset is replaced only after successful processing.
"""

import csv
import hashlib
import io
import json
import random
import re
import shutil
import tempfile
import time

from pathlib import Path

import requests
import torch
import torch.nn.functional as F

from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "facebook/dinov2-small"

device = "cuda" if torch.cuda.is_available() else "cpu"

API_URL = "https://commons.wikimedia.org/w/api.php"

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
CACHE_DIR = BASE_DIR / "candidate_images"

# Maximum candidate images per class
IMAGES_PER_CLASS = 20

# Maximum images kept per class
KEEP_IMAGES = 15

# Minimum images required for train, val, and test
MIN_IMAGES = 3

# Number of nearest neighbors for similarity
K_NEIGHBORS = 3

# Minimum similarity margin
MIN_MARGIN = 0.0

# Duplicate similarity threshold
DUPLICATE_THRESHOLD = 0.97

# Image configuration
IMAGE_SIZE = 512

# Wikimedia search configuration
SEARCH_LIMIT = 20
MAX_SEARCH_PAGES = 2

# HTTP request configuration
TIMEOUT = 30
REQUEST_DELAY = 2.0
MAX_RETRIES = 3
RETRY_DELAY = 5.0
MAX_RETRY_DELAY = 60.0

# Random seed
SEED = 42

# Search queries
CLASSES = {
    "corn": [
        "intitle:maize",
        "intitle:corn",
        '"corn field"',
        '"maize plant"',
    ],

    "soil": [
        "intitle:soil",
        "intitle:dirt",
        '"garden soil"',
        '"agricultural soil"',
    ],

    "grass": [
        "intitle:grass",
        "intitle:lawn",
        '"green grass"',
        '"grass field"',
    ],

    "rock": [
        "intitle:rock",
        "intitle:stone",
        "intitle:gravel",
        "intitle:pebble",
    ],
}

# Required keywords in filenames
KEYWORDS = {
    "corn": [
        "corn", "maize", "zea mays"
    ],

    "soil": [
        "soil", "dirt", "earth"
    ],

    "grass": [
        "grass", "lawn", "meadow"
    ],

    "rock": [
        "rock", "rocks", "stone", "stones",
        "gravel", "pebble", "pebbles"
    ],
}

# Reject irrelevant filenames
EXCLUDE_WORDS = [
    "diagram",
    "chart",
    "logo",
    "icon",
    "drawing",
    "illustration",
    "painting",
    "poster",
    "screenshot",
    "micrograph",
    "cornwall",
    "popcorn",
    "cornstarch",
    "rock band",
    "rock music",
]

HEADERS = {
    "User-Agent": (
        "DINOv2DatasetResearch/4.0 "
        "(educational computer vision project)"
    )
}


# ==========================================
# RATE LIMIT HANDLING
# ==========================================

# Custom exception for Wikimedia rate limits
class RateLimitError(Exception):
    pass


# Get server-requested retry delay
def get_retry_delay(response, attempt):

    retry_after = response.headers.get("Retry-After")

    if retry_after:

        try:
            return max(0.0, float(retry_after))

        except ValueError:

            from datetime import datetime, timezone
            from email.utils import parsedate_to_datetime

            try:

                retry_date = parsedate_to_datetime(
                    retry_after
                )

                if retry_date.tzinfo is None:

                    retry_date = retry_date.replace(
                        tzinfo=timezone.utc
                    )

                delay = (
                    retry_date - datetime.now(timezone.utc)
                ).total_seconds()

                return max(0.0, delay)

            except (ValueError, TypeError, OverflowError):
                pass

    # Exponential backoff when no valid Retry-After
    return RETRY_DELAY * (2 ** attempt)


# ==========================================
# SAFE HTTP REQUEST
# ==========================================

# Send HTTP requests and handle rate limits
def safe_request(session, url, params=None):

    for attempt in range(MAX_RETRIES):

        try:

            response = session.get(
                url,
                params=params,
                timeout=TIMEOUT
            )

        except (
            requests.ConnectionError,
            requests.Timeout
        ) as error:

            if attempt == MAX_RETRIES - 1:
                raise

            delay = RETRY_DELAY * (2 ** attempt)

            print(
                f"\nConnection error: {error}"
            )

            print(
                f"Retrying in {delay:.1f} seconds..."
            )

            time.sleep(delay)
            continue

        # Handle rate limiting
        if response.status_code == 429:

            delay = get_retry_delay(
                response,
                attempt
            )

            print("\nHTTP 429: Too Many Requests")

            response.close()

            # Stop if server requests a long cooldown
            if delay > MAX_RETRY_DELAY:

                raise RateLimitError(
                    f"Wikimedia requested a "
                    f"{delay:.0f}-second cooldown."
                )

            # Stop after maximum retries
            if attempt == MAX_RETRIES - 1:

                raise RateLimitError(
                    "Wikimedia is still rate-limiting requests."
                )

            print(
                f"Retrying in {delay:.1f} seconds..."
            )

            time.sleep(delay)
            continue

        # Handle temporary server errors
        if response.status_code in (500, 502, 503, 504):

            delay = get_retry_delay(
                response,
                attempt
            )

            if delay > MAX_RETRY_DELAY:

                response.close()

                raise RateLimitError(
                    "Server requested a long retry delay."
                )

            if attempt == MAX_RETRIES - 1:
                response.raise_for_status()

            print(
                f"\nHTTP {response.status_code}. "
                f"Retrying in {delay:.1f} seconds..."
            )

            response.close()

            time.sleep(delay)
            continue

        # Raise unexpected HTTP errors
        response.raise_for_status()

        # Slow down requests
        time.sleep(REQUEST_DELAY)

        return response

    raise RuntimeError(
        "Maximum HTTP retries exceeded."
    )


# ==========================================
# LOAD DINOV2
# ==========================================

# Load pretrained DINOv2
def load_dinov2():

    print(f"Using device: {device}")
    print("Loading DINOv2...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model = model.to(device)

    model.eval()

    print("DINOv2 loaded successfully!")

    return processor, model


# ==========================================
# EXTRACT FEATURES
# ==========================================

# Extract normalized CLS features
def get_features(image, processor, model):

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():

        outputs = model(**inputs)

    # Extract CLS token
    features = outputs.last_hidden_state[:, 0, :]

    # Normalize features
    features = F.normalize(
        features,
        p=2,
        dim=1
    )

    return features.squeeze(0).cpu()


# ==========================================
# SEARCH IMAGES
# ==========================================

# Search Wikimedia Commons progressively
def search_images(session, query):

    continuation = {}

    for _ in range(MAX_SEARCH_PAGES):

        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": 6,
            "gsrlimit": SEARCH_LIMIT,
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": 600,
            "format": "json",
            "formatversion": 2,
        }

        params.update(continuation)

        response = safe_request(
            session,
            API_URL,
            params=params
        )

        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        pages = data.get(
            "query", {}
        ).get("pages", [])

        # Yield results one at a time
        for page in pages:
            yield page

        if "continue" not in data:
            break

        continuation = data["continue"]


# ==========================================
# FILTER IMAGE TITLES
# ==========================================

# Check whether filename matches category
def valid_title(title, class_name):

    title = title.lower()

    title = re.sub(
        r"^file:",
        "",
        title
    )

    title = re.sub(
        r"\.[^.]+$",
        "",
        title
    )

    title = re.sub(
        r"[_\-]+",
        " ",
        title
    )

    # Reject irrelevant filenames
    for word in EXCLUDE_WORDS:

        if re.search(
            rf"\b{re.escape(word)}\b",
            title
        ):
            return False

    # Require class keyword
    return any(
        re.search(
            rf"\b{re.escape(word)}\b",
            title
        )
        for word in KEYWORDS[class_name]
    )


# ==========================================
# DOWNLOAD IMAGE
# ==========================================

# Download and prepare image
def download_image(session, url):

    response = safe_request(
        session,
        url
    )

    with Image.open(
        io.BytesIO(response.content)
    ) as image:

        # Correct orientation
        image = ImageOps.exif_transpose(image)

        # Convert to RGB
        image = image.convert("RGB")

        # Reject small images
        if min(image.size) < 160:

            raise ValueError(
                "Image too small"
            )

        image.thumbnail(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        return image.copy()


# ==========================================
# LOAD CACHED IMAGES
# ==========================================

# Load previously downloaded images
def load_cached_images(class_name):

    class_dir = CACHE_DIR / class_name

    class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    images = []

    metadata_files = sorted(
        class_dir.glob("*.json")
    )

    for metadata_path in metadata_files:

        if len(images) >= IMAGES_PER_CLASS:
            break

        try:

            with metadata_path.open(
                "r",
                encoding="utf-8"
            ) as file:

                metadata = json.load(file)

            image_path = metadata_path.with_suffix(
                ".jpg"
            )

            if not image_path.exists():
                continue

            if metadata.get("class") != class_name:
                continue

            if not valid_title(
                metadata.get("title", ""),
                class_name
            ):
                continue

            with Image.open(image_path) as image:

                image = image.convert("RGB").copy()

            images.append({
                "image": image,
                "class": class_name,
                "title": metadata["title"],
                "url": metadata["url"],
                "query": metadata["query"],
                "page_id": metadata["page_id"],
                "hash": metadata["hash"],
            })

        except (
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError
        ) as error:

            print(
                f"Skipping invalid cache: {error}"
            )

    return images


# ==========================================
# SAVE CACHED IMAGE
# ==========================================

# Save downloaded image immediately
def save_cached_image(item):

    class_dir = CACHE_DIR / item["class"]

    class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = str(item["page_id"])

    image_path = class_dir / f"{filename}.jpg"

    metadata_path = class_dir / f"{filename}.json"

    # Save image
    item["image"].save(
        image_path,
        "JPEG",
        quality=90
    )

    # Save metadata
    metadata = {
        "class": item["class"],
        "title": item["title"],
        "url": item["url"],
        "query": item["query"],
        "page_id": item["page_id"],
        "hash": item["hash"],
    }

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


# ==========================================
# COLLECT CANDIDATE IMAGES
# ==========================================

# Load cached images and download missing candidates
def collect_images(session, class_name, seen_hashes):

    print(
        f"\n========== {class_name.upper()} =========="
    )

    # Load cached images
    cached_images = load_cached_images(
        class_name
    )

    images = []

    seen_pages = set()

    # Reuse cached candidates
    for item in cached_images:

        page_id = item["page_id"]
        image_hash = item["hash"]

        if image_hash in seen_hashes:
            continue

        seen_hashes.add(image_hash)

        seen_pages.add(page_id)

        images.append(item)

    print(
        f"Loaded {len(images)} cached images."
    )

    # Stop immediately if cache is full
    if len(images) >= IMAGES_PER_CLASS:

        print(
            "Candidate limit reached using cache."
        )

        return images

    # Search for missing candidates
    for query in CLASSES[class_name]:

        if len(images) >= IMAGES_PER_CLASS:
            break

        print(f"\nSearching: {query}")

        for page in search_images(session, query):

            if len(images) >= IMAGES_PER_CLASS:
                break

            page_id = page.get("pageid")

            title = page.get("title", "")

            # Skip duplicate pages
            if page_id in seen_pages:
                continue

            seen_pages.add(page_id)

            # Filter filenames
            if not valid_title(
                title,
                class_name
            ):
                continue

            info = (
                page.get("imageinfo") or [{}]
            )[0]

            url = (
                info.get("thumburl")
                or info.get("url")
            )

            mime = info.get("mime", "")

            if not url or mime not in (
                "image/jpeg",
                "image/png",
                "image/webp",
                "image/tiff",
            ):
                continue

            try:

                image = download_image(
                    session,
                    url
                )

            # Rate-limit errors stop the whole run
            except RateLimitError:
                raise

            except (
                requests.RequestException,
                OSError,
                ValueError
            ) as error:

                print(
                    f"Skipping: {title} ({error})"
                )

                continue

            # Calculate image hash
            image_hash = hashlib.sha256(
                image.tobytes()
            ).hexdigest()

            if image_hash in seen_hashes:
                continue

            seen_hashes.add(image_hash)

            item = {
                "image": image,
                "class": class_name,
                "title": title,
                "url": info.get(
                    "descriptionurl",
                    url
                ),
                "query": query,
                "page_id": page_id,
                "hash": image_hash,
            }

            images.append(item)

            # Save candidate immediately
            save_cached_image(item)

            print(
                f"Downloaded "
                f"[{len(images)}/{IMAGES_PER_CLASS}]: "
                f"{title}"
            )

    print(
        f"\nCollected {len(images)} candidates."
    )

    if len(images) < MIN_IMAGES:

        raise RuntimeError(
            f"Not enough candidates for {class_name}."
        )

    return images


# ==========================================
# EXTRACT DATASET FEATURES
# ==========================================

# Extract DINOv2 features from all candidates
def extract_dataset_features(images, processor, model):

    print(
        "\n========== EXTRACTING FEATURES =========="
    )

    features_list = []

    total = len(images)

    for index, item in enumerate(images):

        features = get_features(
            item["image"],
            processor,
            model
        )

        features_list.append(features)

        if (
            (index + 1) % 10 == 0
            or index + 1 == total
        ):

            print(
                f"Extracting features: "
                f"{index + 1}/{total}"
            )

    features = torch.stack(
        features_list
    )

    print(
        f"Feature matrix: {features.shape}"
    )

    return features


# ==========================================
# CALCULATE SIMILARITY MATRIX
# ==========================================

# Compare all images against all other images
def calculate_similarity_matrix(features):

    print(
        "\n========== ALL-TO-ALL SIMILARITY =========="
    )

    # Normalize features
    features = F.normalize(
        features,
        p=2,
        dim=1
    )

    # Calculate cosine similarity
    similarity_matrix = features @ features.T

    print(
        f"Similarity matrix: {similarity_matrix.shape}"
    )

    return similarity_matrix


# ==========================================
# TOP-K SIMILARITY
# ==========================================

# Calculate average of nearest neighbors
def top_k_similarity(scores, k):

    if scores.numel() == 0:
        return 0.0

    k = min(
        k,
        int(scores.numel())
    )

    top_scores = torch.topk(
        scores,
        k=k
    ).values

    return float(
        top_scores.mean().item()
    )


# ==========================================
# CALCULATE CLASS SCORES
# ==========================================

# Compare each image with all four categories
def calculate_class_scores(images, similarity_matrix):

    print(
        "\n========== CLASS SIMILARITY SCORES =========="
    )

    class_indices = {}

    # Group indices by category
    for class_name in CLASSES:

        class_indices[class_name] = [
            index
            for index, item in enumerate(images)
            if item["class"] == class_name
        ]

    # Calculate scores
    for index, item in enumerate(images):

        assigned_class = item["class"]

        class_scores = {}

        for class_name in CLASSES:

            # Exclude self-comparison
            indices = [
                other_index
                for other_index in class_indices[class_name]
                if other_index != index
            ]

            if not indices:

                class_scores[class_name] = 0.0
                continue

            scores = similarity_matrix[
                index,
                torch.tensor(
                    indices,
                    dtype=torch.long
                )
            ]

            class_scores[class_name] = top_k_similarity(
                scores,
                K_NEIGHBORS
            )

        within_similarity = class_scores[
            assigned_class
        ]

        other_scores = {
            name: score
            for name, score in class_scores.items()
            if name != assigned_class
        }

        best_other_class = max(
            other_scores,
            key=lambda name: other_scores[name]
        )

        best_other_similarity = other_scores[
            best_other_class
        ]

        margin = (
            within_similarity
            - best_other_similarity
        )

        # Save scores
        item["within_similarity"] = within_similarity
        item["other_similarity"] = best_other_similarity
        item["best_other_class"] = best_other_class
        item["margin"] = margin
        item["class_scores"] = class_scores

    print("Class similarity scores calculated!")


# ==========================================
# SELECT BEST IMAGES
# ==========================================

# Rank candidates and reject ambiguous or duplicate images
def select_best_images(images, similarity_matrix):

    print(
        "\n========== SELECTING BEST IMAGES =========="
    )

    selected_indices = set()

    for class_name in CLASSES:

        print(f"\nClass: {class_name.upper()}")

        indices = [
            index
            for index, item in enumerate(images)
            if item["class"] == class_name
        ]

        # Rank by similarity margin
        indices.sort(
            key=lambda index: (
                images[index]["margin"],
                images[index]["within_similarity"]
            ),
            reverse=True
        )

        accepted = []

        for index in indices:

            item = images[index]

            # Reject ambiguous images
            if item["margin"] < MIN_MARGIN:

                item["status"] = "rejected_margin"
                continue

            # Check duplicates
            duplicate = False

            for accepted_index in accepted:

                similarity = float(
                    similarity_matrix[
                        index,
                        accepted_index
                    ].item()
                )

                if similarity >= DUPLICATE_THRESHOLD:

                    duplicate = True
                    break

            if duplicate:

                item["status"] = "rejected_duplicate"
                continue

            # Accept image
            accepted.append(index)
            selected_indices.add(index)

            item["status"] = "accepted"

            if len(accepted) >= KEEP_IMAGES:
                break

        # Mark remaining images
        for index in indices:

            if "status" not in images[index]:

                images[index]["status"] = "rejected_rank"

        print(
            f"Accepted: {len(accepted)}/{len(indices)}"
        )

        if len(accepted) < MIN_IMAGES:

            raise RuntimeError(
                f"Only {len(accepted)} images accepted "
                f"for {class_name}. "
                "Review candidate images or adjust filtering."
            )

    return selected_indices


# ==========================================
# SAVE DATASET
# ==========================================

# Save accepted images into train, val, and test
def save_dataset(images, selected_indices, output_dir):

    print(
        "\n========== SAVING DATASET =========="
    )

    rng = random.Random(SEED)

    total_saved = 0

    for class_name in CLASSES:

        accepted = [
            images[index]
            for index in sorted(selected_indices)
            if images[index]["class"] == class_name
        ]

        rng.shuffle(accepted)

        count = len(accepted)

        # Calculate dataset splits
        val_count = max(
            1,
            round(count * 0.15)
        )

        test_count = max(
            1,
            round(count * 0.15)
        )

        train_count = (
            count - val_count - test_count
        )

        if train_count < 1:

            raise RuntimeError(
                f"Not enough training images for {class_name}"
            )

        splits = (
            ["train"] * train_count
            + ["val"] * val_count
            + ["test"] * test_count
        )

        for index, (item, split) in enumerate(
            zip(accepted, splits),
            start=1
        ):

            folder = output_dir / split / class_name

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

            filename = (
                f"{class_name}_{index:03d}.jpg"
            )

            output_path = folder / filename

            # Save image
            item["image"].save(
                output_path,
                "JPEG",
                quality=90
            )

            # Store saved path
            item["saved_path"] = str(
                output_path.relative_to(output_dir)
            )

            total_saved += 1

        print(
            f"{class_name}: "
            f"{train_count} train, "
            f"{val_count} val, "
            f"{test_count} test"
        )

    return total_saved


# ==========================================
# SAVE SIMILARITY REPORT
# ==========================================

# Save candidate scores to CSV
def save_report(images, output_dir):

    csv_path = output_dir / "sources.csv"

    fieldnames = [
        "class",
        "title",
        "source",
        "query",
        "within_similarity",
        "other_similarity",
        "best_other_class",
        "margin",
        "corn_similarity",
        "soil_similarity",
        "grass_similarity",
        "rock_similarity",
        "status",
        "saved_path",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for item in images:

            class_scores = item["class_scores"]

            writer.writerow({
                "class": item["class"],
                "title": item["title"],
                "source": item["url"],
                "query": item["query"],

                "within_similarity": round(
                    item["within_similarity"], 4
                ),

                "other_similarity": round(
                    item["other_similarity"], 4
                ),

                "best_other_class": item["best_other_class"],

                "margin": round(
                    item["margin"], 4
                ),

                "corn_similarity": round(
                    class_scores["corn"], 4
                ),

                "soil_similarity": round(
                    class_scores["soil"], 4
                ),

                "grass_similarity": round(
                    class_scores["grass"], 4
                ),

                "rock_similarity": round(
                    class_scores["rock"], 4
                ),

                "status": item.get(
                    "status", "rejected"
                ),

                "saved_path": item.get(
                    "saved_path", ""
                ),
            })

    print(
        f"\nSimilarity report saved: {csv_path}"
    )


# ==========================================
# REPLACE EXISTING DATASET
# ==========================================

# Replace dataset after successful processing
def replace_dataset(new_dir, dataset_dir):

    backup_dir = dataset_dir.with_name(
        dataset_dir.name + "_backup"
    )

    if backup_dir.exists():

        raise RuntimeError(
            f"Backup folder already exists: {backup_dir}"
        )

    had_old_dataset = dataset_dir.exists()

    # Back up old dataset
    if had_old_dataset:

        dataset_dir.rename(
            backup_dir
        )

    try:

        new_dir.rename(
            dataset_dir
        )

    except Exception:

        # Restore original dataset
        if had_old_dataset:

            backup_dir.rename(
                dataset_dir
            )

        raise

    # Delete backup after success
    if had_old_dataset:

        shutil.rmtree(
            backup_dir
        )


# ==========================================
# MAIN
# ==========================================

def main():

    print(
        "\n========== DINOv2 DATASET BUILDER =========="
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    session = requests.Session()

    session.headers.update(
        HEADERS
    )

    all_images = []
    seen_hashes = set()

    try:

        # Load cached candidates and download missing images
        for class_name in CLASSES:

            images = collect_images(
                session,
                class_name,
                seen_hashes
            )

            all_images.extend(images)

    except RateLimitError as error:

        # Stop without losing cached downloads
        print(
            "\n========== DOWNLOAD PAUSED =========="
        )

        print(error)

        print(
            "\nDownloaded images have been saved in:"
        )

        print(CACHE_DIR)

        print(
            "\nThe existing dataset was not replaced."
        )

        print(
            "Resume by rerunning the script after "
            "Wikimedia's requested cooldown."
        )

        return

    finally:

        session.close()

    # Verify each category has enough candidates
    for class_name in CLASSES:

        count = sum(
            item["class"] == class_name
            for item in all_images
        )

        if count < MIN_IMAGES:

            print(
                f"\nNot enough images for {class_name}: {count}"
            )

            print(
                "Existing dataset was not replaced."
            )

            return

    print(
        "\n========== DOWNLOAD COMPLETE =========="
    )

    print(
        f"Total candidate images: {len(all_images)}"
    )

    # Load pretrained DINOv2
    processor, model = load_dinov2()

    # Extract features
    features = extract_dataset_features(
        all_images,
        processor,
        model
    )

    # Calculate all-to-all similarity
    similarity_matrix = calculate_similarity_matrix(
        features
    )

    # Calculate class scores
    calculate_class_scores(
        all_images,
        similarity_matrix
    )

    # Select best images
    selected_indices = select_best_images(
        all_images,
        similarity_matrix
    )

    # Build dataset in temporary folder
    with tempfile.TemporaryDirectory(
        prefix="dataset_new_",
        dir=BASE_DIR
    ) as temp_dir:

        new_dir = Path(temp_dir) / "dataset"

        new_dir.mkdir()

        # Save selected images
        total_saved = save_dataset(
            all_images,
            selected_indices,
            new_dir
        )

        # Save CSV report
        save_report(
            all_images,
            new_dir
        )

        # Replace existing dataset
        replace_dataset(
            new_dir,
            DATASET_DIR
        )

    print(
        "\n========== COMPLETE =========="
    )

    print(
        f"Dataset saved: {DATASET_DIR}"
    )

    print(
        f"Cached images: {CACHE_DIR}"
    )

    print(
        f"Total candidates: {len(all_images)}"
    )

    print(
        f"Total accepted: {total_saved}"
    )

    print(
        f"Total rejected: "
        f"{len(all_images) - total_saved}"
    )

    print(
        "\nReview accepted images before training."
    )


if __name__ == "__main__":
    main()