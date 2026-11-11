#!/usr/bin/env python3
"""
download_dataset.py

Download a small image dataset from Wikimedia Commons and review it using DINOv2.

Classes: corn, soil, grass, rock.

Workflow:
  1. Download a small number of targeted Wikimedia Commons photos.
  2. Save candidates and metadata in candidate_images/<class>/.
  3. Compute DINOv2 cosine similarities and flag near duplicates.
  4. Write review.csv and review_sheet.jpg (NO automatic label verification).
  5. After reviewing the images, mark approved=yes in review.csv.
  6. Run with --build to create dataset/train, dataset/val, dataset/test.

Commands:
  python3 download_dataset.py                    # Download + create review file
  python3 download_dataset.py --cache-only       # Recreate review file, no network
  python3 download_dataset.py --build            # Build from manually approved rows
  python3 download_dataset.py --per-class 6      # Smaller download

Install: pip install requests pillow torch transformers

WARNING: DINOv2 similarity is NOT a semantic correctness check.
"""

import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoImageProcessor, AutoModel

# ==========================================
# CONFIGURATION
# ==========================================

# Paths and DINOv2 model
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "candidate_images"
DATASET_DIR = BASE_DIR / "dataset"
REVIEW_CSV = BASE_DIR / "review.csv"
REVIEW_SHEET = BASE_DIR / "review_sheet.jpg"
MODEL_NAME = "facebook/dinov2-small"
CLASSES = ("corn", "soil", "grass", "rock")
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "DINOv2TerrainDataset/1.0 (educational research; Wikimedia Commons)"

# High-precision queries. Avoid generic intitle:corn (corn snakes, popcorn, etc.).
# Focus on agricultural photos, not generic titles like "corn" or "rock".
SEARCH_QUERIES = {
    "corn": ["intitle:\"Zea mays\"", "intitle:\"maize plant\"", "intitle:\"corn stalk\"", "intitle:\"maize field\""],
    "soil": ["intitle:\"agricultural soil\"", "intitle:\"soil surface\"", "intitle:\"bare soil\"", "intitle:\"topsoil\""],
    "grass": ["intitle:\"grass field\"", "intitle:\"lawn grass\"", "intitle:\"grass meadow\"", "intitle:\"grassland\""],
    "rock": ["intitle:\"gravel ground\"", "intitle:\"rocky ground\"", "intitle:\"stone gravel\"", "intitle:\"pebbles\""],
}

# A filename must include an appropriate term; this is only a weak label heuristic.
# Required words found in a Wikimedia filename/title.
REQUIRED = {
    "corn": ("zea mays", "maize", "corn plant", "corn stalk", "corn field", "corn crop"),
    "soil": ("soil", "topsoil", "bare earth", "dirt ground"),
    "grass": ("grass", "grassland", "lawn", "meadow"),
    "rock": ("rock", "rocks", "rocky", "gravel", "pebble", "pebbles", "stone", "stones"),
}

# Reject diagrams, foods, animals and other known false matches.
# Common false positives to exclude before downloading.
BAD_TERMS = (
    "diagram", "graph", "logo", "icon", "illustration", "drawing", "painting",
    "poster", "screenshot", "map", "micrograph", "microscope", "histology",
    "seed packet", "clipart", "vector", "cartoon", "render", "model 3d",
    "corn snake", "cornsnake", "popcorn", "cornbread", "cornmeal", "corn starch",
    "cornstarch", "corn flakes", "candy corn", "sweetcorn soup",
    "cornwall", "cornish", "unicorn", "capricorn", "cornflower",
    "rock band", "rock music", "rock concert", "rock climbing", "rock guitar",
    "grasshopper", "grass snake", "lemongrass", "sea grass", "seagrass",
    "soil map", "soil chart", "soil profile diagram",
)

# Wikimedia request settings: deliberately slow and sequential.
REQUEST_DELAY = 4.0
TIMEOUT = 20
MAX_RETRIES = 2
MAX_COOLDOWN = 45
IMAGE_SIZE = 384
MIN_SIDE = 160
DUPLICATE_THRESHOLD = 0.97
SEED = 42



# ==========================================
# HTTP REQUESTS AND RATE LIMITS
# ==========================================

class RateLimitError(RuntimeError):
    """Stop this run when Wikimedia requests a cooldown."""


def request(session, url, params=None):
    """One sequential HTTP request with rate-limit-aware retries."""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, params=params, timeout=TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as error:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Request failed: {error}") from error
            time.sleep(5 * (attempt + 1))
            continue
        # Respect Wikimedia rate limits. Stop the run rather than hammering the API.
        if response.status_code in (429, 503):
            value = response.headers.get("Retry-After", "")
            try:
                delay = float(value)
            except ValueError:
                try:
                    delay = max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
                except (ValueError, TypeError, OverflowError):
                    delay = 10 * (attempt + 1)
            response.close()
            if delay > MAX_COOLDOWN or attempt == MAX_RETRIES - 1:
                raise RateLimitError(f"HTTP {response.status_code}: pause downloads; retry later.")
            print(f"  HTTP {response.status_code}; cooling down {delay:.0f}s")
            time.sleep(delay)
            continue
        try:
            response.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return response
        except requests.HTTPError:
            response.close()
            raise
    raise RuntimeError("Request retry limit reached")



# ==========================================
# WIKIMEDIA TITLE FILTERING
# ==========================================

def normalize_title(title):
    title = re.sub(r"^File:", "", title, flags=re.I)
    title = re.sub(r"\.[a-z0-9]{2,5}$", "", title, flags=re.I)
    title = re.sub(r"[_\-]+", " ", title.lower())
    return re.sub(r"\s+", " ", title).strip()


def has_phrase(text, phrase):
    return re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", text) is not None


def valid_title(title, class_name):
    text = normalize_title(title)
    if any(has_phrase(text, bad) for bad in BAD_TERMS):
        return False
    return any(has_phrase(text, term) for term in REQUIRED[class_name])



# ==========================================
# SEARCH AND DOWNLOAD
# ==========================================

def search(session, query, limit):
    """Make one small Wikimedia API request per query (no pagination)."""
    response = request(session, API_URL, {
        "action": "query", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"filetype:bitmap {query}", "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": IMAGE_SIZE,
        "format": "json", "formatversion": 2,
    })
    try:
        data = response.json()
    finally:
        response.close()
    if "error" in data:
        raise RuntimeError(f"Wikimedia API error: {data['error']}")
    return data.get("query", {}).get("pages", [])


def fetch_image(session, url):
    response = request(session, url)
    try:
        if len(response.content) > 8 * 1024 * 1024:
            raise ValueError("Image download exceeds 8 MB")
        with Image.open(io.BytesIO(response.content)) as source:
            if min(source.size) < MIN_SIDE:
                raise ValueError("Image too small")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            return image.copy()
    finally:
        response.close()



# ==========================================
# CACHE MANAGEMENT
# ==========================================

def load_cache(class_name):
    """Only use cache entries previously saved by this script's metadata format."""
    folder = CACHE_DIR / class_name
    entries = []
    for path in sorted(folder.glob("*.json")):
        image_path = path.with_suffix(".jpg")
        if not image_path.exists():
            continue
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            if info.get("class") != class_name:
                continue
            # Recheck title: older caches often contain incorrect labels.
            if not valid_title(info.get("title", ""), class_name):
                continue
            with Image.open(image_path) as original:
                image = original.convert("RGB").copy()
            entries.append({**info, "image": image, "image_path": image_path})
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(f"  Skipping broken cache {path.name}: {error}")
    return entries


def collect(session, class_name, target, search_limit, cache_only):
    entries = load_cache(class_name)
    seen_ids = {str(item["page_id"]) for item in entries}
    print(f"{class_name}: {len(entries)} valid cached candidate(s)")
    if cache_only or len(entries) >= target:
        return entries[:target]

    for query in SEARCH_QUERIES[class_name]:
        if len(entries) >= target:
            break
        print(f"  Searching {query}")
        try:
            pages = search(session, query, search_limit)
        except RateLimitError:
            raise
        except (requests.RequestException, RuntimeError) as error:
            print(f"  Query failed: {error}")
            continue
        for page in pages:
            if len(entries) >= target:
                break
            page_id = page.get("pageid")
            title = page.get("title", "")
            if page_id is None or str(page_id) in seen_ids or not valid_title(title, class_name):
                continue
            seen_ids.add(str(page_id))
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mime") not in {"image/jpeg", "image/png", "image/webp"}:
                continue
            url = info.get("thumburl") or info.get("url")
            if not url:
                continue
            try:
                image = fetch_image(session, url)
            except RateLimitError:
                raise
            except (requests.RequestException, RuntimeError, OSError, ValueError) as error:
                print(f"  Skipping {title}: {error}")
                continue
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()
            folder = CACHE_DIR / class_name
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{page_id}.jpg"
            image.save(path, "JPEG", quality=90)
            metadata = {"class": class_name, "page_id": str(page_id), "title": title,
                        "url": url, "query": query, "hash": image_hash}
            (folder / f"{page_id}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            entries.append({**metadata, "image": image, "image_path": path})
            print(f"  Saved {len(entries)}/{target}: {title[:72]}")
    return entries[:target]



# ==========================================
# DINOv2 FEATURE EXTRACTION
# ==========================================

def extract_features(entries):
    """DINOv2 embeddings: measure visual similarity, NOT whether the label is correct."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Extracting DINOv2 features on {device}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    features = []
    for number, item in enumerate(entries, 1):
        inputs = processor(images=item["image"], return_tensors="pt").to(device)
        with torch.inference_mode():
            vector = model(**inputs).last_hidden_state[:, 0, :]
        features.append(F.normalize(vector.float(), dim=1).squeeze(0).cpu())
        print(f"  Features: {number}/{len(entries)}", end="\r", flush=True)
    print()
    return torch.stack(features)



# ==========================================
# SIMILARITY ANALYSIS AND REVIEW CSV
# ==========================================

def review_candidates(entries, similarity):
    """Flag duplicates; retain existing approval decisions by class + page ID."""
    old_decisions = {}
    if REVIEW_CSV.exists():
        with REVIEW_CSV.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                old_decisions[(row.get("class"), row.get("page_id"))] = row.get("approved", "")

    rows = []
    # Sort by category so the contact sheet and CSV match.
    for index, item in enumerate(entries):
        same_class = [j for j, other in enumerate(entries) if j != index and other["class"] == item["class"]]
        other_class = [j for j, other in enumerate(entries) if other["class"] != item["class"]]
        within = max((float(similarity[index, j]) for j in same_class), default=0.0)
        between = max((float(similarity[index, j]) for j in other_class), default=0.0)
        # Flag duplicates only against earlier entries of this class.
        duplicate = any(float(similarity[index, j]) >= DUPLICATE_THRESHOLD for j in same_class if j < index)
        key = (item["class"], str(item["page_id"]))
        approved = old_decisions.get(key, "")
        # Never override manual decisions, but duplicate rows are not eligible for build.
        rows.append({
            "class": item["class"], "page_id": str(item["page_id"]),
            "title": item["title"], "query": item["query"],
            "image_path": str(item["image_path"].relative_to(BASE_DIR)),
            "source": item["url"], "within_similarity": f"{within:.4f}",
            "other_similarity": f"{between:.4f}", "margin": f"{within-between:.4f}",
            "near_duplicate": "yes" if duplicate else "no", "approved": approved,
        })
    if not rows:
        print("No candidates to write to review.csv.")
        return rows

    with REVIEW_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows



# ==========================================
# IMAGE REVIEW CONTACT SHEET
# ==========================================

def save_contact_sheet(entries, rows):
    """Image sheet labeled by class and page_id for easy manual review."""
    width, height, columns = 230, 210, 4
    count_rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, count_rows * height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (item, row) in enumerate(zip(entries, rows)):
        x = (index % columns) * width
        y = (index // columns) * height
        thumbnail = item["image"].copy()
        thumbnail.thumbnail((210, 165), Image.Resampling.LANCZOS)
        sheet.paste(thumbnail, (x + (width - thumbnail.width) // 2, y + 5))
        draw.text((x + 8, y + 175), f"{row['class']} | ID {row['page_id']}", fill="black")
        draw.text((x + 8, y + 190), "DUPLICATE" if row["near_duplicate"] == "yes" else "", fill="black")
    sheet.save(REVIEW_SHEET, "JPEG", quality=90)



# ==========================================
# BUILD TRAIN / VAL / TEST DATASET
# ==========================================

def build_dataset(rows, keep):
    """Build ONLY from manually approved, nonduplicate images."""
    approved_by_class = {}
    for class_name in CLASSES:
        approved_by_class[class_name] = [
            row for row in rows if row["class"] == class_name
            and row["approved"].strip().lower() in {"yes", "y", "true", "1"}
            and row["near_duplicate"] == "no"
        ][:keep]
        if len(approved_by_class[class_name]) < 3:
            print(f"Cannot build: {class_name} has only {len(approved_by_class[class_name])} approved, nonduplicate images (need 3).")
            print("Open review_sheet.jpg and set approved=yes in review.csv for correct images.")
            return False

    rng = random.Random(SEED)
    with tempfile.TemporaryDirectory(prefix="dataset_new_", dir=BASE_DIR) as temp:
        new_dataset = Path(temp) / "dataset"
        new_dataset.mkdir()
        manifest = []
        for class_name, candidates in approved_by_class.items():
            items = list(candidates)
            rng.shuffle(items)
            # At least one image per validation and test split.
            train_count = len(items) - 2
            for number, row in enumerate(items):
                split = "train" if number < train_count else ("val" if number == train_count else "test")
                destination = new_dataset / split / class_name / f"{class_name}_{number+1:03d}.jpg"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(BASE_DIR / row["image_path"], destination)
                manifest.append({"split": split, "class": class_name, "file": str(destination.relative_to(new_dataset)),
                                 "source": row["source"], "title": row["title"]})
            print(f"{class_name}: {train_count} train, 1 val, 1 test")
        with (new_dataset / "sources.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        # Safely replace only after all copies succeed; retain previous dataset as backup.
        backup = BASE_DIR / "dataset_backup"
        if backup.exists():
            print(f"Cannot replace dataset while {backup} exists; move/remove that backup first.")
            return False
        if DATASET_DIR.exists():
            DATASET_DIR.rename(backup)
        try:
            new_dataset.rename(DATASET_DIR)
        except Exception:
            if backup.exists() and not DATASET_DIR.exists():
                backup.rename(DATASET_DIR)
            raise
        print(f"Dataset created: {DATASET_DIR}")
        if backup.exists():
            print(f"Previous dataset kept at: {backup}")
    return True



# ==========================================
# MAIN PROGRAM
# ==========================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=8, help="Candidate images per class (default: 8)")
    parser.add_argument("--keep", type=int, default=6, help="Maximum approved images per class (default: 6)")
    parser.add_argument("--search-limit", type=int, default=10, help="Wikimedia search results per query (default: 10)")
    parser.add_argument("--cache-only", action="store_true", help="No HTTP requests; process existing candidate cache")
    parser.add_argument("--build", action="store_true", help="Build dataset using review.csv approvals (no download)")
    args = parser.parse_args()
    if args.per_class < 3 or args.keep < 3 or args.search_limit < 1:
        parser.error("--per-class and --keep must be >= 3; --search-limit must be >= 1")

    # --build never contacts Wikimedia or loads the DINOv2 model.
    if args.build:
        if not REVIEW_CSV.exists():
            parser.error("review.csv not found. Run the downloader and review candidates first.")
        with REVIEW_CSV.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        build_dataset(rows, args.keep)
        return

    # Search and download candidates unless --cache-only is enabled.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        for class_name in CLASSES:
            try:
                entries.extend(collect(session, class_name, args.per_class, args.search_limit, args.cache_only))
            except RateLimitError as error:
                print(f"\nPaused: {error}. Saved images remain in {CACHE_DIR}.")
                print("Rerun later; cached downloads will be reused.")
                return
    if not entries:
        print("No usable images found. No dataset was modified.")
        return
    if len(entries) < 2:
        print("Need at least two images to compute similarities.")
        return
    # Build review artifacts from DINOv2 visual embeddings.
    features = extract_features(entries)
    similarity = features @ features.T
    rows = review_candidates(entries, similarity)
    save_contact_sheet(entries, rows)
    print(f"\nReview images: {REVIEW_SHEET}")
    print(f"Mark correct images as approved=yes in {REVIEW_CSV}")
    print("Then run: python3 download_dataset.py --build")
    print("No training dataset was changed before manual approval.")


if __name__ == "__main__":
    main()
