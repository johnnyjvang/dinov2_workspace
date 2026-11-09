# DINOv2 Image Similarity

Python script for comparing two images using pretrained DINOv2 Small visual features. Each image is represented by a 384-dimensional feature vector, and the script reports similarity and distance metrics without training a model.

---

## Current Features

- Compare any two JPG, PNG, or WebP images.
- Extract image-level features using `facebook/dinov2-small`.
- Run on an NVIDIA GPU when available, otherwise CPU.
- Optionally estimate Mahalanobis distance using a folder of reference images.

## Comparison Methods

| Metric | Interpretation | Reference images needed? |
| --- | --- | --- |
| Cosine similarity | Higher = more similar | No |
| Euclidean distance (L2) | Lower = more similar | No |
| Manhattan distance (L1) | Lower = more similar | No |
| Pearson correlation | Higher = more similar | No |
| Mahalanobis distance | Lower = more similar | Yes |

---

## Setup (Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision transformers pillow
```

The pretrained DINOv2 model downloads automatically on first use.

## How to Run

Compare two images (four metrics):

```bash
python3 compare_images.py images/corn_field_1.jpg images/corn_field_2.jpg
```

Include Mahalanobis distance (all five metrics):

```bash
python3 compare_images.py images/corn_field_1.jpg images/corn_field_2.jpg --reference-dir images/references
```

`--reference-dir` is only used for Mahalanobis distance. Use multiple representative reference photos; larger datasets give more reliable covariance estimates.

---

## Repository Structure

```text
dinov2_similarity/
├── compare_images.py
├── README.md
├── .gitignore
└── images/              # Local images; excluded from Git
```

## Notes

- The script compares **visual feature representations**, not image identity or semantic labels.
- Similarity metrics use different scales and should not be compared numerically with one another.
- No fine-tuning or custom training is required.