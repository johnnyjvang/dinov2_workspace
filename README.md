# DINOv2 Workspace

Python workspace for experimenting with image similarity and terrain classification using pretrained DINOv2 Small visual features.

Each image is represented by a 384-dimensional feature vector extracted using `facebook/dinov2-small`. These features can be used to compare image similarity or train a lightweight classifier to recognize different terrain categories.

The current terrain classification experiment focuses on four classes: **corn, soil, grass, and rock**.

---

## Current Features

- Compare two images using DINOv2 visual features.
- Extract 384-dimensional feature vectors using `facebook/dinov2-small`.
- Train a linear terrain classifier using manually collected images.
- Evaluate k-Nearest Neighbors (k-NN) using cosine distance.
- Automatically split images into training, validation, and testing sets.
- Classify multiple images using a saved classifier.
- Evaluate accuracy, class scores, prediction margins, and confusion matrices.
- Save trained classifier weights for future predictions.
- Run on an NVIDIA GPU when available, otherwise CPU.

## Python Scripts

| Script | Description | Status |
| --- | --- | --- |
| `compare_images.py` | Compare visual similarity between two images using DINOv2. | Working |
| `train_classifier.py` | Train and evaluate a terrain classifier using manually labeled images. | Working |
| `predict_image.py` | Classify new images using the saved terrain classifier. | Working |
| `download_dataset.py` | Automatically collect and filter images from Wikimedia Commons. | Work in Progress |

---

## 1. Image Similarity

`compare_images.py` compares two images using pretrained DINOv2 features without training a new model.

### Comparison Methods

| Metric | Interpretation | Reference Images Needed? |
| --- | --- | --- |
| Cosine Similarity | Higher = more similar | No |
| Euclidean Distance (L2) | Lower = more similar | No |
| Manhattan Distance (L1) | Lower = more similar | No |
| Pearson Correlation | Higher = more similar | No |
| Mahalanobis Distance | Lower = more similar | Yes |

### How to Run

Compare two images using four metrics:

```bash
python3 compare_images.py images/corn_field_1.jpg images/corn_field_2.jpg
```

Include Mahalanobis distance:

```bash
python3 compare_images.py images/corn_field_1.jpg images/corn_field_2.jpg --reference-dir images/references
```

`--reference-dir` is only used for Mahalanobis distance. Multiple representative reference images are recommended for more reliable covariance estimates.

**Note:** Image similarity measures how closely visual feature representations match. It does not directly identify or classify objects.

---

## 2. Terrain Classification

`train_classifier.py` trains a lightweight classifier using DINOv2 features extracted from manually collected terrain images.

The current experiment uses four categories:

- Corn
- Soil
- Grass
- Rock

DINOv2 is used as a frozen feature extractor. Only the linear classification layer is trained.

### Dataset Structure

Images were manually collected and organized into four folders, with **10 images per class (40 total)**.

```text
dataset/
├── corn/
│   ├── corn_01.jpg
│   ├── corn_02.jpg
│   └── ... (10 images)
├── soil/
│   ├── soil_01.jpg
│   ├── soil_02.jpg
│   └── ... (10 images)
├── grass/
│   ├── grass_01.jpg
│   ├── grass_02.jpg
│   └── ... (10 images)
└── rock/
    ├── rock_01.jpg
    ├── rock_02.jpg
    └── ... (10 images)
```

The script automatically splits each class into training, validation, and testing images.

| Dataset | Images per Class | Total Images |
| --- | ---: | ---: |
| Training (80%) | 8 | 32 |
| Validation (10%) | 1 | 4 |
| Testing (10%) | 1 | 4 |
| **Total** | **10** | **40** |

Images remain in their original folders. The splits are performed automatically using a fixed random seed.

### Training Methods

Two approaches are evaluated:

**1. k-Nearest Neighbors (k-NN)**

- Uses cosine distance between DINOv2 feature vectors.
- Predicts classes based on similar labeled images.
- Does not require neural network training.

**2. Linear Classifier**

- Uses frozen DINOv2 feature vectors.
- Trains a linear layer with 384 inputs and 4 outputs.
- Uses cross-entropy loss and the Adam optimizer.
- Saves the model with the lowest validation loss.

### Training Configuration

| Parameter | Value |
| --- | --- |
| Pretrained Model | `facebook/dinov2-small` |
| Feature Dimensions | 384 |
| Classes | 4 |
| Training Images | 32 |
| Validation Images | 4 |
| Test Images | 4 |
| Learning Rate | 0.001 |
| Training Epochs | 200 (experiment) |
| Optimizer | Adam |
| Loss Function | Cross-Entropy |
| Feature Normalization | L2 |
| Saved Classifier | `models/terrain_classifier.pth` |

---

## 3. Setup (Ubuntu)

Tested using Ubuntu 24.04 and a Python virtual environment.

### Create Virtual Environment

Clone the repository:

```bash
git clone https://github.com/johnnyjvang/dinov2_workspace.git
cd dinov2_workspace
```

Install virtual environment support:

```bash
sudo apt update
sudo apt install python3-venv python3-pip -y
```

Create and activate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision transformers pillow numpy scikit-learn
```

The pretrained DINOv2 model downloads automatically on first use.

For NVIDIA GPU acceleration, install the appropriate PyTorch build for your system.

### Activate Environment for Future Sessions

```bash
cd ~/dinov2_workspace
source .venv/bin/activate
```

To exit:

```bash
deactivate
```

---

## 4. Train Terrain Classifier

Before training, manually place images inside the four `dataset/` class folders.

### Train Using Default Configuration

```bash
python3 train_classifier.py
```

### Train Using 200 Epochs

```bash
python3 train_classifier.py --epochs 200
```

The script will:

1. Validate the dataset and image files.
2. Automatically split images into training, validation, and testing groups.
3. Extract DINOv2 features from each image.
4. Evaluate k-NN classification.
5. Train a linear classifier using frozen DINOv2 features.
6. Evaluate accuracy, precision, recall, F1, and confusion matrix.
7. Save the trained classifier.

The resulting file is:

```text
models/terrain_classifier.pth
```

This file contains the trained linear classifier weights, biases, class labels, feature dimensions, and DINOv2 model information.

It does not contain the pretrained DINOv2 weights or the original training images.

---

## 5. Predict Terrain Classes

`predict_image.py` uses the trained classifier to predict terrain categories for new images.

The script supports testing a single image or automatically processing multiple images.

### Test Image Structure

```text
test_images/
├── corn_test.jpg
├── grass_test.jpg
├── rock_test.jpg
└── soil_test.jpg
```

Additional images can be added without modifying the script.

### Run All Test Images

```bash
python3 predict_image.py
```

### Run a Single Image

```bash
python3 predict_image.py --image test_images/corn_test.jpg
```

### Prediction Output

The script displays:

- Expected class based on the filename.
- Predicted class.
- Highest softmax class score.
- Difference between the highest and second-highest scores.
- Correct/incorrect result.
- Full class scores.
- Overall and per-class accuracy.
- Confusion matrix.

Results are also saved to:

```text
results/predictions.csv
```

**Note:** Filename labels are used only for evaluation. The classifier makes predictions using image features, not filenames.

Softmax scores are not calibrated probabilities and should not be interpreted as guaranteed confidence levels.

---

## 6. Experimental Results

**Experiment Date:** November 11, 2026

**Model:** DINOv2 Small + Linear Classifier

**Dataset:** 40 manually collected images (10 per class)

**Training:** 200 epochs

### Initial vs. Updated Results

The initial classifier was trained using 30 epochs. A second experiment increased training to 200 epochs.

Both experiments correctly classified all four test images, but the second produced larger prediction scores and margins.

| Class | Initial Score (30 Epochs) | Updated Score (200 Epochs) |
| --- | ---: | ---: |
| Corn | 30.0% | 51.1% |
| Grass | 32.7% | 67.8% |
| Rock | 28.3% | 43.6% |
| Soil | 27.7% | 40.4% |

### Updated Prediction Results

| Image | Expected | Predicted | Top Score | Margin | Result |
| --- | --- | --- | ---: | ---: | --- |
| corn_test.jpg | Corn | Corn | 51.1% | 32.3% | PASS |
| grass_test.jpg | Grass | Grass | 67.8% | 56.4% | PASS |
| rock_test.jpg | Rock | Rock | 43.6% | 22.4% | PASS |
| soil_test.jpg | Soil | Soil | 40.4% | 13.9% | PASS |

**Overall Accuracy: 4/4 (100%)**

### Full Class Scores

| Test Image | Corn | Soil | Grass | Rock |
| --- | ---: | ---: | ---: | ---: |
| Corn | **51.1%** | 15.9% | 18.8% | 14.2% |
| Grass | 11.3% | 11.4% | **67.8%** | 9.5% |
| Rock | 19.4% | 21.2% | 15.8% | **43.6%** |
| Soil | 15.3% | **40.4%** | 17.8% | 26.5% |

### Observations

- All four test images were classified correctly.
- Increasing training epochs from 30 to 200 resulted in stronger separation between class scores.
- Grass produced the highest top-class score at 67.8%.
- Soil and rock showed relatively smaller prediction margins, suggesting more visual overlap.
- The classifier demonstrates that pretrained DINOv2 features can be used for lightweight terrain classification.

**Limitations:** Only four external test images were evaluated. Although accuracy was 100%, this is insufficient to determine reliable generalization. Additional testing with unseen images and varied environmental conditions is needed.

Higher softmax scores also do not necessarily indicate improved accuracy or calibrated confidence.

---

## 7. Automated Dataset Download (Work in Progress)

`download_dataset.py` was developed to automate collecting images from Wikimedia Commons for terrain classification.

### Intended Workflow

1. Search Wikimedia Commons for terrain-related images.
2. Download candidate images for corn, soil, grass, and rock.
3. Extract DINOv2 features.
4. Compare images using cosine similarity.
5. Identify near-duplicate or visually similar images.
6. Review and approve correctly labeled images.
7. Generate a dataset for classifier training.

### Current Issues

Initial testing showed that search results sometimes included unrelated or incorrectly labeled images.

For example, searching for corn could return images unrelated to corn plants.

This occurs because keyword-based searches do not reliably verify image contents, while DINOv2 visual similarity alone cannot confirm semantic labels.

Download requests also encountered rate-limiting issues.

**Current Status:** Work in progress. The terrain classifier currently uses manually collected and verified images instead.

Potential future improvements include better search queries, automatic semantic verification using a vision-language model, and improved download rate limiting.

---

## 8. Repository Structure

```text
dinov2_workspace/
├── README.md
├── .gitignore
├── compare_images.py
├── download_dataset.py       # Work in progress
├── train_classifier.py
├── predict_image.py
├── .venv/                    # Python virtual environment
├── images/                   # Image similarity experiments
│   └── references/
├── dataset/                  # Manually labeled training images
│   ├── corn/
│   ├── soil/
│   ├── grass/
│   └── rock/
├── test_images/              # Images for prediction testing
│   ├── corn_test.jpg
│   ├── grass_test.jpg
│   ├── rock_test.jpg
│   └── soil_test.jpg
├── models/
│   └── terrain_classifier.pth
└── results/
    └── predictions.csv
```

Generated files, local datasets, and virtual environments can be excluded from Git using `.gitignore`.

---

## 9. Future Improvements

- Expand the manually collected dataset with more diverse images.
- Evaluate classification performance on a larger independent test set.
- Compare linear classification against k-NN.
- Improve automated dataset collection and label verification.
- Add live camera classification.
- Investigate object detection and semantic segmentation for identifying multiple terrain types within an image.
- Integrate terrain perception with ROS 2 for agricultural mobile robotics.

---

## Notes

- DINOv2 is used as a pretrained visual feature extractor.
- The linear terrain classifier is trained separately; DINOv2 itself is not fine-tuned.
- Image-level classification predicts one category per image.
- Softmax scores are not calibrated confidence estimates.
- A larger and more diverse dataset is required for reliable real-world evaluation.
- The current project is an experimental foundation for future robotic perception applications.