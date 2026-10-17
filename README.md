# Leaffliction

Computer-vision pipeline that classifies the disease state of a leaf from a single photo.
Dataset: 7,221 labeled leaf photos across 8 classes (4 apple + 4 grape conditions).

## Dataset

The `images/` folder is intentionally not committed to keep the repository small.
Expected layout:

```text
images/
├── Apple_Black_rot/
├── Apple_healthy/
├── Apple_rust/
├── Apple_scab/
├── Grape_Black_rot/
├── Grape_Esca/
├── Grape_healthy/
└── Grape_spot/
```

Each subdirectory is one class and contains that class's JPEG photos. All scripts accept the `images/` root as input and discover classes from subdirectory names.

## What was done

### 1. Dataset analysis: `Distribution.py`
Walks a dataset directory and plots the per-class distribution (pie + bar chart) so class imbalance is visible before any training.

```bash
python3 Distribution.py images/
```

### 2. Data augmentation: `Augmentation.py`
Applies six augmentation techniques (Flip, Rotate, Skew, Shear, Crop, Distortion) using PIL. Two modes:
- Single image: displays the six variants.
- Directory: balances every class by generating augmented copies until all classes match the biggest.

```bash
python3 Augmentation.py "images/Apple_scab/image (1).JPG"
python3 Augmentation.py images/ --dst augmented_directory
```

### 3. Feature-extraction transformations: `Transformation.py`
Seven PlantCV-based transformations: Original, Gaussian blur, Mask, ROI, Analyze object (shape features), Pseudolandmarks, and a 9-channel color histogram.

```bash
# single image: display all 7 panels
python3 Transformation.py "images/Apple_scab/image (1).JPG"

# batch: save selected transformations to disk
python3 Transformation.py -src images/Apple_healthy -dst dst_dir -mask -blur
```

### 4. Classifier training: `train.py`
Transfer-learning pipeline on top of MobileNetV2:
- **Leakage-safe split:** originals are split 80/20 *before* augmentation; the validation set contains only unmodified originals.
- **Balanced training set:** augments only the training side until all classes are equal in size.
- **Transfer learning:** MobileNetV2 (ImageNet, frozen) + GlobalAveragePooling + Dropout + Dense(8, softmax).
- **Callbacks:** `EarlyStopping` + `ReduceLROnPlateau`.
- **Optional fine-tuning:** unfreezes the top 30 MobileNetV2 layers at LR 1e-5 if the head stage falls short of 95%.
- **Reporting:** per-class accuracy printed at the end.
- **Packaging:** zips the model + augmented images into `learnings.zip`, writes SHA-1 to `signature.txt`.

Achieved **98.75% validation accuracy** (per-class range 96.7% to 100%).

```bash
python3 train.py images/
```

Outputs: `learnings/model.keras`, `learnings/class_names.json`, `learnings.zip`, `signature.txt`.

### 5. Prediction: `predict.py`
Loads the trained model (auto-extracts `learnings.zip` if the folder is missing), predicts the class of one image, and displays the original + a mask-isolated view side-by-side with the predicted class and confidence.

```bash
python3 predict.py "images/Apple_scab/image (1).JPG"
```

## How to test

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Explore the dataset
python3 Distribution.py images/

# 3. Train (writes learnings/, learnings.zip, signature.txt)
python3 train.py images/

# 4. Verify the signature matches the zip
shasum learnings.zip   # must equal the hash in signature.txt

# 5. Classify an image
python3 predict.py "images/Apple_scab/image (1).JPG"
```

A trained `learnings.zip` is shipped separately (not committed). `signature.txt` in this repo is its SHA-1 fingerprint; any mismatch means the zip was altered.
