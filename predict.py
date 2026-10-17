import argparse
import json
import os
import sys
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from Transformation import build_mask, load_image, t_mask

IMAGE_SIZE = (224, 224)
LEARNINGS_DIR = "learnings"
ZIP_PATH = "learnings.zip"
MODEL_FILE = "model.keras"
CLASS_NAMES_FILE = "class_names.json"


def ensure_learnings_dir():
    """Unzip learnings.zip if the learnings/ folder is missing."""
    if os.path.isdir(LEARNINGS_DIR):
        return
    if not os.path.isfile(ZIP_PATH):
        print(
            f"Error: neither '{LEARNINGS_DIR}/' nor '{ZIP_PATH}' found. "
            "Run train.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Unzipping {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(".")


def load_learnings():
    model_path = os.path.join(LEARNINGS_DIR, MODEL_FILE)
    names_path = os.path.join(LEARNINGS_DIR, CLASS_NAMES_FILE)
    if not os.path.isfile(model_path) or not os.path.isfile(names_path):
        print(f"Error: missing {model_path} or {names_path}", file=sys.stderr)
        sys.exit(1)
    model = tf.keras.models.load_model(model_path)
    with open(names_path) as f:
        class_names = json.load(f)
    return model, class_names


def preprocess_for_model(rgb):
    """Resize to the model's input size and apply MobileNetV2 preprocessing."""
    resized = tf.image.resize(rgb, IMAGE_SIZE).numpy()
    batch = np.expand_dims(resized, axis=0)
    return preprocess_input(batch)


def display(rgb, transformed, class_name, confidence):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 5))
    left.imshow(rgb)
    left.set_title("Original")
    left.axis("off")
    right.imshow(transformed)
    right.set_title("Transformed (mask)")
    right.axis("off")
    fig.suptitle(
        f"Class predicted: {class_name}  ({confidence * 100:.1f}%)",
        fontsize=14,
    )
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Classify a leaf image using the trained model."
    )
    parser.add_argument("image", help="Path to a leaf image.")
    args = parser.parse_args()

    try:
        rgb = load_image(args.image)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    ensure_learnings_dir()
    model, class_names = load_learnings()

    probs = model.predict(preprocess_for_model(rgb), verbose=0)[0]
    idx = int(np.argmax(probs))
    class_name = class_names[idx]
    confidence = float(probs[idx])

    print(f"Class predicted: {class_name}  ({confidence * 100:.2f}%)")

    mask = build_mask(rgb)
    transformed = t_mask(rgb, mask)
    display(rgb, transformed, class_name, confidence)


if __name__ == "__main__":
    main()
