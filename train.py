"""Iteration 1: minimal end-to-end classifier.

Loads images directly from a directory (one subfolder per class), splits
train/val, builds a MobileNetV2 transfer-learning model, trains for a
few epochs, and saves the model + class names into ./learnings/.
"""
import argparse
import json
import os
import sys

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3
SEED = 13
LEARNINGS_DIR = "learnings"


def load_datasets(data_dir):
    """Return (train_ds, val_ds, class_names) built from one directory."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    class_names = train_ds.class_names

    # Apply MobileNetV2's preprocessing (scales pixel values to [-1, 1]).
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

    # Prefetch for throughput.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds, class_names


def build_model(num_classes):
    """MobileNetV2 base (frozen) + small classifier head."""
    base = MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_artifacts(model, class_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.keras"))
    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Train a leaf-disease classifier (Iteration 1)."
    )
    parser.add_argument(
        "directory",
        help="Dataset root (one subdirectory per class).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Loading datasets from {args.directory!r} ...")
    train_ds, val_ds, class_names = load_datasets(args.directory)
    print(f"Classes ({len(class_names)}): {class_names}")

    print("Building MobileNetV2 transfer-learning model ...")
    model = build_model(len(class_names))
    model.summary()

    print(f"Training for {EPOCHS} epochs ...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    final_val_acc = history.history["val_accuracy"][-1]
    print(f"\nFinal validation accuracy: {final_val_acc:.4f}")

    print(f"Saving artifacts to ./{LEARNINGS_DIR}/ ...")
    save_artifacts(model, class_names, LEARNINGS_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
