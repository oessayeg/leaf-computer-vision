"""Iteration 2: leakage-safe augmentation pipeline.

Splits the raw dataset into train/val BEFORE augmenting, augments only
the training side (reusing Augmentation.py's techniques to balance classes),
and copies validation originals as-is. Then trains a MobileNetV2 transfer-
learning model and saves it under ./learnings/.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import zipfile

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from Augmentation import TECHNIQUES, load_image, save_image
from Distribution import is_image

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
FINETUNE_EPOCHS = 5
FINETUNE_UNFREEZE_LAYERS = 30
SEED = 13
VAL_FRACTION = 0.2
AUG_ROOT = "augmented_directory"
LEARNINGS_DIR = "learnings"
ZIP_PATH = "learnings.zip"
SIGNATURE_PATH = "signature.txt"


def enumerate_classes(data_dir):
    """Return {class_name: [image_paths]} for every subdir of data_dir."""
    classes = {}
    for name in sorted(os.listdir(data_dir)):
        sub = os.path.join(data_dir, name)
        if not os.path.isdir(sub):
            continue
        images = [
            os.path.join(sub, f)
            for f in sorted(os.listdir(sub))
            if is_image(f) and os.path.isfile(os.path.join(sub, f))
        ]
        if images:
            classes[name] = images
    return classes


def stratified_split(classes, val_fraction, rng):
    """Return (train_dict, val_dict) with per-class val_fraction held out."""
    train, val = {}, {}
    for cls, paths in classes.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        val[cls] = shuffled[:n_val]
        train[cls] = shuffled[n_val:]
    return train, val


def copy_originals(paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for src in paths:
        shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))


def augment_to_balance(src_paths, out_dir, target_count, rng):
    """Generate augmented variants until out_dir has `target_count` images.

    Cycles through originals and picks a random unused technique per pass.
    """
    existing = target_count - len(src_paths)
    if existing <= 0 or not src_paths:
        return
    n = len(src_paths)
    remaining = [list(range(len(TECHNIQUES))) for _ in range(n)]
    for order in remaining:
        rng.shuffle(order)
    pass_counts = [0] * n

    for i in range(existing):
        idx = i % n
        src = src_paths[idx]
        if not remaining[idx]:
            remaining[idx] = list(range(len(TECHNIQUES)))
            rng.shuffle(remaining[idx])
            pass_counts[idx] += 1
        technique_index = remaining[idx].pop()
        technique_name, technique_fn = TECHNIQUES[technique_index]
        try:
            img = load_image(src)
        except Exception as exc:
            print(f"Warning: skipping '{src}': {exc}", file=sys.stderr)
            continue
        variant = technique_fn(img, rng)
        stem, ext = os.path.splitext(os.path.basename(src))
        out_name = f"{stem}_{technique_name}_{pass_counts[idx]}{ext}"
        save_image(variant, os.path.join(out_dir, out_name))


def materialize_dataset(data_dir, aug_root, rng):
    """Build augmented_directory/{train,val}/<class>/ from scratch."""
    if os.path.exists(aug_root):
        shutil.rmtree(aug_root)
    os.makedirs(aug_root)

    classes = enumerate_classes(data_dir)
    if not classes:
        print(f"Error: no class subdirectories found in '{data_dir}'",
              file=sys.stderr)
        sys.exit(1)

    train_split, val_split = stratified_split(classes, VAL_FRACTION, rng)

    train_root = os.path.join(aug_root, "train")
    val_root = os.path.join(aug_root, "val")

    print(f"{'Class':<22} {'Train orig':>10} {'Train tgt':>10} "
          f"{'Val':>6}")
    train_target = max(len(p) for p in train_split.values())
    for cls in sorted(classes):
        train_out = os.path.join(train_root, cls)
        val_out = os.path.join(val_root, cls)
        copy_originals(train_split[cls], train_out)
        copy_originals(val_split[cls], val_out)
        augment_to_balance(train_split[cls], train_out, train_target, rng)
        print(f"{cls:<22} {len(train_split[cls]):>10} {train_target:>10} "
              f"{len(val_split[cls]):>6}")

    print(f"\nAugmented dataset written to '{aug_root}/'.")
    return train_root, val_root


def load_datasets(train_dir, val_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    class_names = train_ds.class_names
    assert class_names == val_ds.class_names, "class mismatch between splits"

    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names


def build_model(num_classes):
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
    return model, base


def training_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def finetune(model, base, train_ds, val_ds):
    """Unfreeze the top layers of MobileNetV2 and train at a tiny LR."""
    base.trainable = True
    for layer in base.layers[:-FINETUNE_UNFREEZE_LAYERS]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\nFine-tuning top {FINETUNE_UNFREEZE_LAYERS} layers "
          f"for {FINETUNE_EPOCHS} epochs...")
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        callbacks=training_callbacks(),
    )


def per_class_report(model, val_ds, class_names):
    """Print accuracy per class to spot weak points."""
    correct = [0] * len(class_names)
    total = [0] * len(class_names)
    for batch_x, batch_y in val_ds:
        preds = tf.argmax(model.predict(batch_x, verbose=0), axis=1).numpy()
        labels = batch_y.numpy()
        for p, lbl in zip(preds, labels):
            total[lbl] += 1
            if p == lbl:
                correct[lbl] += 1
    print("\nPer-class validation accuracy:")
    print(f"{'Class':<22} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    for name, c, t in zip(class_names, correct, total):
        acc = (c / t) if t else 0.0
        print(f"{name:<22} {c:>8} {t:>8} {acc:>8.4f}")


def save_artifacts(model, class_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.keras"))
    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)


def _add_tree_to_zip(zf, root):
    """Recursively add a directory tree to an open zipfile."""
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            arcname = os.path.relpath(path, start=".")
            zf.write(path, arcname)


def package_learnings(zip_path, signature_path):
    """Zip learnings/ + augmented_directory/, then write sha1 to signature."""
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        _add_tree_to_zip(zf, LEARNINGS_DIR)
        if os.path.isdir(AUG_ROOT):
            _add_tree_to_zip(zf, AUG_ROOT)

    sha1 = hashlib.sha1()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha1.update(chunk)
    digest = sha1.hexdigest()

    with open(signature_path, "w") as f:
        f.write(f"{digest}  {zip_path}\n")

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Wrote {zip_path} ({size_mb:.1f} MB)")
    print(f"Wrote {signature_path} ({digest})")


def main():
    parser = argparse.ArgumentParser(
        description="Train a leaf-disease classifier (Iteration 2)."
    )
    parser.add_argument(
        "directory",
        help="Dataset root (one subdirectory per class).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a directory",
              file=sys.stderr)
        sys.exit(1)

    rng = random.Random(SEED)

    print(f"Building augmented dataset from '{args.directory}'...")
    train_dir, val_dir = materialize_dataset(args.directory, AUG_ROOT, rng)

    print("\nLoading TF datasets...")
    train_ds, val_ds, class_names = load_datasets(train_dir, val_dir)
    print(f"Classes ({len(class_names)}): {class_names}")

    print("\nBuilding MobileNetV2 transfer-learning model...")
    model, base = build_model(len(class_names))
    model.summary()

    print(f"\nStage 1: training head for up to {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=training_callbacks(),
    )
    head_val_acc = max(history.history["val_accuracy"])
    print(f"\nStage 1 best val accuracy: {head_val_acc:.4f}")

    if head_val_acc < 0.95:
        finetune(model, base, train_ds, val_ds)

    per_class_report(model, val_ds, class_names)
    final_val_acc = model.evaluate(val_ds, verbose=0)[1]
    print(f"\nFinal validation accuracy: {final_val_acc:.4f}")

    print(f"Saving artifacts to ./{LEARNINGS_DIR}/ ...")
    save_artifacts(model, class_names, LEARNINGS_DIR)

    print("\nPackaging learnings + augmented dataset into zip...")
    package_learnings(ZIP_PATH, SIGNATURE_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
