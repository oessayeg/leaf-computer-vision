import json
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def main():
    val_dir = sys.argv[1] if len(sys.argv) > 1 else "augmented_directory/val"
    model = tf.keras.models.load_model("learnings/model.keras")
    class_names = json.load(open("learnings/class_names.json"))

    ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(224, 224), batch_size=32,
        label_mode="int", shuffle=False, class_names=class_names,
    ).map(lambda x, y: (preprocess_input(x), y))

    correct = total = 0
    for x, y in ds:
        preds = np.argmax(model.predict(x, verbose=0), axis=1)
        labels = y.numpy()
        correct += int((preds == labels).sum())
        total += len(labels)

    acc = correct / total
    print(f"Validation: {correct}/{total} = {acc:.4f}")
    print("PASS (>= 90%)" if acc >= 0.90 else "FAIL (< 90%)")
    return 0 if acc >= 0.90 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
