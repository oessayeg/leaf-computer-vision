import argparse
import os
import random
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from Distribution import load_categories


def flip(img, rng=None):
    if rng is not None and rng.random() < 0.5:
        return ImageOps.flip(img)
    return ImageOps.mirror(img)


def rotate(img, rng=None):
    angle = rng.uniform(-45, 45) if rng else 30
    return img.rotate(angle, resample=Image.BILINEAR, expand=False)


def skew(img, rng=None):
    w, h = img.size
    ratio = rng.uniform(0.15, 0.35) if rng else 0.25
    dx = w * ratio
    coeffs = _perspective_coeffs(
        src=[(0, 0), (w, 0), (w, h), (0, h)],
        dst=[(dx, 0), (w, 0), (w - dx, h), (0, h)],
    )
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)


def shear(img, rng=None):
    w, h = img.size
    factor = rng.uniform(-0.3, 0.3) if rng else 0.2
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, factor, -factor * h / 2, 0, 1, 0),
        Image.BILINEAR,
    )


def crop(img, rng=None):
    w, h = img.size
    scale = rng.uniform(0.6, 0.85) if rng else 0.7
    cw, ch = int(w * scale), int(h * scale)
    if rng:
        left = rng.randint(0, w - cw)
        top = rng.randint(0, h - ch)
    else:
        left = (w - cw) // 2
        top = (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.BILINEAR)


def distortion(img, rng=None):
    w, h = img.size
    step = 40
    offset_range = (3, 10) if rng else (6, 6)
    mesh = []
    for x in range(0, w, step):
        for y in range(0, h, step):
            x2 = min(x + step, w)
            y2 = min(y + step, h)
            off = rng.randint(*offset_range) if rng else 6
            dst_box = (x, y, x2, y2)
            src_quad = (
                x + off, y,
                x, y2 - off,
                x2 - off, y2,
                x2, y + off,
            )
            mesh.append((dst_box, src_quad))
    return img.transform((w, h), Image.MESH, mesh, Image.BILINEAR)


TECHNIQUES = [
    ("Flip", flip),
    ("Rotate", rotate),
    ("Skew", skew),
    ("Shear", shear),
    ("Crop", crop),
    ("Distortion", distortion),
]


def _perspective_coeffs(src, dst):
    matrix = []
    for (x, y), (X, Y) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=float)
    B = np.array(src, dtype=float).reshape(8)
    return list(np.linalg.solve(A, B))


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' does not exist")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"'{path}' is not a regular file")
    return Image.open(path).convert("RGB")


def build_output_path(src, technique, out_dir=None, suffix_index=None):
    directory, filename = os.path.split(src)
    stem, ext = os.path.splitext(filename)
    if suffix_index is None:
        tag = f"_{technique}"
    else:
        tag = f"_{technique}_{suffix_index}"
    target_dir = out_dir if out_dir is not None else directory
    return os.path.join(target_dir, f"{stem}{tag}{ext}")


def save_image(img, path):
    img.save(path)


def apply_all(img):
    return {name: fn(img) for name, fn in TECHNIQUES}


def display(original, variants):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    panels = [("Original", original)] + list(variants.items())
    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


# --- single-image mode ---------------------------------------------------

def run_single_image(path):
    original = load_image(path)
    variants = apply_all(original)
    for name, image in variants.items():
        save_image(image, build_output_path(path, name))
    display(original, variants)


# --- directory / balancing mode -----------------------------------------

def compute_targets(categories):
    """Return {subcategory: needed_count} using per-group max as target."""
    needed = {}
    for _category, subcategories in categories.items():
        target = max(len(imgs) for imgs in subcategories.values())
        for name, imgs in subcategories.items():
            needed[name] = (target, target - len(imgs))
    return needed


def mirror_originals(categories, dst_root):
    """Copy every original image into dst_root/<subcategory>/."""
    for subcategories in categories.values():
        for name, imgs in subcategories.items():
            out_dir = os.path.join(dst_root, name)
            os.makedirs(out_dir, exist_ok=True)
            for src in imgs:
                shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))


def augment_subcategory(src_paths, out_dir, needed, rng):
    """Generate `needed` augmented images by cycling through originals and
    picking a random unused technique per original each pass."""
    if needed <= 0 or not src_paths:
        return
    n = len(src_paths)
    remaining_techniques = [list(range(len(TECHNIQUES))) for _ in range(n)]
    for order in remaining_techniques:
        rng.shuffle(order)
    pass_counts = [0] * n

    for i in range(needed):
        idx = i % n
        src = src_paths[idx]
        if not remaining_techniques[idx]:
            remaining_techniques[idx] = list(range(len(TECHNIQUES)))
            rng.shuffle(remaining_techniques[idx])
            pass_counts[idx] += 1
        technique_index = remaining_techniques[idx].pop()
        technique_name, technique_fn = TECHNIQUES[technique_index]

        try:
            img = load_image(src)
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Warning: skipping '{src}': {e}", file=sys.stderr)
            continue
        variant = technique_fn(img, rng)
        out_path = build_output_path(
            src, technique_name, out_dir=out_dir, suffix_index=pass_counts[idx]
        )
        save_image(variant, out_path)


def run_directory(src_root, dst_root, seed):
    categories = load_categories(src_root)
    if not categories:
        print("No image subdirectories found.", file=sys.stderr)
        sys.exit(1)

    if os.path.abspath(src_root) == os.path.abspath(dst_root):
        print("Error: destination must differ from source", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_root)
    mirror_originals(categories, dst_root)

    targets = compute_targets(categories)
    rng = random.Random(seed)

    print(f"{'Subcategory':<25} {'Before':>8} {'Target':>8} {'Added':>8}")
    for subcategories in categories.values():
        for name, imgs in subcategories.items():
            target, needed = targets[name]
            out_dir = os.path.join(dst_root, name)
            augment_subcategory(imgs, out_dir, needed, rng)
            print(f"{name:<25} {len(imgs):>8} {target:>8} {needed:>8}")

    print(f"\nBalanced dataset written to: {dst_root}")


# --- CLI --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply 6 augmentation techniques to an image, "
            "or balance a dataset directory."
        )
    )
    parser.add_argument(
        "path",
        help="Image file (single-image mode) or directory (balancing mode)",
    )
    parser.add_argument(
        "--dst",
        default="augmented_directory",
        help="Destination directory (directory mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=25,
        help="RNG seed for reproducible augmentation",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: '{args.path}' does not exist", file=sys.stderr)
        sys.exit(1)

    try:
        if os.path.isdir(args.path):
            run_directory(args.path, args.dst, args.seed)
        else:
            run_single_image(args.path)
    except UnidentifiedImageError:
        print(f"Error: '{args.path}' is not a valid image", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
