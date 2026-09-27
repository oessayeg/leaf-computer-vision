import argparse
import os
import sys
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image, ImageOps, UnidentifiedImageError


def flip(img):
    return ImageOps.mirror(img)


def rotate(img):
    return img.rotate(30, resample=Image.BILINEAR, expand=False)


def skew(img):
    w, h = img.size
    dx = w * 0.25
    coeffs = _perspective_coeffs(
        src=[(0, 0), (w, 0), (w, h), (0, h)],
        dst=[(dx, 0), (w, 0), (w - dx, h), (0, h)],
    )
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)


def shear(img):
    w, h = img.size
    shear_factor = 0.2
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, shear_factor, -shear_factor * h / 2, 0, 1, 0),
        Image.BILINEAR,
    )


def crop(img):
    w, h = img.size
    box = (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))
    return img.crop(box).resize((w, h), Image.BILINEAR)


def distortion(img):
    w, h = img.size
    step = 40
    mesh = []
    for x in range(0, w, step):
        for y in range(0, h, step):
            x2 = min(x + step, w)
            y2 = min(y + step, h)
            src_quad = (
                x, y,
                x, y2,
                x2, y2,
                x2, y,
            )
            offset = 6
            dst_box = (x, y, x2, y2)
            src_quad = (
                x + offset, y,
                x, y2 - offset,
                x2 - offset, y2,
                x2, y + offset,
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
    if os.path.isdir(path):
        raise IsADirectoryError(f"'{path}' is a directory (directory mode not implemented in v1)")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"'{path}' is not a regular file")
    return Image.open(path).convert("RGB")


def build_output_path(src, technique):
    directory, filename = os.path.split(src)
    stem, ext = os.path.splitext(filename)
    return os.path.join(directory, f"{stem}_{technique}{ext}")


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


def main():
    parser = argparse.ArgumentParser(
        description="Apply 6 augmentation techniques to a single image."
    )
    parser.add_argument("image_path", help="Path to the source image")
    args = parser.parse_args()

    try:
        original = load_image(args.image_path)
    except (FileNotFoundError, IsADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except UnidentifiedImageError:
        print(f"Error: '{args.image_path}' is not a valid image", file=sys.stderr)
        sys.exit(1)

    variants = apply_all(original)

    for name, image in variants.items():
        out_path = build_output_path(args.image_path, name)
        try:
            save_image(image, out_path)
        except PermissionError as e:
            print(f"Error: cannot write '{out_path}': {e}", file=sys.stderr)
            sys.exit(1)

    display(original, variants)


if __name__ == "__main__":
    main()
