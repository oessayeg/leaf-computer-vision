import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from plantcv import plantcv as pcv

from Distribution import is_image

pcv.params.debug = None
pcv.params.verbose = False


TRANSFORMS = [
    "Original",
    "GaussianBlur",
    "Mask",
    "Roi",
    "AnalyzeObject",
    "Pseudolandmarks",
    "ColorHistogram",
]

FLAG_TO_NAME = {
    "original": "Original",
    "blur": "GaussianBlur",
    "mask": "Mask",
    "roi": "Roi",
    "analyze": "AnalyzeObject",
    "landmarks": "Pseudolandmarks",
    "hist": "ColorHistogram",
}

HIST_CHANNELS = [
    ("blue", ("rgb", 0), "blue"),
    ("blue-yellow", ("lab", 2), "gold"),
    ("green", ("rgb", 1), "green"),
    ("green-magenta", ("lab", 1), "magenta"),
    ("hue", ("hsv", 0), "purple"),
    ("lightness", ("lab", 0), "gray"),
    ("red", ("rgb", 2), "red"),
    ("saturation", ("hsv", 1), "cyan"),
    ("value", ("hsv", 2), "orange"),
]


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' does not exist")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"'{path}' is not a regular file")
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"'{path}' is not a valid image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_mask(rgb):
    """Isolate the leaf from the background using the LAB b-channel."""
    b = pcv.rgb2gray_lab(rgb_img=rgb, channel="b")
    binary = pcv.threshold.binary(gray_img=b, threshold=135,
                                  object_type="light")
    return pcv.fill(bin_img=binary, size=200)


def t_original(rgb):
    return rgb


def t_gaussian_blur(rgb):
    b = pcv.rgb2gray_lab(rgb_img=rgb, channel="b")
    binary = pcv.threshold.binary(gray_img=b, threshold=135,
                                  object_type="light")
    blurred = pcv.gaussian_blur(img=binary, ksize=(5, 5), sigma_x=0)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)


def t_mask(rgb, mask):
    return pcv.apply_mask(img=rgb, mask=mask, mask_color="white")


def t_roi(rgb, mask):
    """Overlay the leaf in green and draw a blue ROI rectangle."""
    h, w = rgb.shape[:2]
    out = rgb.copy()
    out[mask > 0] = (0, 255, 0)
    thickness = max(2, min(h, w) // 80)
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness)
    return out


def t_analyze_object(rgb, mask):
    labeled = (mask > 0).astype(np.int32)
    return pcv.analyze.size(img=rgb.copy(), labeled_mask=labeled)


def t_pseudolandmarks(rgb, mask):
    out = rgb.copy()
    if not np.any(mask):
        return out
    try:
        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
            img=rgb, mask=mask
        )
        left, right, center_h = pcv.homology.y_axis_pseudolandmarks(
            img=rgb, mask=mask
        )
    except Exception as exc:
        print(f"Warning: pseudolandmarks failed: {exc}", file=sys.stderr)
        return out

    groups = [
        (top, (255, 0, 0)),
        (bottom, (0, 0, 255)),
        (center_v, (255, 0, 255)),
        (left, (255, 128, 0)),
        (right, (0, 255, 255)),
        (center_h, (255, 0, 255)),
    ]
    for pts, color in groups:
        if not isinstance(pts, np.ndarray):
            continue
        for p in pts:
            try:
                x, y = int(p[0][0]), int(p[0][1])
            except (TypeError, ValueError):
                continue
            cv2.circle(out, (x, y), 3, color, -1)
    return out


def _channel(rgb, space, index):
    if space == "rgb":
        return rgb[:, :, index]
    if space == "hsv":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, index]
    if space == "lab":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, index]
    raise ValueError(space)


def color_histogram_figure(rgb, mask):
    """Build a 9-channel color histogram figure (restricted to the leaf)."""
    fig = Figure(figsize=(8, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    leaf_pixels = mask > 0
    total = int(leaf_pixels.sum()) or 1
    for name, (space, idx), color in HIST_CHANNELS:
        channel = _channel(rgb, space, idx)
        values = channel[leaf_pixels]
        hist, _ = np.histogram(values, bins=256, range=(0, 256))
        proportion = hist / total * 100.0
        ax.plot(proportion, label=name, color=color, linewidth=1)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.set_title("Color histogram")
    ax.legend(title="color Channel", loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def compute_panels(rgb):
    mask = build_mask(rgb)
    return {
        "Original": t_original(rgb),
        "GaussianBlur": t_gaussian_blur(rgb),
        "Mask": t_mask(rgb, mask),
        "Roi": t_roi(rgb, mask),
        "AnalyzeObject": t_analyze_object(rgb, mask),
        "Pseudolandmarks": t_pseudolandmarks(rgb, mask),
    }, mask


def build_output_path(src, name, out_dir):
    stem, ext = os.path.splitext(os.path.basename(src))
    return os.path.join(out_dir, f"{stem}_{name}{ext}")


def save_rgb(img, path):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def display(panels, hist_fig):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    items = list(panels.items())
    for ax, (title, image) in zip(axes.flat, items):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    last = axes.flat[len(items)]
    hist_fig.canvas.draw()
    hist_array = np.asarray(hist_fig.canvas.buffer_rgba())
    last.imshow(hist_array)
    last.set_title("ColorHistogram")
    last.axis("off")
    for ax in list(axes.flat)[len(items) + 1:]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def run_single_image(path):
    rgb = load_image(path)
    panels, mask = compute_panels(rgb)
    hist_fig = color_histogram_figure(rgb, mask)
    display(panels, hist_fig)


def process_file(src, out_dir, selected):
    try:
        rgb = load_image(src)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Warning: skipping '{src}': {exc}", file=sys.stderr)
        return
    try:
        panels, mask = compute_panels(rgb)
        for name, image in panels.items():
            if name in selected:
                save_rgb(image, build_output_path(src, name, out_dir))
        if "ColorHistogram" in selected:
            hist_fig = color_histogram_figure(rgb, mask)
            hist_fig.savefig(build_output_path(src, "ColorHistogram", out_dir))
    except Exception as exc:
        print(f"Warning: failed to process '{src}': {exc}", file=sys.stderr)


def iter_source_images(src):
    if os.path.isfile(src):
        yield src, "."
        return
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        for name in sorted(files):
            if is_image(name):
                yield os.path.join(root, name), rel


def run_directory(src, dst, selected):
    if not os.path.exists(src):
        print(f"Error: '{src}' does not exist", file=sys.stderr)
        sys.exit(1)
    if os.path.isdir(src) and os.path.abspath(src) == os.path.abspath(dst):
        print("Error: destination must differ from source", file=sys.stderr)
        sys.exit(1)
    os.makedirs(dst, exist_ok=True)

    count = 0
    for src_file, rel in iter_source_images(src):
        out_dir = dst if rel in (".", "") else os.path.join(dst, rel)
        os.makedirs(out_dir, exist_ok=True)
        process_file(src_file, out_dir, selected)
        count += 1
    print(f"Processed {count} image(s). Output: {dst}")


def selected_transforms(args):
    flags = ["original", "blur", "mask", "roi", "analyze", "landmarks", "hist"]
    picked = {FLAG_TO_NAME[f] for f in flags if getattr(args, f)}
    return picked if picked else set(TRANSFORMS)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply 6+ image transformations to a leaf image, or batch-process"
            " a directory."
        )
    )
    parser.add_argument(
        "path", nargs="?",
        help="Image file (single-image display mode).",
    )
    parser.add_argument("-src", help="Source directory (batch mode).")
    parser.add_argument("-dst", help="Destination directory (batch mode).")
    parser.add_argument("-original", action="store_true",
                        help="Save the original image.")
    parser.add_argument("-blur", action="store_true",
                        help="Save the Gaussian blur transformation.")
    parser.add_argument("-mask", action="store_true",
                        help="Save the mask transformation.")
    parser.add_argument("-roi", action="store_true",
                        help="Save the ROI objects transformation.")
    parser.add_argument("-analyze", action="store_true",
                        help="Save the analyze-object transformation.")
    parser.add_argument("-landmarks", action="store_true",
                        help="Save the pseudolandmarks transformation.")
    parser.add_argument("-hist", action="store_true",
                        help="Save the color histogram.")
    args = parser.parse_args()

    if args.src or args.dst:
        if not args.src or not args.dst:
            print("Error: -src and -dst must be provided together",
                  file=sys.stderr)
            sys.exit(1)
        try:
            run_directory(args.src, args.dst, selected_transforms(args))
        except PermissionError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if not args.path:
        parser.error("provide an image path or use -src/-dst for batch mode")

    try:
        run_single_image(args.path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
