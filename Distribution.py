import os
import sys

import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def is_image(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def load_categories(root_dir):
    """Return {category: {subcategory: [image_paths]}}."""
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"'{root_dir}' is not a valid directory")

    categories = {}

    for entry in sorted(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, entry)
        if not os.path.isdir(subdir_path):
            continue

        category = entry.split("_", 1)[0]

        try:
            files = os.listdir(subdir_path)
        except PermissionError as e:
            print(
                f"Warning: cannot read '{subdir_path}': {e}",
                file=sys.stderr,
            )
            continue

        images = [
            os.path.join(subdir_path, f)
            for f in files
            if is_image(f) and os.path.isfile(os.path.join(subdir_path, f))
        ]

        categories.setdefault(category, {})[entry] = images

    return categories


def counts_by_subcategory(subcategories):
    """Return (labels, counts) sorted by label."""
    labels = sorted(subcategories.keys())
    counts = [len(subcategories[label]) for label in labels]
    return labels, counts


def plot_category(category, subcategories):
    labels, counts = counts_by_subcategory(subcategories)

    fig, (pie_ax, bar_ax) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{category} distribution", fontsize=14)

    pie_ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
    pie_ax.set_title("Share per subcategory")

    bar_ax.bar(labels, counts)
    bar_ax.set_title("Image count per subcategory")
    bar_ax.set_ylabel("Number of images")
    bar_ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()


def print_summary(categories):
    for category, subcategories in categories.items():
        total = sum(len(imgs) for imgs in subcategories.values())
        print(
            f"{category}: {total} images "
            f"across {len(subcategories)} subcategories"
        )
        for name, imgs in subcategories.items():
            print(f"  {name}: {len(imgs)}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <directory>", file=sys.stderr)
        sys.exit(1)

    try:
        categories = load_categories(sys.argv[1])
    except (NotADirectoryError, FileNotFoundError, PermissionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not categories:
        print("No image subdirectories found.", file=sys.stderr)
        sys.exit(1)

    print_summary(categories)

    for category, subcategories in categories.items():
        plot_category(category, subcategories)

    plt.show()


if __name__ == "__main__":
    main()
