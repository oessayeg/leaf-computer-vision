import os
import sys

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def is_image(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def load_categories(root_dir):
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
            print(f"Warning: cannot read '{subdir_path}': {e}", file=sys.stderr)
            continue

        images = [
            os.path.join(subdir_path, f)
            for f in files
            if is_image(f) and os.path.isfile(os.path.join(subdir_path, f))
        ]

        bucket = categories.setdefault(category, {})
        bucket[entry] = images

    return categories


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <directory>", file=sys.stderr)
        sys.exit(1)

    try:
        categories = load_categories(sys.argv[1])
    except (NotADirectoryError, FileNotFoundError, PermissionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    for category, subdirs in categories.items():
        total = sum(len(imgs) for imgs in subdirs.values())
        print(f"{category}: {total} images across {len(subdirs)} subcategories")
        for name, imgs in subdirs.items():
            print(f"  {name}: {len(imgs)}")


if __name__ == "__main__":
    main()
