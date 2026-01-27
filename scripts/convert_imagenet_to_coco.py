import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def read_wnids(wnids_path: Path) -> List[str]:
    if not wnids_path:
        return []
    if not wnids_path.exists():
        return []
    return [line.strip() for line in wnids_path.read_text().splitlines() if line.strip()]


def read_words(words_path: Path) -> Dict[str, str]:
    if not words_path or not words_path.exists():
        return {}
    mapping = {}
    for line in words_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            key, rest = line.split("\t", 1)
        except ValueError:
            continue
        mapping[key.strip()] = rest.strip()
    return mapping


def image_size(img_path: Path) -> Tuple[int, int]:
    with Image.open(img_path) as img:
        return img.width, img.height


def build_categories(wnids: List[str], words: Dict[str, str]) -> List[Dict]:
    categories = []
    for idx, wnid in enumerate(wnids):
        categories.append({
            "id": idx,
            "name": wnid,
            "supercategory": words.get(wnid, "")
        })
    return categories


def add_annotation(annotations: List[Dict], ann_id: int, image_id: int, category_id: int, width: int, height: int) -> int:
    annotations.append({
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [0, 0, width, height],
        "area": width * height,
        "iscrowd": 0,
        "segmentation": []
    })
    return ann_id + 1


def link_or_copy(src: Path, dst: Path, copy: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src)
        except OSError:
            # fallback when symlinks are not allowed
            shutil.copy2(src, dst)


def gather_from_directory(
    split_dir: Path,
    split_name: str,
    label_map: Dict[str, int],
    images_out_dir: Path,
    flatten: bool = False,
    copy_images: bool = False,
    progress: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    images: List[Dict] = []
    annotations: List[Dict] = []
    ann_id = 1
    image_id = 1

    val_annotations = split_dir / "val_annotations.txt"
    if val_annotations.exists():
        # tiny-imagenet style validation file
        lines = val_annotations.read_text().splitlines()
        total = len(lines)
        for idx, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) < 2:
                continue
            filename, wnid = parts[0], parts[1]
            category_id = label_map.get(wnid)
            if category_id is None:
                continue
            img_path = split_dir / "images" / filename
            if not img_path.exists():
                img_path = split_dir / filename
            if not img_path.exists():
                continue
            width, height = image_size(img_path)
            rel_name = f"{wnid}_{filename}" if flatten else f"{wnid}/{filename}"
            link_or_copy(img_path, images_out_dir / rel_name, copy=copy_images)
            images.append({
                "id": image_id,
                "file_name": rel_name,
                "width": width,
                "height": height
            })
            ann_id = add_annotation(annotations, ann_id, image_id, category_id, width, height)
            image_id += 1
            if progress and (idx % 1000 == 0 or idx == total):
                print(f"[{split_name}] processed {idx}/{total} entries")
        return images, annotations

    # standard ImageNet-style folders: class subdirectories (optionally with an inner "images" folder as in tiny-imagenet)
    for wnid_dir in sorted(split_dir.iterdir()):
        if not wnid_dir.is_dir():
            continue
        wnid = wnid_dir.name
        category_id = label_map.get(wnid)
        if category_id is None:
            continue
        img_dir = wnid_dir / "images" if (wnid_dir / "images").exists() else wnid_dir
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            width, height = image_size(img_path)
            rel_name = f"{wnid}_{img_path.name}" if flatten else f"{wnid}/{img_path.name}"
            link_or_copy(img_path, images_out_dir / rel_name, copy=copy_images)
            images.append({
                "id": image_id,
                "file_name": rel_name,
                "width": width,
                "height": height
            })
            ann_id = add_annotation(annotations, ann_id, image_id, category_id, width, height)
            image_id += 1
            if progress and image_id % 5000 == 0:
                print(f"[{split_name}] processed {image_id - 1} images")
    if progress and image_id > 1:
        print(f"[{split_name}] processed {image_id - 1} images")
    return images, annotations


def convert_split(
    root: Path,
    split: str,
    categories: List[Dict],
    label_map: Dict[str, int],
    output_dir: Path,
    flatten: bool = False,
    copy_images: bool = False,
    progress: bool = False,
):
    split_dir = root / split
    if not split_dir.exists():
        print(f"[warn] split {split_dir} not found; skipping")
        return

    images_out_dir = output_dir / f"{split}2017"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = gather_from_directory(
        split_dir,
        split,
        label_map,
        images_out_dir,
        flatten=flatten,
        copy_images=copy_images,
        progress=progress,
    )
    coco = {
        "info": {
            "description": f"ImageNet to COCO classification ({split})",
            "version": "1.1"
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    out_file = ann_dir / f"instances_{split}2017.json"
    out_file.write_text(json.dumps(coco, indent=2))
    print(f"Wrote {len(images)} images and {len(annotations)} annotations to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert ImageNet-style dataset to RF-DETR COCO-style classification layout")
    parser.add_argument("dataset_root", type=Path, help="Path to ImageNet-style dataset root (with train/val/test)")
    parser.add_argument("output_dir", type=Path, help="Where to write COCO-formatted dataset")
    parser.add_argument("--wnids", type=Path, default=None, help="Optional wnids.txt file; if absent, class folders are used")
    parser.add_argument("--words", type=Path, default=None, help="Optional words.txt for human-readable names")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Dataset splits to convert")
    parser.add_argument("--flatten", action="store_true", help="Do not nest images under class folders; prefixes wnid to filenames")
    parser.add_argument("--copy", dest="copy_images", action="store_true", help="Copy images instead of symlinking")
    parser.add_argument("--progress", action="store_true", help="Print progress while converting")

    args = parser.parse_args()

    wnids = read_wnids(args.wnids) if args.wnids else []
    if not wnids and args.dataset_root.exists():
        # derive from training folders if wnids not provided
        train_dir = args.dataset_root / "train"
        if train_dir.exists():
            wnids = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    words = read_words(args.words) if args.words else {}

    categories = build_categories(wnids, words)
    label_map = {cat["name"]: cat["id"] for cat in categories}

    if not categories:
        raise ValueError("No categories found. Provide --wnids or ensure train subfolders exist.")

    for split in args.splits:
        convert_split(
            args.dataset_root,
            split,
            categories,
            label_map,
            args.output_dir,
            flatten=args.flatten,
            copy_images=args.copy_images,
            progress=args.progress,
        )


if __name__ == "__main__":
    main()
