import argparse
import json
import re
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image


STATE_LABELS = {"kgo_empty", "kgo_full"}
TARGET_BOX_LABEL = "kgo_platform"
DEFAULT_DATASETS_DIR = Path("datasets")
DEFAULT_OUTPUT_DIR = Path("datasetCreation/cropped_datumaro")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Datumaro archives into cropped classification dataset.")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--archives",
        nargs="*",
        default=None,
        help="Optional list of Datumaro zip names to process. Defaults to all zip files in datasets dir.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove previously generated crops for kgo_empty/kgo_full before processing.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "item"


def find_annotation_entry(zip_file: zipfile.ZipFile) -> str:
    preferred = ("annotations/default.json", "annotations/train.json", "annotations/val.json", "annotations/test.json")
    names = {entry.filename for entry in zip_file.infolist()}
    for candidate in preferred:
        if candidate in names:
            return candidate

    for name in names:
        if name.startswith("annotations/") and name.endswith(".json"):
            return name

    raise FileNotFoundError("No Datumaro annotation json found inside archive.")


def find_image_entry(zip_file: zipfile.ZipFile, image_name: str) -> str:
    preferred = (
        f"images/default/{image_name}",
        f"images/train/{image_name}",
        f"images/val/{image_name}",
        f"images/test/{image_name}",
    )
    names = {entry.filename for entry in zip_file.infolist()}
    for candidate in preferred:
        if candidate in names:
            return candidate

    suffix = f"/{image_name}"
    for name in names:
        if name.endswith(suffix):
            return name

    raise FileNotFoundError(f"Image '{image_name}' not found in archive.")


def extract_categories(annotation_payload: dict) -> dict[int, str]:
    labels = annotation_payload["categories"]["label"]["labels"]
    return {idx: item["name"] for idx, item in enumerate(labels)}


def clear_output_dirs(output_dir: Path):
    for class_name in STATE_LABELS:
        class_dir = output_dir / class_name
        if not class_dir.exists():
            continue
        for image_path in class_dir.glob("*.jpg"):
            image_path.unlink()


def process_archive(archive_path: Path, output_dir: Path):
    stats = Counter()
    class_stats = Counter()
    zip_stem = archive_path.stem

    with zipfile.ZipFile(archive_path) as zip_file:
        annotation_entry = find_annotation_entry(zip_file)
        annotation_payload = json.loads(zip_file.read(annotation_entry).decode("utf-8"))
        label_map = extract_categories(annotation_payload)

        for item in annotation_payload.get("items", []):
            image_name = Path(item["image"]["path"]).name
            state_names = {
                label_map[annotation["label_id"]]
                for annotation in item.get("annotations", [])
                if annotation["type"] == "label" and label_map[annotation["label_id"]] in STATE_LABELS
            }

            if len(state_names) != 1:
                stats["skipped_missing_or_ambiguous_state"] += 1
                continue

            state_name = next(iter(state_names))
            target_boxes = [
                annotation["bbox"]
                for annotation in item.get("annotations", [])
                if annotation["type"] == "bbox" and label_map[annotation["label_id"]] == TARGET_BOX_LABEL
            ]

            if not target_boxes:
                stats["skipped_without_target_box"] += 1
                continue

            try:
                image_entry = find_image_entry(zip_file, image_name)
                image = Image.open(BytesIO(zip_file.read(image_entry))).convert("RGB")
            except Exception:
                stats["skipped_broken_image"] += 1
                continue

            class_dir = output_dir / state_name
            class_dir.mkdir(parents=True, exist_ok=True)
            item_prefix = sanitize_name(item.get("id") or Path(image_name).stem)

            for crop_idx, (x, y, width, height) in enumerate(target_boxes):
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(image.width, int(x + width))
                y2 = min(image.height, int(y + height))

                if x2 <= x1 or y2 <= y1:
                    stats["skipped_invalid_bbox"] += 1
                    continue

                crop = image.crop((x1, y1, x2, y2))
                output_name = f"{zip_stem}__{item_prefix}__crop_{crop_idx}.jpg"
                crop.save(class_dir / output_name, quality=95)
                stats["saved_crops"] += 1
                class_stats[state_name] += 1

    return stats, class_stats


def build_summary(output_dir: Path, archive_summaries: dict):
    total_by_class = {}
    for class_name in sorted(STATE_LABELS):
        total_by_class[class_name] = len(list((output_dir / class_name).glob("*.jpg")))

    summary = {
        "output_dir": str(output_dir),
        "archives": archive_summaries,
        "total_by_class": total_by_class,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    datasets_dir = args.datasets_dir
    output_dir = args.output_dir

    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    archives = (
        [datasets_dir / name for name in args.archives]
        if args.archives
        else sorted(datasets_dir.glob("*.zip"))
    )

    if not archives:
        raise FileNotFoundError(f"No zip archives found in {datasets_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_output:
        clear_output_dirs(output_dir)

    archive_summaries = {}
    total_stats = Counter()
    total_class_stats = Counter()

    for archive_path in archives:
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        archive_stats, archive_class_stats = process_archive(archive_path, output_dir)
        total_stats.update(archive_stats)
        total_class_stats.update(archive_class_stats)
        archive_summaries[archive_path.name] = {
            "stats": dict(archive_stats),
            "class_distribution": dict(archive_class_stats),
        }

    build_summary(output_dir, archive_summaries)

    print("Processed archives:")
    for archive_name, summary in archive_summaries.items():
        print(f"  {archive_name}: {summary['stats']}")
    print(f"Total stats: {dict(total_stats)}")
    print(f"Total class distribution: {dict(total_class_stats)}")
    print(f"Saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
