import argparse
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

STATE_LABELS = ("kgo_empty", "kgo_full", "kgo_none")
DEFAULT_DATASETS_DIR = Path("datasets")
DEFAULT_OUTPUT_DIR = Path("classification_dataset")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create classification dataset (kgo_empty/kgo_full/kgo_none) from Datumaro archives."
    )
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
        help="Remove all images in output class folders before processing.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "item"


def pick_entry(names, preferred, fallback):
    for candidate in preferred:
        if candidate in names:
            return candidate
    return next(name for name in names if fallback(name))


def find_image_entry(zip_file: zipfile.ZipFile, image_name: str) -> str:
    names = {entry.filename for entry in zip_file.infolist()}
    return pick_entry(
        names,
        (
            f"images/default/{image_name}",
            f"images/train/{image_name}",
            f"images/val/{image_name}",
            f"images/test/{image_name}",
        ),
        lambda name: name.endswith(f"/{image_name}"),
    )


def extract_categories(annotation_payload: dict) -> dict[int, str]:
    labels = annotation_payload["categories"]["label"]["labels"]
    return {idx: item["name"] for idx, item in enumerate(labels)}


def clear_output_dirs(output_dir: Path):
    for class_name in STATE_LABELS:
        class_dir = output_dir / class_name
        if class_dir.exists():
            for image_path in class_dir.glob("*.jpg"):
                image_path.unlink()


def process_archive(archive_path: Path, output_dir: Path):
    saved = {class_name: 0 for class_name in STATE_LABELS}
    zip_stem = archive_path.stem

    with zipfile.ZipFile(archive_path) as zip_file:
        annotation_entry = pick_entry(
            {entry.filename for entry in zip_file.infolist()},
            ("annotations/default.json", "annotations/train.json", "annotations/val.json", "annotations/test.json"),
            lambda name: name.startswith("annotations/") and name.endswith(".json"),
        )
        annotation_payload = json.loads(zip_file.read(annotation_entry).decode("utf-8"))
        label_map = extract_categories(annotation_payload)

        for item in annotation_payload.get("items", []):
            image_name = Path(item["image"]["path"]).name
            state_names = {
                label_map[annotation["label_id"]]
                for annotation in item.get("annotations", [])
                if annotation["type"] == "label" and label_map[annotation["label_id"]] in STATE_LABELS
            }

            if len(state_names) == 0:
                class_name = "kgo_none"
            elif len(state_names) == 1:
                class_name = next(iter(state_names))
            else:
                continue

            try:
                image_entry = find_image_entry(zip_file, image_name)
                image = Image.open(BytesIO(zip_file.read(image_entry))).convert("RGB")
            except Exception:
                continue

            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            item_id = sanitize_name(item.get("id") or Path(image_name).stem)
            output_filename = f"{zip_stem}__{item_id}.jpg"
            image.save(class_dir / output_filename, quality=95)
            saved[class_name] += 1

    return saved


def build_summary(output_dir: Path, archive_summaries: dict):
    total = {}
    for class_name in STATE_LABELS:
        class_dir = output_dir / class_name
        total[class_name] = len(list(class_dir.glob("*.jpg"))) if class_dir.exists() else 0
    summary = {
        "output_dir": str(output_dir),
        "archives": archive_summaries,
        "total_by_class": total,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    datasets_dir = args.datasets_dir
    output_dir = args.output_dir

    archives = [datasets_dir / name for name in args.archives] if args.archives else sorted(datasets_dir.glob("*.zip"))

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_output:
        clear_output_dirs(output_dir)

    archive_summaries = {}
    for archive_path in archives:
        archive_summaries[archive_path.name] = {
            "class_distribution": process_archive(archive_path, output_dir),
        }

    build_summary(output_dir, archive_summaries)

    print("Processed archives:")
    for archive_name, summary in archive_summaries.items():
        print(f"  {archive_name}: {summary['class_distribution']}")
    print(f"Saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()