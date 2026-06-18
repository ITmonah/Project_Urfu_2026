import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image


DATASETS_DIR = Path("datasets")
REQUIRED_LABELS = {"kgo_platform", "kgo_empty", "kgo_full"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def dataset_archives() -> list[Path]:
    return sorted(DATASETS_DIR.glob("*.zip"))


@pytest.mark.data_quality
def test_dataset_archives_are_versioned_and_available():
    archives = dataset_archives()

    assert archives, "No dataset archives found. Run `dvc pull datasets.dvc` before data quality tests."
    assert {archive.name for archive in archives} == {"39.zip", "40.zip", "42.zip"}


@pytest.mark.data_quality
@pytest.mark.parametrize("archive_path", dataset_archives(), ids=lambda path: path.name)
def test_datumaro_archive_structure_is_valid(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        names = {entry.filename for entry in archive.infolist() if not entry.is_dir()}
        annotation_names = sorted(name for name in names if name.startswith("annotations/") and name.endswith(".json"))
        image_names = {name for name in names if Path(name).suffix.lower() in IMAGE_SUFFIXES}

        assert len(annotation_names) == 1
        assert image_names

        annotation_payload = json.loads(archive.read(annotation_names[0]).decode("utf-8"))
        assert {"info", "categories", "items"}.issubset(annotation_payload)

        labels = annotation_payload["categories"]["label"]["labels"]
        label_names = {label["name"] for label in labels}
        assert REQUIRED_LABELS.issubset(label_names)

        items = annotation_payload["items"]
        assert items
        assert len(items) == len(image_names)

        for item in items:
            image_path = _image_archive_path(item, names)
            assert image_path in names

            for annotation in item.get("annotations", []):
                label_id = annotation.get("label_id")
                assert isinstance(label_id, int)
                assert 0 <= label_id < len(labels)

                if annotation.get("type") == "bbox":
                    bbox = annotation.get("bbox")
                    assert isinstance(bbox, list)
                    assert len(bbox) == 4
                    assert all(isinstance(value, (int, float)) for value in bbox)
                    assert bbox[2] > 0
                    assert bbox[3] > 0


@pytest.mark.data_quality
@pytest.mark.parametrize("archive_path", dataset_archives(), ids=lambda path: path.name)
def test_sample_images_are_decodable(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        image_entries = [
            entry
            for entry in archive.infolist()
            if not entry.is_dir() and Path(entry.filename).suffix.lower() in IMAGE_SUFFIXES
        ]

        assert image_entries
        for entry in image_entries[:10]:
            with Image.open(BytesIO(archive.read(entry))) as image:
                image.verify()


def _image_archive_path(item: dict, archive_names: set[str]) -> str:
    image_path = item["image"]["path"]
    if image_path.startswith("images/") and image_path in archive_names:
        return image_path

    image_name = Path(image_path).name
    subset = item.get("attr", {}).get("subset")
    candidates = []
    if subset:
        candidates.append(f"images/{subset}/{image_name}")
    candidates.extend(
        [
            f"images/default/{image_name}",
            f"images/train/{image_name}",
            f"images/val/{image_name}",
            f"images/test/{image_name}",
        ]
    )

    for candidate in candidates:
        if candidate in archive_names:
            return candidate

    return candidates[0]
