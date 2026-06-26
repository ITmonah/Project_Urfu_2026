import json

import pytest

from scripts.evaluate_metrics import (
    PredictionRow,
    build_classification_metrics,
    load_datumaro_samples,
    merge_dataset_results,
    normalize_prediction,
)


def write_datumaro_dataset(root, items):
    annotations_dir = root / "annotations"
    annotations_dir.mkdir(parents=True)
    payload = {
        "categories": {
            "label": {
                "labels": [
                    {"name": "kgo_full"},
                    {"name": "kgo_empty"},
                    {"name": "kgo_platform"},
                    {"name": "garbage_can"},
                ]
            }
        },
        "items": items,
    }
    (annotations_dir / "default.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_datumaro_samples_keeps_only_labeled_state_images(tmp_path):
    dataset_root = tmp_path / "dataset"
    write_datumaro_dataset(
        dataset_root,
        [
            {
                "image": {"path": "unlabeled.jpg"},
                "annotations": [{"type": "bbox", "label_id": 2, "bbox": [0, 0, 10, 10]}],
            },
            {
                "image": {"path": "empty.jpg"},
                "annotations": [{"type": "label", "label_id": 1}],
            },
            {
                "image": {"path": "full.jpg"},
                "annotations": [
                    {"type": "label", "label_id": 0},
                    {"type": "bbox", "label_id": 2, "bbox": [0, 0, 10, 10]},
                ],
            },
        ],
    )

    result = load_datumaro_samples(dataset_root)

    assert result.total_items == 3
    assert result.unlabeled_items == 1
    assert result.multi_label_items == 0
    assert result.label_counts == {"kgo_empty": 1, "kgo_full": 1}
    assert [(sample.dataset, sample.image, sample.true_label) for sample in result.samples] == [
        ("dataset", "empty.jpg", "kgo_empty"),
        ("dataset", "full.jpg", "kgo_full"),
    ]
    assert result.samples[0].image_path == dataset_root / "images" / "default" / "empty.jpg"


def test_merge_dataset_results_combines_samples_and_counts(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    write_datumaro_dataset(
        first_root,
        [
            {"image": {"path": "empty.jpg"}, "annotations": [{"type": "label", "label_id": 1}]},
            {"image": {"path": "skip.jpg"}, "annotations": []},
        ],
    )
    write_datumaro_dataset(
        second_root,
        [
            {"image": {"path": "full.jpg"}, "annotations": [{"type": "label", "label_id": 0}]},
        ],
    )

    merged = merge_dataset_results(
        [
            load_datumaro_samples(first_root, dataset_name="first"),
            load_datumaro_samples(second_root, dataset_name="second"),
        ]
    )

    assert merged.total_items == 3
    assert merged.unlabeled_items == 1
    assert merged.label_counts == {"kgo_empty": 1, "kgo_full": 1}
    assert [(sample.dataset, sample.image) for sample in merged.samples] == [
        ("first", "empty.jpg"),
        ("second", "full.jpg"),
    ]


def test_normalize_prediction_maps_empty_or_unknown_to_no_zone():
    assert normalize_prediction("") == "no_zone"
    assert normalize_prediction(None) == "no_zone"
    assert normalize_prediction("missing") == "no_zone"
    assert normalize_prediction("kgo_full") == "kgo_full"


def test_build_classification_metrics_counts_no_zone_as_prediction_error():
    rows = [
        PredictionRow("dataset", "1.jpg", "kgo_empty", "kgo_empty", True),
        PredictionRow("dataset", "2.jpg", "kgo_empty", "kgo_full", False),
        PredictionRow("dataset", "3.jpg", "kgo_full", "kgo_full", True),
        PredictionRow("dataset", "4.jpg", "kgo_full", "no_zone", False),
    ]

    metrics = build_classification_metrics(rows)

    assert metrics["total"] == 4
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["macro_precision"] == pytest.approx(0.75)
    assert metrics["macro_recall"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert metrics["weighted_f1"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert metrics["support"] == {"kgo_empty": 2, "kgo_full": 2}
    assert metrics["no_zone_count"] == 1
    assert metrics["confusion_labels"] == ["kgo_empty", "kgo_full", "no_zone"]
    assert metrics["confusion_matrix"] == [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 0],
    ]
