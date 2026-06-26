"""Evaluate KGO pipelines on a Datumaro/CVAT-style dataset.

The script intentionally keeps metric calculations independent from sklearn so
unit tests and CI can exercise the logic without installing ML dependencies.
Inference and plotting dependencies are imported lazily by the CLI path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRUE_LABELS = ("kgo_empty", "kgo_full")
CONFUSION_LABELS = ("kgo_empty", "kgo_full", "no_zone")
DEFAULT_PIPELINES = ("new_classifier_nodino", "sam", "smp")


@dataclass(frozen=True)
class EvaluationSample:
    dataset: str
    image: str
    image_path: Path
    true_label: str


@dataclass(frozen=True)
class DatasetLoadResult:
    samples: list[EvaluationSample]
    total_items: int
    unlabeled_items: int
    multi_label_items: int
    label_counts: dict[str, int]


@dataclass(frozen=True)
class PredictionRow:
    dataset: str
    image: str
    true_label: str
    pred_label: str
    correct: bool
    error: str = ""


def normalize_prediction(label: str | None) -> str:
    label = (label or "").strip()
    return label if label in TRUE_LABELS else "no_zone"


def load_datumaro_samples(dataset_dir: Path, dataset_name: str | None = None) -> DatasetLoadResult:
    dataset_name = dataset_name or dataset_dir.name
    annotation_path = dataset_dir / "annotations" / "default.json"
    with annotation_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    labels = [item["name"] for item in payload["categories"]["label"]["labels"]]
    samples: list[EvaluationSample] = []
    label_counts: dict[str, int] = {label: 0 for label in TRUE_LABELS}
    unlabeled_items = 0
    multi_label_items = 0

    for item in payload.get("items", []):
        item_labels = [
            labels[annotation["label_id"]]
            for annotation in item.get("annotations", [])
            if annotation.get("type") == "label" and labels[annotation["label_id"]] in TRUE_LABELS
        ]

        if not item_labels:
            unlabeled_items += 1
            continue

        if len(item_labels) > 1:
            multi_label_items += 1

        true_label = item_labels[0]
        image_name = item["image"]["path"]
        image_path = dataset_dir / "images" / "default" / image_name
        samples.append(
            EvaluationSample(
                dataset=dataset_name,
                image=image_name,
                image_path=image_path,
                true_label=true_label,
            )
        )
        label_counts[true_label] += 1

    return DatasetLoadResult(
        samples=samples,
        total_items=len(payload.get("items", [])),
        unlabeled_items=unlabeled_items,
        multi_label_items=multi_label_items,
        label_counts=label_counts,
    )


def merge_dataset_results(results: list[DatasetLoadResult]) -> DatasetLoadResult:
    samples: list[EvaluationSample] = []
    label_counts = {label: 0 for label in TRUE_LABELS}
    total_items = 0
    unlabeled_items = 0
    multi_label_items = 0

    for result in results:
        samples.extend(result.samples)
        total_items += result.total_items
        unlabeled_items += result.unlabeled_items
        multi_label_items += result.multi_label_items
        for label in TRUE_LABELS:
            label_counts[label] += result.label_counts.get(label, 0)

    return DatasetLoadResult(
        samples=samples,
        total_items=total_items,
        unlabeled_items=unlabeled_items,
        multi_label_items=multi_label_items,
        label_counts=label_counts,
    )


def build_confusion_matrix(rows: list[PredictionRow], labels: tuple[str, ...] = CONFUSION_LABELS) -> list[list[int]]:
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for row in rows:
        true_label = row.true_label if row.true_label in index else "no_zone"
        pred_label = row.pred_label if row.pred_label in index else "no_zone"
        matrix[index[true_label]][index[pred_label]] += 1

    return matrix


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def build_classification_metrics(rows: list[PredictionRow]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.correct)
    no_zone_count = sum(1 for row in rows if row.pred_label == "no_zone")
    class_metrics: dict[str, dict[str, float | int]] = {}

    for label in TRUE_LABELS:
        tp = sum(1 for row in rows if row.true_label == label and row.pred_label == label)
        fp = sum(1 for row in rows if row.true_label != label and row.pred_label == label)
        fn = sum(1 for row in rows if row.true_label == label and row.pred_label != label)
        support = sum(1 for row in rows if row.true_label == label)

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_precision = sum(float(class_metrics[label]["precision"]) for label in TRUE_LABELS) / len(TRUE_LABELS)
    macro_recall = sum(float(class_metrics[label]["recall"]) for label in TRUE_LABELS) / len(TRUE_LABELS)
    macro_f1 = sum(float(class_metrics[label]["f1"]) for label in TRUE_LABELS) / len(TRUE_LABELS)

    support_total = sum(int(class_metrics[label]["support"]) for label in TRUE_LABELS)
    weighted_precision = safe_divide(
        sum(float(class_metrics[label]["precision"]) * int(class_metrics[label]["support"]) for label in TRUE_LABELS),
        support_total,
    )
    weighted_recall = safe_divide(
        sum(float(class_metrics[label]["recall"]) * int(class_metrics[label]["support"]) for label in TRUE_LABELS),
        support_total,
    )
    weighted_f1 = safe_divide(
        sum(float(class_metrics[label]["f1"]) * int(class_metrics[label]["support"]) for label in TRUE_LABELS),
        support_total,
    )

    return {
        "total": total,
        "accuracy": safe_divide(correct, total),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "support": {label: int(class_metrics[label]["support"]) for label in TRUE_LABELS},
        "no_zone_count": no_zone_count,
        "inference_error_count": sum(1 for row in rows if row.error),
        "per_class": class_metrics,
        "confusion_labels": list(CONFUSION_LABELS),
        "confusion_matrix": build_confusion_matrix(rows),
    }


def evaluate_pipeline(pipeline_name: str, samples: list[EvaluationSample]) -> list[PredictionRow]:
    from fastapi_app.inference import run_inference

    rows: list[PredictionRow] = []
    total = len(samples)

    for index, sample in enumerate(samples, start=1):
        print(f"[{pipeline_name}] {index}/{total}: {sample.image}", flush=True)
        error = ""
        try:
            result = run_inference(sample.image_path.read_bytes(), pipeline_name)
            pred_label = normalize_prediction(result.get("label"))
        except Exception as exc:  # noqa: BLE001 - evaluation should finish and report failed images.
            pred_label = "no_zone"
            error = str(exc)

        rows.append(
            PredictionRow(
                dataset=sample.dataset,
                image=sample.image,
                true_label=sample.true_label,
                pred_label=pred_label,
                correct=pred_label == sample.true_label,
                error=error,
            )
        )

    return rows


def write_predictions_csv(path: Path, rows: list[PredictionRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("dataset", "image", "true_label", "pred_label", "correct"))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "image": row.image,
                    "true_label": row.true_label,
                    "pred_label": row.pred_label,
                    "correct": int(row.correct),
                }
            )


def write_errors_csv(path: Path, rows: list[PredictionRow]) -> None:
    error_rows = [row for row in rows if row.error]
    if not error_rows:
        return

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("dataset", "image", "true_label", "pred_label", "error"))
        writer.writeheader()
        for row in error_rows:
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "image": row.image,
                    "true_label": row.true_label,
                    "pred_label": row.pred_label,
                    "error": row.error,
                }
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_confusion_matrix_plot(path: Path, metrics: dict[str, Any], title: str) -> None:
    import matplotlib.pyplot as plt

    labels = metrics["confusion_labels"]
    matrix = metrics["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_class_metrics_plot(path: Path, metrics: dict[str, Any], title: str) -> None:
    import matplotlib.pyplot as plt

    metric_names = ("precision", "recall", "f1")
    x_positions = range(len(TRUE_LABELS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, metric_name in enumerate(metric_names):
        values = [metrics["per_class"][label][metric_name] for label in TRUE_LABELS]
        shifted = [position + (offset - 1) * width for position in x_positions]
        ax.bar(shifted, values, width=width, label=metric_name)

    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xticks(list(x_positions), TRUE_LABELS)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        "pipeline",
        "total",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "no_zone_count",
        "inference_error_count",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def format_score(value: float) -> str:
    return f"{value:.4f}"


def write_summary_md(path: Path, rows: list[dict[str, Any]], dataset_stats: DatasetLoadResult) -> None:
    lines = [
        "# Evaluation Summary",
        "",
        f"- Total items in annotations: {dataset_stats.total_items}",
        f"- Evaluated labeled images: {len(dataset_stats.samples)}",
        f"- Unlabeled items skipped: {dataset_stats.unlabeled_items}",
        f"- Multi-label items: {dataset_stats.multi_label_items}",
        f"- Label support: {dataset_stats.label_counts}",
        "",
        "| Pipeline | Total | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | No Zone | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| {pipeline} | {total} | {accuracy} | {macro_precision} | {macro_recall} | {macro_f1} | "
            "{weighted_f1} | {no_zone_count} | {inference_error_count} |".format(
                pipeline=row["pipeline"],
                total=row["total"],
                accuracy=format_score(row["accuracy"]),
                macro_precision=format_score(row["macro_precision"]),
                macro_recall=format_score(row["macro_recall"]),
                macro_f1=format_score(row["macro_f1"]),
                weighted_f1=format_score(row["weighted_f1"]),
                no_zone_count=row["no_zone_count"],
                inference_error_count=row["inference_error_count"],
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KGO pipelines and save metrics with plots.")
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        default=[Path("datasets/42")],
        help="One or more dataset roots with annotations/default.json.",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        default=None,
        help="Alias for --dataset when passing multiple dataset roots.",
    )
    parser.add_argument("--pipelines", nargs="+", default=list(DEFAULT_PIPELINES), help="Pipeline names to evaluate.")
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation"), help="Output directory for reports.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N labeled samples.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dirs = args.datasets or args.dataset
    dataset_stats = merge_dataset_results(
        [load_datumaro_samples(dataset_dir, dataset_name=str(dataset_dir)) for dataset_dir in dataset_dirs]
    )
    samples = dataset_stats.samples[: args.limit] if args.limit else dataset_stats.samples

    if not samples:
        raise RuntimeError(f"No labeled samples found in: {', '.join(str(dataset_dir) for dataset_dir in dataset_dirs)}")

    args.output.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for pipeline_name in args.pipelines:
        rows = evaluate_pipeline(pipeline_name, samples)
        metrics = build_classification_metrics(rows)
        metrics_with_meta = {
            "pipeline": pipeline_name,
            "datasets": [str(dataset_dir) for dataset_dir in dataset_dirs],
            "total_annotation_items": dataset_stats.total_items,
            "evaluated_labeled_images": len(samples),
            "unlabeled_items_skipped": dataset_stats.unlabeled_items,
            "multi_label_items": dataset_stats.multi_label_items,
            **metrics,
        }

        write_json(args.output / f"metrics_{pipeline_name}.json", metrics_with_meta)
        write_predictions_csv(args.output / f"predictions_{pipeline_name}.csv", rows)
        write_errors_csv(args.output / f"errors_{pipeline_name}.csv", rows)

        if not args.no_plots:
            save_confusion_matrix_plot(
                args.output / f"confusion_matrix_{pipeline_name}.png",
                metrics,
                f"Confusion Matrix: {pipeline_name}",
            )
            save_class_metrics_plot(
                args.output / f"class_metrics_{pipeline_name}.png",
                metrics,
                f"Class Metrics: {pipeline_name}",
            )

        summary_rows.append({"pipeline": pipeline_name, **metrics})

    write_summary_csv(args.output / "summary.csv", summary_rows)
    write_summary_md(args.output / "summary.md", summary_rows, dataset_stats)
    print(f"Evaluation report saved to: {args.output}")


if __name__ == "__main__":
    main()
