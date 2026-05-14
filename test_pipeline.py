import os
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from pipeline import (
    load_models,
    process_image,
    process_and_visualize,
    DEFAULT_THRESHOLD,
)


SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
LABEL_FOLDERS = ['kgo_full', 'kgo_empty']


def collect_test_data(root_dir):
    data = []

    for label in LABEL_FOLDERS:
        folder = os.path.join(root_dir, label)

        if not os.path.isdir(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue

        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                data.append((os.path.join(folder, fname), label))

    return data


def calculate_metrics(y_true, y_pred):
    tp = tn = fp = fn = 0

    for gt, pred in zip(y_true, y_pred):
        if gt == 'kgo_full' and pred == 'kgo_full':
            tp += 1
        elif gt == 'kgo_empty' and pred == 'kgo_empty':
            tn += 1
        elif gt == 'kgo_empty' and pred == 'kgo_full':
            fp += 1
        elif gt == 'kgo_full' and pred == 'kgo_empty':
            fn += 1

    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total': total,
        'skipped': 0,
    }


def _print_metrics(metrics):
    print(f"Accuracy : {metrics['accuracy']  * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall   : {metrics['recall']    * 100:.2f}%")
    print(f"F1-score : {metrics['f1']        * 100:.2f}%")
    print()
    print(f"TP: {metrics['tp']}  TN: {metrics['tn']}  "
          f"FP: {metrics['fp']}  FN: {metrics['fn']}")
    if metrics.get('skipped', 0):
        print(f"Skipped: {metrics['skipped']}")


def run_single_test(image_path, threshold=DEFAULT_THRESHOLD, output_dir=None):
    print(f"IMAGE: {image_path}")
    print(f"THRESHOLD: {threshold:.2f}")

    result = process_and_visualize(
        image_path=image_path,
        output_dir=output_dir,
        show_full=True,
        show_crop=True,
        threshold=threshold,
    )

    print()
    print(f"LABEL      : {result['label']}")
    print(f"PERCENTAGE : {result['percentage']:.2f}%")
    print()

    return result

def run_mass_test(root_dir, threshold=DEFAULT_THRESHOLD,
                  save_errors_to=None, save_csv=None):
    data = collect_test_data(root_dir)

    if not data:
        print("No test images found.")
        return None

    print()
    print(f"IMAGES FOUND : {len(data)}")
    print(f"THRESHOLD    : {threshold:.2f}")
    print()

    if save_errors_to:
        os.makedirs(save_errors_to, exist_ok=True)

    y_true, y_pred = [], []
    errors = []
    skipped = 0
    total = len(data)

    for idx, (img_path, true_label) in enumerate(data, start=1):
        fname = os.path.basename(img_path)
        print(f"[{idx}/{total}] {fname}", end=' ', flush=True)

        try:
            pred_label = process_image(img_path, threshold=threshold)
        except Exception as e:
            print(f"FAILED: {e}")
            skipped += 1
            continue

        if pred_label == "":
            print("NO PLATFORM")
            skipped += 1
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)

        if pred_label == true_label:
            print("OK")
        else:
            print(f"ERROR (true={true_label}, pred={pred_label})")
            errors.append({'image': img_path, 'true': true_label, 'pred': pred_label})

            if save_errors_to:
                name = os.path.splitext(fname)[0]
                out_dir = os.path.join(
                    save_errors_to,
                    f"{name}_GT_{true_label}_PRED_{pred_label}"
                )
                try:
                    process_and_visualize(
                        image_path=img_path,
                        output_dir=out_dir,
                        show_full=False,
                        show_crop=False,
                        threshold=threshold,
                    )
                except Exception as vis_err:
                    print(f"  VIS ERROR: {vis_err}")

    metrics = calculate_metrics(y_true, y_pred)
    metrics['skipped'] = skipped
    _print_metrics(metrics)

    if errors:
        print()
        print("=" * 60)
        print("ERRORS")
        print("=" * 60)
        df = pd.DataFrame(errors)
        print(df.to_string(index=False))

        if save_csv:
            df.to_csv(save_csv, index=False)
            print(f"\nCSV SAVED: {save_csv}")
    else:
        print("\nNo errors.")

    print()
    return {'metrics': metrics, 'errors': errors}

def find_best_threshold(root_dir, start=0.30, end=0.90, step=0.02):
    thresholds = np.arange(start, end + 1e-9, step)

    print()
    print(f"THRESHOLD SEARCH  [{start:.2f} … {end:.2f}, step={step:.2f}]")

    best_threshold = None
    best_f1 = -1.0

    for thr in thresholds:
        thr = float(thr)
        print(f"\n--- threshold = {thr:.2f} ---")
        result = run_mass_test(
            root_dir=root_dir,
            threshold=thr,
            save_errors_to=None,
            save_csv=None,
        )
        if result is None:
            continue

        f1 = result['metrics']['f1']
        print(f"F1 = {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr

    print()
    print("BEST RESULT")

    if best_threshold is not None:
        print(f"BEST THRESHOLD: {best_threshold:.2f}")
        print(f"BEST F1       : {best_f1:.4f}")
    else:
        print("No valid results found.")

    return best_threshold


def main():
    parser = argparse.ArgumentParser(description='KGO Pipeline Testing')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # ---- single ----
    p_single = subparsers.add_parser('single', help='Run on one image')
    p_single.add_argument('image_path', type=str)
    p_single.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    p_single.add_argument('--output_dir', type=str, default=None)

    # ---- mass ----
    p_mass = subparsers.add_parser('mass', help='Run on a labeled dataset')
    p_mass.add_argument('root_dir', type=str)
    p_mass.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    p_mass.add_argument('--save_errors_to', type=str, default=None)
    p_mass.add_argument('--save_csv', type=str, default=None)

    # ---- search_threshold ----
    p_search = subparsers.add_parser('search_threshold',
                                     help='Grid-search best threshold')
    p_search.add_argument('root_dir', type=str)
    p_search.add_argument('--start', type=float, default=0.30)
    p_search.add_argument('--end',   type=float, default=0.90)
    p_search.add_argument('--step',  type=float, default=0.02)

    args = parser.parse_args()

    print()
    print("LOADING MODELS")
    load_models()

    if args.mode == 'single':
        run_single_test(
            image_path=args.image_path,
            threshold=args.threshold,
            output_dir=args.output_dir,
        )
    elif args.mode == 'mass':
        run_mass_test(
            root_dir=args.root_dir,
            threshold=args.threshold,
            save_errors_to=args.save_errors_to,
            save_csv=args.save_csv,
        )
    elif args.mode == 'search_threshold':
        find_best_threshold(
            root_dir=args.root_dir,
            start=args.start,
            end=args.end,
            step=args.step,
        )


if __name__ == '__main__':
    main()