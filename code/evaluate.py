#!/usr/bin/env python3
"""
Evaluation workflow for the Multi-Modal Evidence Review Agent.

Compares agent output against a ground-truth CSV (sample_claims with expected outputs).
Reports per-field accuracy and overall metrics.

Usage:
    python evaluate.py --ground-truth sample_claims.csv --predictions output.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

EVAL_FIELDS = [
    "evidence_standard_met",
    "issue_type",
    "object_part",
    "claim_status",
    "severity",
    "valid_image",
]

RISK_FLAG_FIELDS = ["risk_flags"]


def load_csv(path: str) -> dict[tuple, dict]:
    """Load CSV, keyed by (user_id, image_paths)."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["user_id"].strip(), row["image_paths"].strip())
            rows[key] = {k: v.strip() for k, v in row.items()}
    return rows


def normalize_flags(flags_str: str) -> set[str]:
    if not flags_str or flags_str.lower() == "none":
        return {"none"}
    return {f.strip() for f in flags_str.split(";")}


def evaluate(ground_truth_path: str, predictions_path: str) -> dict:
    gt = load_csv(ground_truth_path)
    pred = load_csv(predictions_path)

    matched = 0
    unmatched_keys = []
    field_scores: dict[str, list[bool]] = {f: [] for f in EVAL_FIELDS}
    risk_flag_scores: list[float] = []

    for key, gt_row in gt.items():
        if key not in pred:
            unmatched_keys.append(key)
            logger.warning(f"Missing prediction for: {key}")
            for f in EVAL_FIELDS:
                field_scores[f].append(False)
            risk_flag_scores.append(0.0)
            continue

        matched += 1
        pred_row = pred[key]

        for f in EVAL_FIELDS:
            gt_val = gt_row.get(f, "").lower().strip()
            pred_val = pred_row.get(f, "").lower().strip()
            field_scores[f].append(gt_val == pred_val)

        gt_flags = normalize_flags(gt_row.get("risk_flags", "none"))
        pred_flags = normalize_flags(pred_row.get("risk_flags", "none"))
        if gt_flags or pred_flags:
            intersection = gt_flags & pred_flags
            union = gt_flags | pred_flags
            jaccard = len(intersection) / len(union) if union else 1.0
        else:
            jaccard = 1.0
        risk_flag_scores.append(jaccard)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Ground truth rows: {len(gt)}")
    print(f"Predictions matched: {matched}")
    if unmatched_keys:
        print(f"Unmatched keys: {len(unmatched_keys)}")

    print("\nPer-field accuracy:")
    overall_correct = 0
    overall_total = 0
    for f in EVAL_FIELDS:
        scores = field_scores[f]
        if scores:
            acc = sum(scores) / len(scores)
            overall_correct += sum(scores)
            overall_total += len(scores)
            status = "✓" if acc >= 0.7 else "✗"
            print(f"  {status} {f:<30} {acc:.1%}  ({sum(scores)}/{len(scores)})")

    if risk_flag_scores:
        avg_jaccard = sum(risk_flag_scores) / len(risk_flag_scores)
        print(f"  ~ risk_flags (Jaccard avg)         {avg_jaccard:.1%}")

    if overall_total:
        overall_acc = overall_correct / overall_total
        print(f"\nOverall field accuracy: {overall_acc:.1%}")

    print("=" * 60 + "\n")

    return {
        "matched": matched,
        "field_scores": {f: (sum(s) / len(s) if s else 0.0) for f, s in field_scores.items()},
        "risk_flag_jaccard": sum(risk_flag_scores) / len(risk_flag_scores) if risk_flag_scores else 0.0,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate agent predictions against ground truth")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth CSV (sample_claims.csv)")
    parser.add_argument("--predictions", required=True, help="Path to agent predictions CSV (output.csv)")
    args = parser.parse_args()

    if not Path(args.ground_truth).exists():
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        sys.exit(1)
    if not Path(args.predictions).exists():
        logger.error(f"Predictions file not found: {args.predictions}")
        sys.exit(1)

    evaluate(args.ground_truth, args.predictions)


if __name__ == "__main__":
    main()
