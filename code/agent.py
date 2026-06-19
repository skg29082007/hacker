#!/usr/bin/env python3
"""
Multi-Modal Evidence Review Agent
===================================
Verifies damage claims (car, laptop, package) using images, claim conversation,
user history, and evidence requirements.

Usage:
    python agent.py [--claims CLAIMS_CSV] [--output OUTPUT_CSV] [--data-root DATA_ROOT]
                    [--log LOG_FILE] [--use-anthropic] [--model MODEL]

Environment variables:
    OPENAI_API_KEY     — required unless USE_ANTHROPIC=1
    ANTHROPIC_API_KEY  — required if --use-anthropic flag is set
    OPENAI_BASE_URL    — optional, override OpenAI API base URL (e.g. for Replit proxy)
    USE_ANTHROPIC      — set to "1" to use Anthropic Claude instead of OpenAI

Paths (relative to --data-root, default "."):
    dataset/test.csv           — claims input
    data/user_history.csv      — user history
    data/evidence_requirements.csv — evidence requirements
    images/                    — image files
    output.csv                 — predictions output
    log.txt                    — chat transcript log
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from utils import load_csv_as_dicts, write_csv, setup_logging
from analyzer import ClaimAnalyzer, OUTPUT_FIELDNAMES

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review Agent")
    parser.add_argument(
        "--claims",
        default=None,
        help="Path to claims CSV (default: dataset/test.csv relative to data-root)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for output CSV (default: output.csv relative to data-root)",
    )
    parser.add_argument(
        "--history",
        default=None,
        help="Path to user history CSV (default: data/user_history.csv relative to data-root)",
    )
    parser.add_argument(
        "--evidence-reqs",
        default=None,
        help="Path to evidence requirements CSV (default: data/evidence_requirements.csv relative to data-root)",
    )
    parser.add_argument(
        "--data-root",
        default=".",
        help="Root directory where images/ and CSVs live (default: current directory)",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Path to log/transcript file (default: log.txt relative to data-root)",
    )
    parser.add_argument(
        "--use-anthropic",
        action="store_true",
        default=os.environ.get("USE_ANTHROPIC", "0") == "1",
        help="Use Anthropic Claude instead of OpenAI (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name (OpenAI: gpt-4o, gpt-4-turbo | Anthropic: claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process the first N claims (for testing)",
    )
    return parser.parse_args()


def resolve_path(data_root: str, default_relative: str, override: str | None) -> str:
    if override:
        return override
    return os.path.join(data_root, default_relative)


def main():
    args = parse_args()

    data_root = os.path.abspath(args.data_root)

    log_path = resolve_path(data_root, "log.txt", args.log)
    setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("Multi-Modal Evidence Review Agent starting")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Log file:  {log_path}")

    claims_path = resolve_path(data_root, "dataset/test.csv", args.claims)
    history_path = resolve_path(data_root, "data/user_history.csv", args.history)
    evidence_path = resolve_path(data_root, "data/evidence_requirements.csv", args.evidence_reqs)
    output_path = resolve_path(data_root, "output.csv", args.output)

    logger.info(f"Claims:    {claims_path}")
    logger.info(f"History:   {history_path}")
    logger.info(f"Evidence:  {evidence_path}")
    logger.info(f"Output:    {output_path}")

    if not os.path.exists(claims_path):
        logger.error(f"Claims file not found: {claims_path}")
        sys.exit(1)
    if not os.path.exists(history_path):
        logger.error(f"User history file not found: {history_path}")
        sys.exit(1)
    if not os.path.exists(evidence_path):
        logger.error(f"Evidence requirements file not found: {evidence_path}")
        sys.exit(1)

    claims = load_csv_as_dicts(claims_path)
    history_rows = load_csv_as_dicts(history_path)
    evidence_reqs = load_csv_as_dicts(evidence_path)

    history_lookup = {row["user_id"]: row for row in history_rows}

    logger.info(f"Loaded {len(claims)} claims, {len(history_rows)} history records, {len(evidence_reqs)} evidence requirements")

    if args.sample:
        claims = claims[:args.sample]
        logger.info(f"Sample mode: processing {len(claims)} claim(s)")

    model_kwargs = {}
    if args.model:
        if args.use_anthropic:
            model_kwargs["anthropic_model"] = args.model
        else:
            model_kwargs["openai_model"] = args.model

    analyzer = ClaimAnalyzer(
        data_root=data_root,
        use_anthropic=args.use_anthropic,
        **model_kwargs,
    )

    results = []
    errors = 0

    for claim in tqdm(claims, desc="Analyzing claims", unit="claim"):
        try:
            result = analyzer.analyze_claim(claim, history_lookup, evidence_reqs)
            results.append(result)
        except Exception as e:
            logger.error(f"Unexpected error for {claim.get('user_id', '?')}: {e}", exc_info=True)
            errors += 1
            results.append(analyzer._fallback_result(claim, False, str(e)))

    write_csv(output_path, results, OUTPUT_FIELDNAMES)

    logger.info("=" * 60)
    logger.info(f"Done. Processed {len(results)} claims. Errors: {errors}.")
    logger.info(f"Output written to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
