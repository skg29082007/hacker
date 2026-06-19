#!/usr/bin/env python3
"""
Setup script: copies the provided CSV files into the expected data/ directory structure.
Run this once after cloning the repo, if the data files are not already in place.

Alternatively, pass --claims, --history, --evidence-reqs explicitly to agent.py.
"""

import os
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(description="Set up data directory structure")
    parser.add_argument("--claims", help="Path to claims CSV")
    parser.add_argument("--history", help="Path to user_history CSV")
    parser.add_argument("--evidence-reqs", help="Path to evidence_requirements CSV")
    parser.add_argument("--data-root", default=".", help="Destination root directory")
    args = parser.parse_args()

    root = args.data_root
    os.makedirs(os.path.join(root, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(root, "data"), exist_ok=True)

    if args.claims:
        dest = os.path.join(root, "dataset", "test.csv")
        shutil.copy2(args.claims, dest)
        print(f"Copied claims -> {dest}")

    if args.history:
        dest = os.path.join(root, "data", "user_history.csv")
        shutil.copy2(args.history, dest)
        print(f"Copied history -> {dest}")

    if args.evidence_reqs:
        dest = os.path.join(root, "data", "evidence_requirements.csv")
        shutil.copy2(args.evidence_reqs, dest)
        print(f"Copied evidence_requirements -> {dest}")


if __name__ == "__main__":
    main()
