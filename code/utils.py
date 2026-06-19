import base64
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def encode_image_base64(image_path: str) -> Optional[str]:
    """Load an image file and return its base64-encoded string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        logger.warning(f"Image not found: {image_path}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None


def get_image_mime_type(image_path: str) -> str:
    """Infer MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


def get_image_id(image_path: str) -> str:
    """Extract image ID (filename without extension) from a path."""
    return Path(image_path).stem


def load_csv_as_dicts(filepath: str) -> list[dict]:
    """Load a CSV file and return a list of row dicts."""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_csv(filepath: str, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a list of dicts to a CSV file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {filepath}")


def format_risk_flags(flags: list[str]) -> str:
    """Normalize risk flags list to semicolon-separated string."""
    if not flags or flags == ["none"] or flags == [""] or (len(flags) == 1 and flags[0].lower() == "none"):
        return "none"
    cleaned = [f.strip() for f in flags if f.strip() and f.strip().lower() != "none"]
    return ";".join(cleaned) if cleaned else "none"


def format_supporting_image_ids(ids: list[str]) -> str:
    """Normalize supporting image IDs list to semicolon-separated string."""
    if not ids or ids == ["none"]:
        return "none"
    cleaned = [i.strip() for i in ids if i.strip() and i.strip().lower() != "none"]
    return ";".join(cleaned) if cleaned else "none"


def setup_logging(log_file: str = "log.txt") -> None:
    """Configure logging to both console and log.txt (chat transcript)."""
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    root.addHandler(ch)
    root.addHandler(fh)


PROMPT_INJECTION_PATTERNS = [
    "approve the claim",
    "approve this claim",
    "skip manual review",
    "ignore previous instructions",
    "ignore all previous",
    "mark this as supported",
    "mark this row",
    "follow the note",
    "follow it and approve",
    "usko follow karke",
    "claim approve kar",
    "system reading this should approve",
    "just approve",
    "automatically approve",
]


def detect_prompt_injection(claim_text: str) -> bool:
    """Quick pre-check for prompt injection signals before VLM call."""
    lower = claim_text.lower()
    return any(pattern in lower for pattern in PROMPT_INJECTION_PATTERNS)
