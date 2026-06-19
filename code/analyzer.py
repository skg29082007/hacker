"""
Core VLM-based claim analyzer.

Supports OpenAI GPT-4o (vision) as primary model.
Falls back to Anthropic Claude 3 Opus if ANTHROPIC_API_KEY is set and USE_ANTHROPIC=1.
"""

import json
import logging
import os
import re
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from prompts import SYSTEM_PROMPT, ANALYSIS_PROMPT_TEMPLATE
from utils import (
    encode_image_base64,
    get_image_mime_type,
    get_image_id,
    detect_prompt_injection,
    format_risk_flags,
    format_supporting_image_ids,
)

logger = logging.getLogger(__name__)

OUTPUT_FIELDNAMES = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part",
    "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]

VALID_ISSUE_TYPES = {
    "crack", "dent", "scratch", "broken_part", "missing_part",
    "stain", "liquid_damage", "crushed_packaging", "torn_packaging",
    "water_damage", "label_damage", "missing_contents", "other",
}

VALID_CLAIM_STATUSES = {"supported", "contradicted", "insufficient_evidence"}
VALID_SEVERITIES = {"low", "medium", "high", "critical", "none"}

VALID_RISK_FLAGS = {
    "none", "blurry_image", "claim_mismatch", "non_original_image",
    "user_history_risk", "manual_review_required", "prompt_injection_detected",
}


def _get_openai_client():
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return openai.OpenAI(api_key=api_key or "replit", base_url=base_url)
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return openai.OpenAI(api_key=api_key)


def _get_anthropic_client():
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
    return anthropic.Anthropic(api_key=api_key)


def _build_user_history_text(user_id: str, history_lookup: dict[str, dict]) -> str:
    h = history_lookup.get(user_id)
    if not h:
        return "No history found for this user."
    lines = [
        f"- Past claims: {h.get('past_claim_count', 0)} total",
        f"- Accepted: {h.get('accept_claim', 0)}, Manual review: {h.get('manual_review_claim', 0)}, Rejected: {h.get('rejected_claim', 0)}",
        f"- Claims in last 90 days: {h.get('last_90_days_claim_count', 0)}",
        f"- History flags: {h.get('history_flags', 'none')}",
        f"- Summary: {h.get('history_summary', 'N/A')}",
    ]
    return "\n".join(lines)


def _build_evidence_requirements_text(claim_object: str, evidence_reqs: list[dict]) -> str:
    relevant = [
        r for r in evidence_reqs
        if r.get("claim_object", "").lower() in (claim_object.lower(), "all")
    ]
    if not relevant:
        return "No specific requirements found."
    lines = []
    for r in relevant:
        lines.append(f"[{r['requirement_id']}] {r['applies_to']}: {r['minimum_image_evidence']}")
    return "\n".join(lines)


def _parse_json_response(raw: str) -> dict:
    """Extract JSON from model response, handling markdown fences."""
    raw = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        raw = fence_match.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            return json.loads(json_match.group(0))
        raise


def _normalize_result(result: dict, claim_row: dict, injection_detected: bool) -> dict:
    """Normalize and validate all fields, apply injection flag, format for CSV."""
    risk_flags = result.get("risk_flags", ["none"])
    if isinstance(risk_flags, str):
        risk_flags = [f.strip() for f in risk_flags.split(";")]

    if injection_detected and "prompt_injection_detected" not in risk_flags:
        risk_flags = [f for f in risk_flags if f.lower() != "none"]
        risk_flags.append("prompt_injection_detected")

    user_history_flags = ""
    h = claim_row.get("_user_history_flags", "")
    if h and h != "none":
        for hf in h.split(";"):
            hf = hf.strip()
            if hf and hf in VALID_RISK_FLAGS and hf not in risk_flags:
                risk_flags.append(hf)

    issue_type = result.get("issue_type", "other")
    if issue_type not in VALID_ISSUE_TYPES:
        issue_type = "other"

    claim_status = result.get("claim_status", "insufficient_evidence")
    if claim_status not in VALID_CLAIM_STATUSES:
        claim_status = "insufficient_evidence"

    severity = result.get("severity", "none")
    if severity not in VALID_SEVERITIES:
        severity = "none"

    supporting_ids = result.get("supporting_image_ids", ["none"])
    if isinstance(supporting_ids, str):
        supporting_ids = [s.strip() for s in supporting_ids.split(";")]

    evidence_met = result.get("evidence_standard_met", False)
    if isinstance(evidence_met, str):
        evidence_met = evidence_met.lower() == "true"

    valid_image = result.get("valid_image", True)
    if isinstance(valid_image, str):
        valid_image = valid_image.lower() == "true"

    return {
        "user_id": claim_row["user_id"],
        "image_paths": claim_row["image_paths"],
        "user_claim": claim_row["user_claim"],
        "claim_object": claim_row["claim_object"],
        "evidence_standard_met": str(evidence_met).lower(),
        "evidence_standard_met_reason": result.get("evidence_standard_met_reason", ""),
        "risk_flags": format_risk_flags(risk_flags),
        "issue_type": issue_type,
        "object_part": result.get("object_part", "unknown"),
        "claim_status": claim_status,
        "claim_status_justification": result.get("claim_status_justification", ""),
        "supporting_image_ids": format_supporting_image_ids(supporting_ids),
        "valid_image": str(valid_image).lower(),
        "severity": severity,
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(Exception),
)
def _call_openai(client, model: str, system_prompt: str, user_text: str, image_items: list[dict]) -> str:
    content = []
    for item in image_items:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{item['mime']};base64,{item['data']}",
                "detail": "high",
            },
        })
    content.append({"type": "text", "text": user_text})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=1024,
        temperature=0.1,
    )
    return response.choices[0].message.content


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(Exception),
)
def _call_anthropic(client, model: str, system_prompt: str, user_text: str, image_items: list[dict]) -> str:
    content = []
    for item in image_items:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": item["mime"],
                "data": item["data"],
            },
        })
    content.append({"type": "text", "text": user_text})

    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
    )
    return response.content[0].text


class ClaimAnalyzer:
    def __init__(
        self,
        data_root: str = ".",
        use_anthropic: bool = False,
        openai_model: str = "gpt-4o",
        anthropic_model: str = "claude-3-5-sonnet-20241022",
    ):
        self.data_root = data_root
        self.use_anthropic = use_anthropic
        self.openai_model = openai_model
        self.anthropic_model = anthropic_model

        if use_anthropic:
            self.client = _get_anthropic_client()
            logger.info(f"Using Anthropic model: {anthropic_model}")
        else:
            self.client = _get_openai_client()
            logger.info(f"Using OpenAI model: {openai_model}")

    def analyze_claim(
        self,
        claim_row: dict,
        history_lookup: dict[str, dict],
        evidence_reqs: list[dict],
    ) -> dict:
        user_id = claim_row["user_id"]
        image_paths_str = claim_row["image_paths"]
        user_claim = claim_row["user_claim"]
        claim_object = claim_row["claim_object"]

        image_paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]

        injection_detected = detect_prompt_injection(user_claim)
        if injection_detected:
            logger.warning(f"[{user_id}] Prompt injection detected in claim text.")

        image_items = []
        missing_images = []
        for path in image_paths:
            full_path = os.path.join(self.data_root, path)
            data = encode_image_base64(full_path)
            if data:
                image_items.append({
                    "id": get_image_id(path),
                    "path": path,
                    "mime": get_image_mime_type(path),
                    "data": data,
                })
            else:
                missing_images.append(path)

        if missing_images:
            logger.warning(f"[{user_id}] Missing images: {missing_images}")

        if not image_items:
            logger.error(f"[{user_id}] No images could be loaded. Returning insufficient_evidence.")
            return self._fallback_result(claim_row, injection_detected, "No images could be loaded.")

        history_text = _build_user_history_text(user_id, history_lookup)
        evidence_text = _build_evidence_requirements_text(claim_object, evidence_reqs)

        image_id_list = [item["id"] for item in image_items]
        user_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            user_id=user_id,
            claim_object=claim_object,
            image_paths=image_paths_str,
            user_claim=user_claim,
            user_history=history_text,
            evidence_requirements=evidence_text,
            num_images=len(image_items),
        )
        user_prompt += f"\n\nNote: The images are provided above, identified as: {', '.join(image_id_list)}."

        logger.info(f"[{user_id}] Sending {len(image_items)} image(s) to VLM for claim_object={claim_object}")

        try:
            if self.use_anthropic:
                raw_response = _call_anthropic(
                    self.client, self.anthropic_model, SYSTEM_PROMPT, user_prompt, image_items
                )
            else:
                raw_response = _call_openai(
                    self.client, self.openai_model, SYSTEM_PROMPT, user_prompt, image_items
                )

            logger.debug(f"[{user_id}] Raw VLM response:\n{raw_response}")

            result_dict = _parse_json_response(raw_response)

            user_hist = history_lookup.get(user_id, {})
            claim_row["_user_history_flags"] = user_hist.get("history_flags", "none")

            normalized = _normalize_result(result_dict, claim_row, injection_detected)
            logger.info(
                f"[{user_id}] Result: status={normalized['claim_status']}, "
                f"severity={normalized['severity']}, flags={normalized['risk_flags']}"
            )
            return normalized

        except Exception as e:
            logger.error(f"[{user_id}] VLM call failed: {e}", exc_info=True)
            return self._fallback_result(claim_row, injection_detected, str(e))

    def _fallback_result(self, claim_row: dict, injection_detected: bool, reason: str) -> dict:
        flags = ["prompt_injection_detected"] if injection_detected else ["manual_review_required"]
        user_hist = claim_row.get("_user_history_flags", "")
        if user_hist and user_hist != "none":
            for hf in user_hist.split(";"):
                hf = hf.strip()
                if hf and hf in VALID_RISK_FLAGS and hf not in flags:
                    flags.append(hf)
        return {
            "user_id": claim_row["user_id"],
            "image_paths": claim_row["image_paths"],
            "user_claim": claim_row["user_claim"],
            "claim_object": claim_row["claim_object"],
            "evidence_standard_met": "false",
            "evidence_standard_met_reason": f"Processing error: {reason}",
            "risk_flags": format_risk_flags(flags),
            "issue_type": "other",
            "object_part": "unknown",
            "claim_status": "insufficient_evidence",
            "claim_status_justification": f"Unable to process claim due to error: {reason}",
            "supporting_image_ids": "none",
            "valid_image": "false",
            "severity": "none",
        }
