# Multi-Modal Evidence Review Agent

A VLM-powered damage claim verification system for car, laptop, and package claims.

## Architecture

```
code/
├── agent.py          — Main orchestration script (entry point)
├── analyzer.py       — VLM call logic, result normalization
├── prompts.py        — System prompt + analysis prompt templates
├── utils.py          — Image encoding, CSV I/O, logging, injection detection
├── evaluate.py       — Evaluation workflow (compare predictions vs ground truth)
├── setup_data.py     — Helper to copy CSVs into expected directory structure
└── requirements.txt  — Python dependencies
```

## Approach

1. **Claim parsing** — Extracts the core damage claim from multi-language conversations (English, Hindi, Spanish, Chinese, mixed).
2. **Vision analysis** — Sends all images (base64) + structured prompt to GPT-4o or Claude 3.5 Sonnet.
3. **Evidence requirements** — Injects the relevant `evidence_requirements.csv` entries into the prompt.
4. **User history** — Provides history as risk context (does not override visual evidence).
5. **Prompt injection detection** — Pre-screens claim text + instructs the model to flag manipulation attempts.
6. **Structured output** — Model returns JSON; the agent normalizes and validates every field.

## Setup

```bash
pip install -r requirements.txt
```

Set your API key:
```bash
export OPENAI_API_KEY="sk-..."        # for GPT-4o (default)
# OR
export ANTHROPIC_API_KEY="sk-ant-..." # for Claude
```

## Running on the test dataset

The repo is expected to have this structure:
```
<repo-root>/
├── dataset/
│   └── test.csv          — claims to evaluate
├── data/
│   ├── user_history.csv
│   └── evidence_requirements.csv
├── images/
│   └── test/
│       └── case_XXX/
│           └── img_X.jpg
└── code/                 — this directory
```

Run from the **repo root**:
```bash
cd <repo-root>
python code/agent.py
```

This writes `output.csv` and `log.txt` to the repo root.

### Custom paths

```bash
python code/agent.py \
  --claims dataset/test.csv \
  --history data/user_history.csv \
  --evidence-reqs data/evidence_requirements.csv \
  --output output.csv \
  --log log.txt \
  --data-root .
```

### Use Anthropic Claude instead of OpenAI

```bash
python code/agent.py --use-anthropic --model claude-3-5-sonnet-20241022
```

### Quick test (first 3 claims only)

```bash
python code/agent.py --sample 3
```

## Evaluation workflow

Compare agent output against sample claims with known ground truth:

```bash
python code/evaluate.py \
  --ground-truth data/sample_claims.csv \
  --predictions output.csv
```

Reports per-field accuracy for: `evidence_standard_met`, `issue_type`, `object_part`, `claim_status`, `severity`, `valid_image`, and Jaccard similarity for `risk_flags`.

## Output schema

```
user_id, image_paths, user_claim, claim_object,
evidence_standard_met, evidence_standard_met_reason,
risk_flags, issue_type, object_part,
claim_status, claim_status_justification,
supporting_image_ids, valid_image, severity
```

## Risk flags produced

| Flag | Triggered when |
|---|---|
| `none` | No issues detected |
| `blurry_image` | Image quality is too poor to evaluate |
| `claim_mismatch` | Images show different damage/object than claimed |
| `non_original_image` | Images appear to be screenshots or stock photos |
| `user_history_risk` | User has flagged history |
| `manual_review_required` | Ambiguous evidence needs human review |
| `prompt_injection_detected` | Claim text contains instructions to manipulate output |

## Chat transcript logging

All interactions are logged to `log.txt` (configurable via `--log`). This includes:
- Each claim processed with its user_id
- VLM raw responses (DEBUG level)
- Final decision per claim
- Errors and warnings
