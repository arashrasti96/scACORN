import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


_PAREN_MARKDOWN_LINK_RE = re.compile(r"\s*\(\[[^\]]+\]\((?:https?|ftp)://[^)]+\)\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?|ftp)://[^)]+\)")
_RAW_URL_RE = re.compile(r"(?:https?|ftp)://\S+")
_SPACE_RE = re.compile(r"\s+")

STRUCTURED_REASONING_QUESTION_TEXT = (
    "What is the cell type? Answer using the exact field order shown in the answer format. "
    "Keep each field concise and grounded in the cell sentence."
)

STRUCTURED_REASONING_ANSWER_FORMAT = "\n".join(
    [
        "LABEL: <cell type>",
        "EVIDENCE: <3-5 supporting markers or none>",
        "MISSING: <0-3 expected but absent markers or none>",
        "CONFLICTS: <0-2 contradictory markers or none>",
        "QUALITY: <0-3 confounders or none>",
        "FINAL: <cell type>",
    ]
)


def clean_reasoning_text(text: str | None) -> str:
    """Remove citation/link noise from rationale text while preserving content."""
    if not text:
        return ""

    cleaned = str(text)
    # Normalize non-breaking hyphen (U+2011) and other dash variants to ASCII hyphen
    cleaned = cleaned.replace("\u2011", "-").replace("\u2010", "-").replace("\u2013", "-").replace("\u2014", "-")
    cleaned = _PAREN_MARKDOWN_LINK_RE.sub("", cleaned)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _RAW_URL_RE.sub("", cleaned)
    cleaned = cleaned.replace("..", ".")
    cleaned = cleaned.replace(" ;", ";")
    cleaned = cleaned.replace(" .", ".")
    cleaned = _SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _clean_reasoning_value(value):
    if isinstance(value, str):
        return clean_reasoning_text(value)
    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            if isinstance(item, str):
                cleaned_item = clean_reasoning_text(item)
                if cleaned_item:
                    cleaned_items.append(cleaned_item)
            else:
                cleaned_items.append(item)
        return cleaned_items
    if isinstance(value, dict):
        return {key: _clean_reasoning_value(inner_value) for key, inner_value in value.items()}
    return value


def is_reasoning_dataset_sample(sample: dict) -> bool:
    metadata = sample.get("metadata") if isinstance(sample, dict) else None
    metadata_source = metadata.get("source") if isinstance(metadata, dict) else None
    return bool(
        isinstance(sample, dict)
        and (
            sample.get("structured_output")
            or sample.get("textual_ground_truth")
            or metadata_source == "reasoning_dataset"
        )
    )


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if not value:
            continue
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(value)
    return result


def _clean_marker_item(value: str | None) -> str:
    cleaned = clean_reasoning_text(value)
    if not cleaned:
        return ""
    for separator in (" - ", " -- ", " | "):
        if separator in cleaned:
            cleaned = cleaned.split(separator, 1)[0].strip()
    if " (" in cleaned:
        cleaned = cleaned.split(" (", 1)[0].strip()
    return cleaned.rstrip(".,;:")


def _clean_quality_item(value: str | None) -> str:
    cleaned = clean_reasoning_text(value)
    if not cleaned:
        return ""
    cleaned = cleaned.replace(";", ",")
    cleaned = _SPACE_RE.sub(" ", cleaned)
    return cleaned.strip(" .,;:")


def _normalize_marker_list(values, limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = [_clean_marker_item(value) for value in values]
    cleaned = [value for value in cleaned if value]
    return _dedupe_preserve_order(cleaned)[:limit]


def _normalize_quality_list(values, limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = [_clean_quality_item(value) for value in values]
    cleaned = [value for value in cleaned if value]
    return _dedupe_preserve_order(cleaned)[:limit]


def _format_structured_section(prefix: str, values: list[str], separator: str = ", ") -> str:
    payload = separator.join(values) if values else "none"
    return f"{prefix}: {payload}"


def build_structured_reasoning_completion(
    label: str,
    evidence: list[str],
    missing: list[str],
    conflicts: list[str],
    quality: list[str],
) -> str:
    return "\n".join(
        [
            f"LABEL: {label}",
            _format_structured_section("EVIDENCE", evidence),
            _format_structured_section("MISSING", missing),
            _format_structured_section("CONFLICTS", conflicts),
            _format_structured_section("QUALITY", quality, separator="; "),
            f"FINAL: {label}",
        ]
    )


def _strip_terminal_label_sentence(text: str, label: str) -> str:
    if not text or not label:
        return text
    escaped_label = re.escape(label)
    patterns = [
        rf"\s*Therefore, this cell is an? {escaped_label}\.?$",
        rf"\s*Therefore, this cell is {escaped_label}\.?$",
        rf"\s*Overall, this cell is an? {escaped_label}\.?$",
        rf"\s*Overall, this cell is {escaped_label}\.?$",
    ]
    stripped = text
    for pattern in patterns:
        stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def _is_malformed_reasoning(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    if lowered.startswith("{") or lowered.startswith("["):
        return True
    if "raw_gpt_output" in lowered or "http://" in lowered or "https://" in lowered:
        return True
    return len(text.split()) < 12


def clean_reasoning_dataset_sample(sample: dict) -> dict | None:
    """Normalize reasoning-style stage-2 rows into a compact structured task.

    This accepts the data-creation JSONL format with fields like `validation_ok`,
    `structured_output`, and `textual_ground_truth`. Samples that pass validation are
    rewritten into a fixed label/evidence/caveat schema for stage-2 SFT.
    """
    if not isinstance(sample, dict):
        return None

    if "validation_ok" not in sample and "textual_ground_truth" not in sample and "structured_output" not in sample:
        passthrough = dict(sample)
        passthrough.pop("raw_gpt_output", None)
        return passthrough

    if sample.get("validation_ok") is not True:
        return None

    label = clean_reasoning_text(
        sample.get("answer")
        or (sample.get("structured_output") or {}).get("label")
        or ""
    )
    if not label:
        return None

    structured_output = _clean_reasoning_value(sample.get("structured_output") or {})
    evidence = _normalize_marker_list(structured_output.get("markers_present"), limit=5)
    missing = _normalize_marker_list(structured_output.get("markers_missing_from_topN"), limit=3)
    conflicts = _normalize_marker_list(structured_output.get("anti_markers_present"), limit=2)
    quality = _normalize_quality_list(structured_output.get("confounders"), limit=3)
    answer = build_structured_reasoning_completion(label, evidence, missing, conflicts, quality)

    normalized = {
        "sample_id": sample.get("id") or sample.get("sample_id"),
        "task_type": "freeform_qa",
        "question": sample.get("question"),
        "question_text": STRUCTURED_REASONING_QUESTION_TEXT,
        "answer_format": STRUCTURED_REASONING_ANSWER_FORMAT,
        "answer": answer,
        "metadata": {
            "source": "reasoning_dataset",
            "donor_id": sample.get("donor_id"),
        },
    }
    return normalized


def extract_genes_from_question(question_text: str) -> list[str]:
    """Extract the ordered gene list from the dataset question field."""
    cs_marker = "Cell sentence:\n"
    cs_start = question_text.find(cs_marker)
    if cs_start == -1:
        cs_marker = "Cell sentence:"
        cs_start = question_text.find(cs_marker)
    if cs_start == -1:
        return []

    after_marker = question_text[cs_start + len(cs_marker):]
    for stopper in ["\nQuestion:", "\nThe cell type"]:
        stop_idx = after_marker.find(stopper)
        if stop_idx != -1:
            after_marker = after_marker[:stop_idx]
            break
    return after_marker.strip().split()


def build_c2s_prompt(ordered_genes: list[str], organism: str = "Homo sapiens") -> str:
    """Build the C2S completion-style prompt used by train/eval scripts."""
    cell_sentence = " ".join(ordered_genes)
    num_genes = len(ordered_genes)
    return (
        f"The following is a list of {num_genes} gene names ordered by descending "
        f"expression level in a {organism} cell. Your task is to give the cell type "
        f"which this cell belongs to based on its gene expression.\n"
        f"Cell sentence: {cell_sentence}.\n"
        f"The cell type corresponding to these genes is:"
    )


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1)


def get_hidden_states(
    model,
    tokenizer,
    prompts: list[str],
    max_len: int,
    enable_grad: bool | None = None,
) -> torch.Tensor:
    """Return last-token hidden state for each prompt. Shape: [B, H]."""
    if enable_grad is None:
        enable_grad = model.training

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(model.device)

    forward_kwargs = {
        "output_hidden_states": True,
    }
    # Gemma CausalLM computes vocab logits even when we only need hidden states.
    # Under PEFT wrappers, `model.forward` hides the Gemma-specific signature but still
    # forwards unknown kwargs to the base model, so key off model type instead.
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in {"gemma2", "gemma"}:
        forward_kwargs["logits_to_keep"] = 1

    with torch.set_grad_enabled(enable_grad), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(**inputs, **forward_kwargs)

    last_hidden = outputs.hidden_states[-1]
    attention_mask = inputs["attention_mask"]
    last_token_pos = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
    hidden = last_hidden[batch_indices, last_token_pos]
    return hidden


def print_recall_at_k(
    name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    run_tag: str,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict | None:
    """Print Recall@K retrieval metrics for a masked subset."""
    n_samples = int(np.sum(mask))
    if n_samples < 2:
        return None

    sub_emb = embeddings[mask]
    sub_labs = labels[mask]
    max_k = min(max(ks), n_samples - 1)
    if max_k < 1:
        return None

    nn_index = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine")
    nn_index.fit(sub_emb)
    _, nn_ids = nn_index.kneighbors(sub_emb)

    # Drop self-neighbor at rank 0.
    nn_ids = nn_ids[:, 1:]
    nn_labels = sub_labs[nn_ids]
    matches = nn_labels == sub_labs[:, None]

    print(f"\n--- Retrieval [{run_tag}] on {name} ({n_samples} samples) ---")
    result = {
        "name": name,
        "run_tag": run_tag,
        "n_samples": n_samples,
        "metric": "cosine",
        "recall": {},
    }
    for k in ks:
        key = f"R@{k}"
        if k > max_k:
            print(f"  Recall@{k:2d}: N/A (need > {k} samples)")
            result["recall"][key] = None
            continue
        recall_k = matches[:, :k].any(axis=1).mean()
        print(f"  Recall@{k:2d}: {recall_k:.4f}")
        result["recall"][key] = float(recall_k)

    return result
