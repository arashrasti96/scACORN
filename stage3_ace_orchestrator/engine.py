from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any
import warnings

import torch
from peft import PeftModel
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CONTRASTIVE_ROOT = Path(__file__).resolve().parent.parent
if str(CONTRASTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTRASTIVE_ROOT))

from utils import build_c2s_prompt  # noqa: E402

from .config import Stage3Paths, Stage3RuntimeConfig
from .registry import ExpertDescriptor, load_expert_registry
from .requests import NormalizedStage3Request


def _normalize_profile_id(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


@contextmanager
def _suppress_meta_parameter_copy_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*copying from a non-meta parameter in the checkpoint to a meta parameter.*",
            category=UserWarning,
        )
        yield


@contextmanager
def _force_cpu_model_construction(dtype: torch.dtype | None = None):
    previous_device = None
    previous_dtype = torch.get_default_dtype()
    set_default_device = getattr(torch, "set_default_device", None)
    get_default_device = getattr(torch, "get_default_device", None)
    if callable(set_default_device) and callable(get_default_device):
        previous_device = get_default_device()
        set_default_device("cpu")
    if dtype is not None:
        torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        if dtype is not None:
            torch.set_default_dtype(previous_dtype)
        if previous_device is not None and callable(set_default_device):
            set_default_device(previous_device)


def activate_adapters(model: PeftModel, adapter_names: str | list[str]) -> None:
    if isinstance(adapter_names, str):
        model.set_adapter(adapter_names)
        return

    if not adapter_names:
        raise ValueError("At least one adapter must be activated")
    primary_adapter = adapter_names[0]
    if primary_adapter not in model.peft_config:
        raise ValueError(f"Adapter {primary_adapter} not found.")
    model.active_adapter = primary_adapter
    base_model = getattr(model, "base_model", None)
    if base_model is None or not hasattr(base_model, "set_adapter"):
        raise TypeError("This PEFT model does not support activating multiple adapters")
    base_model.set_adapter(adapter_names)


def _single_device_map() -> dict[str, int] | None:
    if torch.cuda.is_available():
        return {"": torch.cuda.current_device()}
    return None


def _model_has_adapter(model: PeftModel | Any, adapter_name: str | None) -> bool:
    if not adapter_name or not isinstance(model, PeftModel):
        return False
    peft_config = getattr(model, "peft_config", None)
    return isinstance(peft_config, dict) and adapter_name in peft_config


def _load_base_config(base_model: str):
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    index_path = Path(base_model) / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
        if "model.embed_tokens.weight" in weight_map and "lm_head.weight" not in weight_map:
            config.tie_word_embeddings = True
    return config


def _tie_output_embeddings_before_device_move(model: Any) -> None:
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if not callable(get_input_embeddings) or not callable(get_output_embeddings):
        return

    input_embeddings = get_input_embeddings()
    output_embeddings = get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        return

    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    if input_weight is None or output_weight is None:
        return
    if getattr(output_weight, "is_meta", False) and not getattr(input_weight, "is_meta", False):
        output_embeddings.weight = input_weight


def _raise_if_model_has_meta_parameters(model: Any, context: str) -> None:
    meta_names = [name for name, parameter in model.named_parameters() if getattr(parameter, "is_meta", False)]
    if meta_names:
        preview = ", ".join(meta_names[:10])
        suffix = "..." if len(meta_names) > 10 else ""
        raise RuntimeError(f"Model still has meta parameters {context}: {preview}{suffix}")


def _has_meta_parameters(model: Any) -> bool:
    return any(getattr(parameter, "is_meta", False) for parameter in model.parameters())


def _load_base_model_from_safetensors(base_model: str, config: Any):
    model_path = Path(base_model)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Cannot manually load base model; missing {index_path}")

    with _force_cpu_model_construction(torch.bfloat16):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    with index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle).get("weight_map", {})
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        shard_state = load_safetensors_file(str(model_path / shard_name), device="cpu")
        model.load_state_dict(shard_state, strict=False)
        del shard_state

    model.to(dtype=torch.bfloat16)
    return model


def load_base_model(base_model: str, use_4bit: bool):
    config = _load_base_config(base_model)
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=config,
            quantization_config=quant_config,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        _tie_output_embeddings_before_device_move(model)
        _raise_if_model_has_meta_parameters(model, "after 4-bit base load")
        return model

    model = _load_base_model_from_safetensors(base_model, config)
    _tie_output_embeddings_before_device_move(model)
    _raise_if_model_has_meta_parameters(model, "after bf16 base load")
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda", torch.cuda.current_device()))
    return model


def _resolve_model_device(model: PeftModel | Any) -> torch.device:
    for parameter in model.parameters():
        if not getattr(parameter, "is_meta", False):
            return parameter.device
    return model.device


def _embedding_vocab_size(model: PeftModel | Any) -> int | None:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        num_embeddings = getattr(embeddings, "num_embeddings", None)
        if isinstance(num_embeddings, int):
            return num_embeddings
        weight = getattr(embeddings, "weight", None)
        if weight is not None and getattr(weight, "ndim", 0) >= 1:
            return int(weight.shape[0])
    base_model = getattr(model, "base_model", None)
    if base_model is not None and base_model is not model:
        return _embedding_vocab_size(base_model)
    return None


def _validate_input_ids_for_model(model: PeftModel | Any, input_ids: torch.Tensor) -> None:
    vocab_size = _embedding_vocab_size(model)
    if vocab_size is None or input_ids.numel() == 0:
        return
    min_id = int(input_ids.min().item())
    max_id = int(input_ids.max().item())
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"Token ids are outside the model embedding table: min={min_id} max={max_id} vocab_size={vocab_size}"
        )


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    trimmed = text.strip()
    if not trimmed:
        return None
    if trimmed.startswith("```"):
        lines = trimmed.splitlines()[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        trimmed = "\n".join(lines).strip()
    if trimmed and not trimmed.startswith("{"):
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start != -1 and end != -1 and end > start:
            trimmed = trimmed[start : end + 1]
    try:
        parsed = json.loads(trimmed)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_stage2_text_payload(text: str) -> dict[str, Any] | None:
    fields: dict[str, Any] = {}
    structured_lines: list[str] = []
    field_pattern = re.compile(
        r"\b(label|final|positive_markers|negative_markers|evidence|context|confidence)\s*:",
        flags=re.IGNORECASE,
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = field_pattern.search(line)
        if not match:
            continue
        line = line[match.start() :].strip()
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower()
        normalized_value = value.strip()
        if normalized_key != "positive_markers":
            structured_lines.append(line)
        if normalized_key in {"positive_markers", "negative_markers", "evidence"}:
            fields[normalized_key] = [
                token.strip()
                for token in normalized_value.split(",")
                if token.strip() and token.strip().lower() != "none"
            ]
            continue
        fields[normalized_key] = normalized_value

    if not fields:
        return None

    final_label = fields.get("final") or fields.get("label")
    if final_label:
        fields["final_label"] = final_label
    if "evidence" in fields:
        fields["supporting_evidence"] = list(fields["evidence"])
        fields["supporting_genes"] = list(fields["evidence"])
    if fields:
        fields["raw_response"] = "\n".join(structured_lines).strip() or text.strip()
        fields["parse_fallback"] = "stage2_key_value"
    return fields


def _parse_free_text_fallback(text: str) -> dict[str, Any] | None:
    trimmed = text.strip()
    if not trimmed:
        return None

    lines = [line.strip() for line in trimmed.splitlines() if line.strip()]
    if not lines:
        return None

    parsed: dict[str, Any] = {}
    lowered_lines = [line.lower() for line in lines]

    def _extract_csv(line: str) -> list[str]:
        _, value = line.split(":", 1)
        items = [token.strip() for token in re.split(r"[,;]", value) if token.strip()]
        return items

    for idx, line in enumerate(lowered_lines):
        original = lines[idx]
        if line.startswith("final label:"):
            parsed["final_label"] = original.split(":", 1)[1].strip()
        elif line.startswith("label:") and "final_label" not in parsed:
            parsed["final_label"] = original.split(":", 1)[1].strip()
        elif line.startswith("supporting genes:"):
            parsed["supporting_genes"] = _extract_csv(original)
            parsed.setdefault("supporting_evidence", list(parsed["supporting_genes"]))
        elif line.startswith("evidence:") and "supporting_evidence" not in parsed:
            parsed["supporting_evidence"] = _extract_csv(original)
        elif line.startswith("negative markers:"):
            parsed["negative_markers"] = _extract_csv(original)
        elif line.startswith("confidence:"):
            parsed["confidence"] = original.split(":", 1)[1].strip().lower()
        elif line.startswith("predicted tissue:"):
            parsed["predicted_tissue"] = original.split(":", 1)[1].strip()

    if "final_label" not in parsed:
        match = re.search(r"(?:final label|best[- ]supported label)\s*:\s*([^\n]+)", trimmed, flags=re.IGNORECASE)
        if match:
            parsed["final_label"] = match.group(1).strip()

    if "final_label" not in parsed and len(lines) == 1:
        plain_label = lines[0].strip().strip(". ")
        if plain_label and len(plain_label.split()) <= 12:
            parsed["final_label"] = plain_label

    if not parsed:
        return None

    parsed["raw_response"] = trimmed
    parsed["parse_fallback"] = "free_text"
    return parsed


def _clean_label_text(value: str) -> str:
    label = value.strip().strip(".;, ")
    label = re.split(r"\s+CONTEXT\s*:", label, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    label = re.split(r"\s+LABEL\s*:", label, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return label.strip().strip(".;, ")


def _parse_cell_type_completion(text: str) -> dict[str, Any] | None:
    trimmed = text.strip()
    if not trimmed:
        return None

    first_line = next((line.strip() for line in trimmed.splitlines() if line.strip()), "")
    if not first_line:
        return None
    if ":" in first_line:
        key, value = first_line.split(":", 1)
        if key.strip().lower() in {"label", "final", "final_label", "cell_type", "cell type"}:
            label = _clean_label_text(value)
        else:
            label = _clean_label_text(first_line)
    else:
        label = _clean_label_text(first_line)

    if not label or len(label.split()) > 12:
        return None
    return {
        "final_label": label,
        "label": label,
        "raw_response": trimmed,
        "parse_fallback": "label_only_cell_type_completion",
    }


def _parse_expert_payload(text: str, task_type: str = "") -> dict[str, Any] | None:
    if task_type in {"cell_type", "domain_replay"}:
        return _parse_json_payload(text) or _parse_stage2_text_payload(text) or _parse_cell_type_completion(text)
    return _parse_json_payload(text) or _parse_stage2_text_payload(text) or _parse_free_text_fallback(text)


def _remove_positive_marker_lines(text: str) -> str:
    return "\n".join(
        line
        for line in text.splitlines()
        if not line.strip().lower().startswith("positive_markers:")
    ).strip()


def _sanitize_parsed_output(parsed_output: dict[str, Any] | None) -> dict[str, Any] | None:
    if not parsed_output:
        return None

    sanitized = dict(parsed_output)
    sanitized.pop("positive_markers", None)
    if isinstance(sanitized.get("raw_response"), str):
        sanitized["raw_response"] = _remove_positive_marker_lines(str(sanitized["raw_response"]))
    if "evidence" in sanitized and "supporting_evidence" not in sanitized:
        sanitized["supporting_evidence"] = sanitized["evidence"]
    return sanitized


def _sanitize_alias(prefix: str, value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value)
    return f"{prefix}_{cleaned}"[:120]


@dataclass(frozen=True)
class ExpertExecutionResult:
    expert_name: str
    prompt: str
    generated_text: str
    parsed_output: dict[str, Any] | None
    latency_seconds: float
    task_type: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_public_tool_payload(self) -> dict[str, Any]:
        sanitized_output = _sanitize_parsed_output(self.parsed_output)
        if sanitized_output is None and self.generated_text.strip():
            sanitized_output = {
                "raw_response": _remove_positive_marker_lines(self.generated_text.strip()),
                "parse_fallback": "raw_only",
            }
        return {
            "expert_name": self.expert_name,
            "task_type": self.task_type,
            "latency_seconds": self.latency_seconds,
            "metadata": self.metadata,
            "answer": sanitized_output,
        }


class Stage3ExpertRuntime:
    def __init__(
        self,
        paths: Stage3Paths | None = None,
        config: Stage3RuntimeConfig | None = None,
        *,
        outputs_root: Path | None = None,
    ) -> None:
        self.paths = paths or Stage3Paths.discover()
        self.config = config or Stage3RuntimeConfig()
        self.registry = load_expert_registry(
            self.paths,
            outputs_root=outputs_root,
            limit=self.config.registry_limit,
        )
        self._model: PeftModel | Any | None = None
        self._tokenizers: dict[str, Any] = {}
        self._loaded_stage2_aliases: OrderedDict[str, str] = OrderedDict()
        self._loaded_stage1_aliases: dict[str, str] = {}
        self._call_log: list[dict[str, Any]] = []
        self._active_ranked_genes: tuple[str, ...] = ()
        self._active_profile_gene_lists: dict[str, tuple[str, ...]] = {}
        self._expert_lock = threading.RLock()

    def reset_call_log(self) -> None:
        self._call_log = []

    def get_call_log(self) -> list[dict[str, Any]]:
        return list(self._call_log)

    def set_active_ranked_genes(self, genes: list[str] | tuple[str, ...]) -> None:
        self._active_ranked_genes = tuple(str(gene).strip() for gene in genes if str(gene).strip())

    def get_active_ranked_genes(self) -> tuple[str, ...]:
        return self._active_ranked_genes

    def set_active_profile_gene_lists(self, profile_gene_lists: dict[str, list[str] | tuple[str, ...]]) -> None:
        normalized_profile_gene_lists: dict[str, tuple[str, ...]] = {}
        for raw_profile_id, raw_genes in profile_gene_lists.items():
            normalized_profile_id = _normalize_profile_id(raw_profile_id)
            genes = tuple(str(gene).strip() for gene in raw_genes if str(gene).strip())
            if normalized_profile_id and genes:
                normalized_profile_gene_lists[normalized_profile_id] = genes
        self._active_profile_gene_lists = normalized_profile_gene_lists

    def get_active_profile_gene_lists(self) -> dict[str, tuple[str, ...]]:
        return dict(self._active_profile_gene_lists)

    def get_active_profile_genes(self, profile_id: str | None) -> tuple[str, ...] | None:
        return self._active_profile_gene_lists.get(_normalize_profile_id(profile_id))

    def get_active_profile_ids(self) -> list[str]:
        return list(self._active_profile_gene_lists)

    def clear_active_ranked_genes(self) -> None:
        self._active_ranked_genes = ()
        self._active_profile_gene_lists = {}

    def list_experts(self, *, tissue_filter: str | None = None, limit: int | None = None) -> list[ExpertDescriptor]:
        experts = list(self.registry.values())
        if tissue_filter:
            lowered = tissue_filter.lower()
            experts = [
                expert
                for expert in experts
                if lowered in expert.tissue_label.lower() or lowered in expert.description.lower()
            ]
        if limit is not None:
            experts = experts[:limit]
        return experts

    def format_expert_catalog(self, *, tissue_filter: str | None = None, limit: int | None = None) -> str:
        experts = self.list_experts(tissue_filter=tissue_filter, limit=limit)
        if not experts:
            return "No experts available."
        return "\n".join(
            (
                f"- {expert.name}: {expert.description} "
                f"Capabilities: {', '.join(expert.supported_task_types or ('cell_type',))}."
            )
            for expert in experts
        )

    def run_expert(self, expert_name: str, request: NormalizedStage3Request) -> ExpertExecutionResult:
        with self._expert_lock:
            if self._active_profile_gene_lists:
                request = self._verify_active_profile_request(request)
            elif self._active_ranked_genes:
                request = request.with_tool_overrides(genes_csv=", ".join(self._active_ranked_genes))
            return self._run_expert_locked(expert_name, request)

    def _verify_active_profile_request(self, request: NormalizedStage3Request) -> NormalizedStage3Request:
        profile_id = str((request.metadata or {}).get("tool_profile_id") or "").strip()
        if not profile_id:
            raise RuntimeError(
                "Active sample contains multiple cell/profile gene lists. Expert calls must specify profile_id "
                "and use the canonical 200-gene list for that profile."
            )

        canonical_genes = self.get_active_profile_genes(profile_id)
        if canonical_genes is None:
            available_profiles = ", ".join(self.get_active_profile_ids()) or "none"
            raise RuntimeError(
                f"No canonical gene list is available for profile_id='{profile_id}'. "
                f"Available profile IDs: {available_profiles}."
            )
        if len(canonical_genes) != self.config.top_genes:
            raise RuntimeError(
                f"Canonical gene list for profile_id='{profile_id}' has {len(canonical_genes)} genes; "
                f"expected exactly {self.config.top_genes}. Stopping because multi-cell expert calls must use "
                "the fixed trained prompt length."
            )
        if tuple(request.genes) != canonical_genes:
            raise RuntimeError(
                f"Expert call genes for profile_id='{profile_id}' did not match the canonical "
                f"{self.config.top_genes}-gene list for the active sample. Stopping instead of allowing "
                "a drifted multi-cell prompt."
            )
        return request

    def _run_expert_locked(self, expert_name: str, request: NormalizedStage3Request) -> ExpertExecutionResult:
        expert = self.registry[expert_name]
        model = self._ensure_expert_loaded(expert)
        tokenizer = self._ensure_tokenizer(expert)
        prompt = self._build_prompt(request)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len,
        ).to(_resolve_model_device(model))
        _validate_input_ids_for_model(model, inputs["input_ids"])
        start = time.time()
        with torch.no_grad():
            if self.config.temperature == 0.0:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    use_cache=False,
                )
            else:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    use_cache=False,
                )
        latency = time.time() - start
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        if not generated.strip():
            raise RuntimeError(
                f"Expert '{expert.name}' returned an empty generation for task_type='{request.task_type}'. "
                "Stopping to prevent a silent no-output tool response."
            )
        parsed = _parse_expert_payload(generated, "cell_type")
        sanitized_output = _sanitize_parsed_output(parsed)
        result = ExpertExecutionResult(
            expert_name=expert.name,
            prompt=prompt,
            generated_text=generated,
            parsed_output=parsed,
            latency_seconds=latency,
            task_type=request.task_type,
            metadata={
                "tissue_label": expert.tissue_label,
                "stage1_adapter_path": str(expert.stage1_adapter_path) if expert.stage1_adapter_path else None,
            },
        )
        self._call_log.append(
            {
                "expert_name": expert.name,
                "profile_id": (request.metadata or {}).get("tool_profile_id"),
                "task_type": request.task_type,
                "task_family": request.task_family,
                "latency_seconds": latency,
                "input_gene_count": len(request.genes),
                "prompt_gene_count": min(len(request.genes), self.config.top_genes),
                "gene_verification": (request.metadata or {}).get("canonical_gene_verifier"),
                "parse_status": "parsed" if parsed else "raw_only",
                "parsed_keys": sorted(parsed.keys()) if parsed else [],
                "output_chars": len(generated.strip()),
                "prompt": prompt,
                "generated_text": generated.strip(),
                "parsed_output": sanitized_output,
            }
        )
        if self.config.verbose:
            print("\n" + "-" * 80)
            print(f"Expert output: {expert.name}")
            print(f"  task_type={request.task_type} task_family={request.task_family} parse_status={'parsed' if parsed else 'raw_only'}")
            print("  prompt:")
            print("    " + prompt.strip().replace("\n", "\n    "))
            print("  generated_text:")
            print("    " + generated.strip().replace("\n", "\n    "))
            if sanitized_output is not None:
                print("  parsed_output:")
                print("    " + json.dumps(sanitized_output, ensure_ascii=False).replace("\n", "\n    "))
            print("-" * 80)
        return result

    def _ensure_base_model(self):
        if self._model is None:
            self._model = load_base_model(self.config.base_model, self.config.use_4bit)
        return self._model

    def _ensure_tokenizer(self, expert: ExpertDescriptor):
        tokenizer = self._tokenizers.get(expert.name)
        if tokenizer is not None:
            return tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(expert.adapter_dir), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self._tokenizers[expert.name] = tokenizer
        return tokenizer

    def _ensure_expert_loaded(self, expert: ExpertDescriptor):
        model = self._ensure_base_model()
        stage2_alias = _sanitize_alias("stage2", expert.name)
        stage1_alias = _sanitize_alias("stage1", expert.name)
        adapter_device = str(_resolve_model_device(model))

        if not isinstance(model, PeftModel):
            with _suppress_meta_parameter_copy_warnings():
                model = PeftModel.from_pretrained(
                    model,
                    str(expert.adapter_dir),
                    adapter_name=stage2_alias,
                    is_trainable=False,
                    low_cpu_mem_usage=False,
                    torch_device=adapter_device,
                )
            self._model = model
            self._loaded_stage2_aliases[expert.name] = stage2_alias

        if expert.name not in self._loaded_stage2_aliases:
            with _suppress_meta_parameter_copy_warnings():
                model.load_adapter(
                    str(expert.adapter_dir),
                    adapter_name=stage2_alias,
                    is_trainable=False,
                    low_cpu_mem_usage=False,
                    torch_device=adapter_device,
                )
            self._loaded_stage2_aliases[expert.name] = stage2_alias

        if expert.stage1_adapter_path and expert.name not in self._loaded_stage1_aliases:
            with _suppress_meta_parameter_copy_warnings():
                model.load_adapter(
                    str(expert.stage1_adapter_path),
                    adapter_name=stage1_alias,
                    is_trainable=False,
                    low_cpu_mem_usage=False,
                    torch_device=adapter_device,
                )
            self._loaded_stage1_aliases[expert.name] = stage1_alias

        self._touch_stage2_alias(expert.name)
        active_aliases = []
        stage2_alias_loaded = self._loaded_stage2_aliases.get(expert.name)
        if not _model_has_adapter(model, stage2_alias_loaded):
            raise RuntimeError(f"Stage-2 adapter is not loaded for expert {expert.name}: {stage2_alias_loaded}")
        active_aliases.append(stage2_alias_loaded)

        stage1_alias_loaded = self._loaded_stage1_aliases.get(expert.name)
        if stage1_alias_loaded and _model_has_adapter(model, stage1_alias_loaded):
            active_aliases.append(stage1_alias_loaded)
        elif stage1_alias_loaded:
            self._loaded_stage1_aliases.pop(expert.name, None)
        activate_adapters(model, active_aliases if len(active_aliases) > 1 else active_aliases[0])
        model.eval()
        return model

    def _touch_stage2_alias(self, expert_name: str) -> None:
        alias = self._loaded_stage2_aliases.pop(expert_name)
        self._loaded_stage2_aliases[expert_name] = alias
        if len(self._loaded_stage2_aliases) <= self.config.expert_cache_size:
            return

        evicted_expert, evicted_alias = self._loaded_stage2_aliases.popitem(last=False)
        model = self._model
        if not isinstance(model, PeftModel):
            return
        if hasattr(model, "delete_adapter") and _model_has_adapter(model, evicted_alias):
            model.delete_adapter(evicted_alias)
            stage1_alias = self._loaded_stage1_aliases.pop(evicted_expert, None)
            if stage1_alias and hasattr(model, "delete_adapter") and _model_has_adapter(model, stage1_alias):
                model.delete_adapter(stage1_alias)
        else:
            self._loaded_stage1_aliases.pop(evicted_expert, None)

    def _build_prompt(self, request: NormalizedStage3Request) -> str:
        genes = list(request.genes)
        if not genes:
            raise RuntimeError(
                f"Cannot call expert with no genes for task_type='{request.task_type}'. "
                "Stopping because expert prompts must use the trained 200-gene C2S template."
            )
        if len(genes) < self.config.top_genes:
            raise RuntimeError(
                f"Cannot call expert with only {len(genes)} genes for task_type='{request.task_type}'. "
                f"Stage-2 experts were trained with the first {self.config.top_genes} ranked genes; "
                "stopping so the generator cannot query an expert with an incompatible prompt."
            )
        return build_c2s_prompt(genes[: self.config.top_genes])
