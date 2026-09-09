from .config import PipelineConfig
from .formatting import ANSWER_SCHEMA_NAME as CURRENT_ANSWER_SCHEMA_NAME
from .marker_catalog import get_marker_profile
from .schemas import Claim, TaskExample, VerificationIssue, VerificationResult


ALLOWED_CONFIDENCE = {"high", "medium", "low"}
NON_INFORMATIVE_GENE_PREFIXES = ("RPL", "RPS", "MT-", "MTRNR", "MALAT1", "NEAT1", "HSP", "LINC")
CORE_REQUIRED_ANSWER_FIELDS = ["LABEL", "EVIDENCE", "EXCLUSIONS", "CONTEXT", "CONFIDENCE", "FINAL"]
MARKER_PANEL_FIELDS = ["NEGATIVE_MARKERS"]


def _parse_answer_fields(answer_text: str) -> tuple[dict[str, str], list[VerificationIssue]]:
    parsed = {}
    issues = []
    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in parsed:
            issues.append(VerificationIssue(field_name="answer_text", message=f"Duplicate answer field {key}"))
        parsed[key] = value
    for field_name in CORE_REQUIRED_ANSWER_FIELDS:
        if field_name not in parsed:
            issues.append(VerificationIssue(field_name="answer_text", message=f"Missing answer field {field_name}:"))
    return parsed, issues


def _is_informative_gene(gene: str) -> bool:
    upper = gene.upper()
    return not upper.startswith(NON_INFORMATIVE_GENE_PREFIXES)


def _verify_answer_shape(answer_text: str) -> tuple[dict[str, str], list[VerificationIssue]]:
    return _parse_answer_fields(answer_text)


def _verify_evidence(genes: list[str], claims: list[Claim]) -> list[VerificationIssue]:
    issues = []
    gene_set = set(genes)
    if not claims:
        issues.append(VerificationIssue(field_name="evidence", message="Missing evidence claims"))
        return issues
    for claim in claims:
        if not claim.verified:
            issues.append(VerificationIssue(field_name="evidence", message=f"Evidence claim {claim.value} is not marked verified"))
        if claim.source_type not in {"observed_gene", "marker_catalog"}:
            issues.append(VerificationIssue(field_name="evidence", message=f"Unsupported evidence source {claim.source_type}"))
        if claim.value not in gene_set:
            issues.append(VerificationIssue(field_name="evidence", message=f"Evidence gene {claim.value} not present in observed genes"))
        if not _is_informative_gene(claim.value):
            issues.append(VerificationIssue(field_name="evidence", message=f"Evidence gene {claim.value} is too generic to be trusted"))
        if claim.source_type == "marker_catalog":
            profile = get_marker_profile(claim.matched_label or None, {"cell_type": claim.matched_label})
            if profile is None:
                issues.append(VerificationIssue(field_name="evidence", message=f"No marker profile found for evidence label {claim.matched_label}"))
            elif claim.value not in profile.positive_markers:
                issues.append(VerificationIssue(field_name="evidence", message=f"Evidence gene {claim.value} is not an approved marker for {claim.matched_label}"))
            if claim.support_direction != "supports_gold":
                issues.append(VerificationIssue(field_name="evidence", message=f"Evidence gene {claim.value} has invalid support direction {claim.support_direction}"))
    if len({claim.value for claim in claims}) != len(claims):
        issues.append(VerificationIssue(field_name="evidence", message="Duplicate evidence genes are not allowed"))
    return issues


def _verify_exclusions(example: TaskExample) -> list[VerificationIssue]:
    issues = []
    gold = example.gold_fields.get("label")
    candidate_set = set(example.candidate_labels)
    if example.task_type == "differential_diagnosis":
        if len(example.candidate_labels) < 2:
            issues.append(VerificationIssue(field_name="candidate_labels", message="Differential diagnosis requires at least two candidate labels"))
        if gold not in candidate_set:
            issues.append(VerificationIssue(field_name="candidate_labels", message="Gold label missing from candidate labels"))
    for claim in example.exclusion_claims:
        if not claim.verified:
            issues.append(VerificationIssue(field_name="exclusions", message=f"Exclusion claim {claim.value} is not marked verified"))
        if claim.value == gold:
            issues.append(VerificationIssue(field_name="exclusions", message="Gold label appears in exclusions"))
        if candidate_set and claim.value not in candidate_set:
            issues.append(VerificationIssue(field_name="exclusions", message=f"Exclusion {claim.value} missing from candidate set"))
        if claim.source_type not in {"candidate_set", "marker_catalog"}:
            issues.append(VerificationIssue(field_name="exclusions", message=f"Unsupported exclusion source {claim.source_type}"))
        if claim.source_type == "marker_catalog":
            profile = get_marker_profile(claim.matched_label or claim.value, {"cell_type": claim.matched_label or claim.value})
            if profile is None:
                issues.append(VerificationIssue(field_name="exclusions", message=f"No marker profile found for exclusion label {claim.matched_label or claim.value}"))
            if claim.support_direction != "against_candidate":
                issues.append(VerificationIssue(field_name="exclusions", message=f"Invalid exclusion support direction {claim.support_direction}"))
            if "reason=missing_candidate_markers" not in claim.details:
                issues.append(VerificationIssue(field_name="exclusions", message=f"Exclusion {claim.value} is missing biological rationale details"))
            else:
                segments = {}
                for chunk in claim.details.split(";"):
                    if "=" not in chunk:
                        continue
                    key, value = chunk.split("=", 1)
                    segments[key] = value
                candidate_markers = [item for item in segments.get("candidate_markers", "").split(",") if item]
                if not candidate_markers:
                    issues.append(VerificationIssue(field_name="exclusions", message=f"Exclusion {claim.value} is missing candidate markers in details"))
                elif profile is not None:
                    for marker in candidate_markers:
                        if marker not in profile.positive_markers:
                            issues.append(VerificationIssue(field_name="exclusions", message=f"Exclusion marker {marker} is not approved for candidate {claim.value}"))
                        if marker in example.genes:
                            issues.append(VerificationIssue(field_name="exclusions", message=f"Exclusion marker {marker} is present in the observed cell for candidate {claim.value}"))
    if example.task_type == "differential_diagnosis" and len(example.exclusion_claims) != max(0, len(example.candidate_labels) - 1):
        issues.append(VerificationIssue(field_name="exclusions", message="Differential diagnosis exclusions must cover all non-gold candidates"))
    return issues


def _verify_context(metadata: dict, claims: list[Claim], config: PipelineConfig) -> list[VerificationIssue]:
    issues = []
    for claim in claims:
        if not claim.verified:
            issues.append(VerificationIssue(field_name="context", message=f"Context claim {claim.value} is not marked verified"))
        if claim.source_type != "source_metadata":
            issues.append(VerificationIssue(field_name="context", message=f"Unsupported context source {claim.source_type}"))
            continue
        if claim.source_key not in config.defaults.context_allowlist:
            issues.append(VerificationIssue(field_name="context", message=f"Context key {claim.source_key} is not whitelisted"))
            continue
        value = metadata.get(claim.source_key)
        expected = f"{claim.source_key}={value}"
        if value is None or claim.value != expected:
            issues.append(VerificationIssue(field_name="context", message=f"Context claim {claim.value} does not match source metadata"))
    return issues


def _verify_no_label_leakage(example: TaskExample) -> list[VerificationIssue]:
    issues = []
    gold = str(example.gold_fields.get("label") or "").strip().lower()
    if not gold:
        return issues
    forbidden_keys = set()
    if example.task_type == "cell_annotation_rationale":
        forbidden_keys = {"cell_type", "cell_type_ontology_term_id"}
    elif example.task_type == "tissue_identification_rationale":
        forbidden_keys = {"tissue", "tissue_general", "tissue_ontology_term_id"}
    elif example.task_type == "cell_state_rationale":
        source_key = example.gold_fields.get("source_key")
        if source_key:
            forbidden_keys = {str(source_key)}
    elif example.task_type == "differential_diagnosis":
        forbidden_keys = {"cell_type", "cell_type_ontology_term_id"}

    for claim in example.context_claims:
        if claim.source_key in forbidden_keys:
            issues.append(VerificationIssue(field_name="context", message=f"Context includes forbidden label-bearing key {claim.source_key}"))
        if gold and gold in claim.value.strip().lower():
            issues.append(VerificationIssue(field_name="context", message="Context appears to leak the gold label"))
    return issues


def _requires_marker_panel_fields(example: TaskExample) -> bool:
    return example.answer_schema_name == CURRENT_ANSWER_SCHEMA_NAME


def _expected_negative_markers(example: TaskExample) -> str:
    canonical_label = str(example.provenance.get("canonical_label_for_grounding") or "").strip() or None
    gold_label = str(example.gold_fields.get("label") or "").strip() or None
    lookup_label = canonical_label or gold_label
    if not lookup_label:
        return "none"

    profile = get_marker_profile(
        lookup_label,
        {**example.metadata, "cell_type": canonical_label or gold_label or lookup_label},
    )
    if profile is None:
        return "none"

    return ", ".join(profile.negative_markers) if profile.negative_markers else "none"


def _verify_answer_alignment(example: TaskExample, parsed_fields: dict[str, str]) -> list[VerificationIssue]:
    issues = []
    gold = str(example.gold_fields.get("label") or "").strip()
    evidence = ", ".join(claim.value for claim in example.evidence_claims) if example.evidence_claims else "none"
    exclusions = ", ".join(claim.value for claim in example.exclusion_claims) if example.exclusion_claims else "none"
    context = "; ".join(claim.value for claim in example.context_claims) if example.context_claims else "none"

    if parsed_fields.get("LABEL") != gold:
        issues.append(VerificationIssue(field_name="answer_text", message="LABEL field does not match gold label"))
    if parsed_fields.get("FINAL") != gold:
        issues.append(VerificationIssue(field_name="answer_text", message="FINAL field does not match gold label"))
    if parsed_fields.get("EVIDENCE") != evidence:
        issues.append(VerificationIssue(field_name="answer_text", message="EVIDENCE field does not match evidence claims"))
    if parsed_fields.get("EXCLUSIONS") != exclusions:
        issues.append(VerificationIssue(field_name="answer_text", message="EXCLUSIONS field does not match exclusion claims"))
    if parsed_fields.get("CONTEXT") != context:
        issues.append(VerificationIssue(field_name="answer_text", message="CONTEXT field does not match context claims"))
    if _requires_marker_panel_fields(example):
        negative_markers = _expected_negative_markers(example)
        if "NEGATIVE_MARKERS" in parsed_fields and parsed_fields.get("NEGATIVE_MARKERS") != negative_markers:
            issues.append(VerificationIssue(field_name="answer_text", message="NEGATIVE_MARKERS field does not match the grounded marker profile"))
    confidence = parsed_fields.get("CONFIDENCE", "").lower()
    if confidence not in ALLOWED_CONFIDENCE:
        issues.append(VerificationIssue(field_name="answer_text", message=f"Invalid CONFIDENCE value {parsed_fields.get('CONFIDENCE', '')}"))
    return issues


def verify_example(example: TaskExample, config: PipelineConfig) -> VerificationResult:
    issues = []
    if not example.genes:
        issues.append(VerificationIssue(field_name="genes", message="Empty gene list"))
    if not example.gold_fields.get("label"):
        issues.append(VerificationIssue(field_name="gold_fields", message="Missing gold label"))
    if example.task_type not in {
        "cell_annotation_rationale",
        "tissue_identification_rationale",
        "cell_state_rationale",
        "differential_diagnosis",
    }:
        issues.append(VerificationIssue(field_name="task_type", message=f"Unsupported task type {example.task_type}"))

    parsed_fields, shape_issues = _verify_answer_shape(example.answer_text)
    issues.extend(shape_issues)
    if _requires_marker_panel_fields(example):
        for field_name in MARKER_PANEL_FIELDS:
            if field_name not in parsed_fields:
                issues.append(VerificationIssue(field_name="answer_text", message=f"Missing answer field {field_name}:"))
    issues.extend(_verify_evidence(example.genes, example.evidence_claims))
    issues.extend(_verify_exclusions(example))
    issues.extend(_verify_context(example.metadata, example.context_claims, config))
    issues.extend(_verify_no_label_leakage(example))
    if not shape_issues:
        issues.extend(_verify_answer_alignment(example, parsed_fields))

    return VerificationResult(
        sample_id=example.sample_id,
        task_type=example.task_type,
        passed=not issues,
        issues=issues,
    )


def verify_examples(examples: list[TaskExample], config: PipelineConfig) -> tuple[list[TaskExample], list[VerificationResult]]:
    kept = []
    results = []
    for example in examples:
        result = verify_example(example, config)
        results.append(result)
        if result.passed:
            kept.append(example)
    return kept, results