from .schemas import Claim


ANSWER_SCHEMA_NAME = "structured_claim_v2"


def _join_text_values(values: list[str] | None, separator: str = ", ") -> str:
    if not values:
        return "none"
    return separator.join(values)


def format_answer(
    label: str,
    evidence_claims: list[Claim],
    exclusion_claims: list[Claim],
    context_claims: list[Claim],
    confidence: str,
    negative_markers: list[str] | None = None,
) -> str:
    evidence = _join_text_values([claim.value for claim in evidence_claims])
    negative_marker_text = _join_text_values(negative_markers)
    exclusions = _join_text_values([claim.value for claim in exclusion_claims])
    context = _join_text_values([claim.value for claim in context_claims], separator="; ")
    return "\n".join(
        [
            f"LABEL: {label}",
            f"NEGATIVE_MARKERS: {negative_marker_text}",
            f"EVIDENCE: {evidence}",
            f"EXCLUSIONS: {exclusions}",
            f"CONTEXT: {context}",
            f"CONFIDENCE: {confidence}",
            f"FINAL: {label}",
        ]
    )