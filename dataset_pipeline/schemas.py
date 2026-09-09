from dataclasses import asdict, dataclass, field


VALID_TASK_TYPES = {
    "cell_annotation_rationale",
    "tissue_identification_rationale",
    "cell_state_rationale",
    "differential_diagnosis",
}


@dataclass
class Claim:
    value: str
    source_type: str
    source_key: str
    verified: bool
    details: str = ""
    claim_kind: str = ""
    source_tier: str = ""
    matched_label: str = ""
    support_direction: str = ""
    score: float = 0.0


@dataclass
class NormalizedCellRecord:
    sample_id: str
    source_name: str
    source_dataset_id: str
    organism: str
    genes: list[str]
    metadata: dict


@dataclass
class TaskExample:
    sample_id: str
    task_type: str
    source_name: str
    source_dataset_id: str
    organism: str
    genes: list[str]
    question_text: str
    answer_text: str
    answer_schema_name: str
    gold_fields: dict
    evidence_claims: list[Claim] = field(default_factory=list)
    exclusion_claims: list[Claim] = field(default_factory=list)
    context_claims: list[Claim] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    candidate_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["evidence_claims"] = [asdict(item) for item in self.evidence_claims]
        payload["exclusion_claims"] = [asdict(item) for item in self.exclusion_claims]
        payload["context_claims"] = [asdict(item) for item in self.context_claims]
        return payload


@dataclass
class VerificationIssue:
    field_name: str
    message: str


@dataclass
class VerificationResult:
    sample_id: str
    task_type: str
    passed: bool
    issues: list[VerificationIssue]

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "task_type": self.task_type,
            "passed": self.passed,
            "issues": [asdict(item) for item in self.issues],
        }