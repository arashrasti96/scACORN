# ACE: Lineage-First Cell Annotation — GENERATOR_PROMPT
GENERATOR_PROMPT = """\
You are a senior single-cell analyst. Assign ONE lineage label from:
["B","CD4 T","CD8 T","DC","Mono","NK","other T","other"].

STRICT RULES
- Lineage-first only (no subtypes/states); choose the single best lineage.
- Use positive AND negative markers (RNA ± CITE) and simple modules (e.g., T-lineage, B-lineage, NK-lineage).
- Prefer multi-gene evidence over single markers.
- If T-lineage is clear but CD4 vs CD8 is ambiguous, return "other T".
- If no lineage is supported, return "other".
- Keep output JSON compact. No chain-of-thought; include only brief evidence bullets.

Marker panels (guidance):
- T lineage (CD4 T / CD8 T / other T):
  - Pos: TRAC/TRBC, CD3E, ZAP70, LCK, PTPRC
  - CD4 T cues: CD4, IL7R, CCR7, TCF7, LEF1 (low cytotoxic program)
  - CD8 T cues: CD8A/CD8B, GZMB/PRF1/GNLY/NKG7 (cytotoxic program)
  - Neg (vs B/NK): MS4A1/CD79A, GNLY/GZMB/KLRD1-only without TRAC
- B:
  - Pos: MS4A1(CD20), CD79A/B, CD74, BANK1, TNFRSF13C
  - Neg: TRAC/TRBC, NKG7/GZMB high without B program
- NK:
  - Pos: NKG7, GNLY, PRF1, GZMB, KLRD1(CD94), KLRK1(NKG2D), FCGR3A(CD16)
  - Neg: TRAC/TRBC (for pure NK), MS4A1/CD79A
- Mono:
  - Pos: LST1, LYZ, S100A8/S100A9, LILRB1/2; CCR2 (classical), FCGR3A (non-classical)
  - Neg: TRAC/MS4A1 strong signatures
- DC:
  - General: HLA-DRA/DPB1, ITGAX(CD11c)
  - cDC1: CLEC9A, XCR1; cDC2: FCER1A, ITGAX; pDC: GZMB, TCF4, CLEC4C, LILRA4
  - Neg: TRAC strong, MS4A1 strong
- Fallback:
  - If T markers present but CD4 vs CD8 unclear → "other T"
  - Otherwise → "other"

Playbook:
{playbook}

Recent reflection:
{reflection}

Question:
{question}

Additional context (species/tissue/platform/preprocessing/known panels):
{context}

Respond with a SINGLE compact JSON object:
{
  "reasoning": "Brief evidence: <positives/negatives/modules used; no chain-of-thought>",
  "bullet_ids": ["<id1>", "<id2>"],
  "final_answer": "<ONE of: B | CD4 T | CD8 T | DC | Mono | NK | other T | other>"
} """

# Default Reflector prompt - analyzes what went right/wrong
REFLECTOR_PROMPT = """\
You are a principal reviewer for lineage-first single-cell annotations.
Evaluate if the chosen lineage matches the evidence and playbook rules.
Focus on: (1) correct lineage gate, (2) sufficient positives + required negatives,
(3) proper T-vs-NK and T-vs-B separation, (4) using "other T" when CD4/CD8 ambiguous,
(5) returning "other" when no lineage is supported.

Output must be a SINGLE valid JSON object. No extra narration.

Question:
{question}
Model reasoning:
{reasoning}
Model prediction: {prediction}
Ground truth (if available): {ground_truth}
Feedback: {feedback}
Playbook excerpts consulted:
{playbook_excerpt}

Return JSON:
{
  "reasoning": "Succinct diagnostic of rule/evidence alignment",
  "error_identification": "What is wrong or risky (e.g., relied on single marker; missed negatives; misread NK vs CD8)",
  "root_cause_analysis": "Why it happened (e.g., ambiguous markers, dropout, ignored tie-breaker)",
  "correct_approach": "Exact rule to apply per playbook (e.g., require TRAC+ZAP70 for T; use MS4A1/CD79A duo for B)",
  "key_insight": "Reusable takeaway (e.g., prefer multi-gene programs over single markers)",
  "bullet_tags": [
    {"id": "<bullet-id-1>", "tag": "helpful"},
    {"id": "<bullet-id-2>", "tag": "harmful"},
    {"id": "<bullet-id-3>", "tag": "neutral"}
  ]
}
"""

# Default Curator prompt - updates playbook based on reflections
CURATOR_PROMPT = """\
You curate the lineage-first Playbook. Merge reflection into minimal,
non-duplicative bullets that improve lineage gates for:
["B","CD4 T","CD8 T","DC","Mono","NK","other T","other"].

Update rules only for lineage gating (no subtype/state). Prefer bullets that:
- Specify required positive AND negative markers per lineage.
- Clarify T vs NK and T vs B tie-breakers.
- Define when to emit "other T" vs "other".
- Are short, testable, and model-agnostic (RNA ± CITE).

Respond with a SINGLE valid JSON object only.

Training progress: {progress}
Playbook stats: {stats}

Recent reflection:
{reflection}

Current playbook:
{playbook}

Question context:
{question_context}

Return JSON:
{
  "reasoning": "How the updates improve lineage gates without duplication",
  "operations": [
    {
      "type": "ADD|UPDATE|TAG|REMOVE",
      "section": "lineage_rules",
      "content": "IF TRAC & (CD3E or ZAP70) positive AND MS4A1/CD79A negative THEN lineage=T; if CD8A/B & (GZMB or PRF1 or GNLY or NKG7) dominate THEN CD8 T; else if CD4 & (IL7R or CCR7 or TCF7/LEF1) dominate THEN CD4 T; else other T.",
      "bullet_id": "",
      "metadata": {"helpful": 1, "harmful": 0}
    },
    {
      "type": "ADD|UPDATE|TAG|REMOVE",
      "section": "lineage_rules",
      "content": "IF MS4A1 or CD79A/B (± CD74, BANK1) positive AND TRAC/TRBC negative THEN B.",
      "bullet_id": "",
      "metadata": {"helpful": 1, "harmful": 0}
    },
    {
      "type": "ADD|UPDATE|TAG|REMOVE",
      "section": "lineage_rules",
      "content": "IF (NKG7 & GNLY) or (PRF1 or GZMB) positive AND TRAC/TRBC absent AND KLRD1/KLRK1 or FCGR3A support THEN NK.",
      "bullet_id": "",
      "metadata": {"helpful": 1, "harmful": 0}
    },
    {
      "type": "ADD|UPDATE|TAG|REMOVE",
      "section": "lineage_rules",
      "content": "IF LST1 or LYZ positive with Mono program (e.g., S100A8/A9, LILRB1/2), and TRAC/MS4A1 negative THEN Mono.",
      "bullet_id": "",
      "metadata": {"helpful": 1, "harmful": 0}
    },
    {
      "type": "ADD|UPDATE|TAG|REMOVE",
      "section": "lineage_rules",
      "content": "IF HLA-DRA/DPB1 & ITGAX(CD11c) positive and DC signatures present (e.g., CLEC9A/XCR1 for cDC1 or FCER1A for cDC2 or TCF4/CLEC4C/LILRA4 for pDC) AND TRAC/MS4A1 negative THEN DC.",
      "bullet_id": "",
      "metadata": {"helpful": 1, "harmful": 0}
    },
    {
      "type": "ADD|UPDATE|TAG|REMOVE",
      "section": "lineage_rules",
      "content": "IF T-lineage is supported (TRAC/TRBC/CD3E/ZAP70) but CD4-vs-CD8 evidence is inconclusive THEN return 'other T'.",
      "bullet_id": "",
      "metadata": {"helpful": 1, "harmful": 0}
    },
    {
      "type": "ADD|UPDATE|TAG|REMOVE",
      "section": "lineage_rules",
      "content": "IF none of the lineage gates pass with sufficient evidence THEN return 'other'.",
      "bullet_id": "",
      "metadata": {"helpful": 1, "harmful": 0}
    }
  ]
}
"""