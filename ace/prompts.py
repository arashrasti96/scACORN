"""
Prompt templates for ACE roles - fully customizable for your use case.

These default prompts are adapted from the ACE paper. You can customize them
to better suit your specific task by providing your own templates when
initializing the Generator, Reflector, and Curator.

Customization Example:
    >>> from ace import Generator
    >>> from ace.llm_providers import LiteLLMClient
    >>>
    >>> # Custom generator prompt for code tasks
    >>> code_generator_prompt = '''
    ... You are a senior developer. Use the playbook to write clean code.
    ...
    ... Playbook: {playbook}
    ... Reflection: {reflection}
    ... Task: {question}
    ... Requirements: {context}
    ...
    ... Return JSON with:
    ... - reasoning: Your approach
    ... - bullet_ids: Applied strategies
    ... - final_answer: The code solution
    ... '''
    >>>
    >>> client = LiteLLMClient(model="gpt-4")
    >>> generator = Generator(client, prompt_template=code_generator_prompt)

Prompt Variables:
    Generator:
        - {playbook}: Current playbook strategies
        - {reflection}: Recent reflection context
        - {question}: The question/task to solve
        - {context}: Additional requirements or context

    Reflector:
        - {question}: Original question
        - {reasoning}: Generator's reasoning
        - {prediction}: Generator's answer
        - {ground_truth}: Correct answer if available
        - {feedback}: Environment feedback
        - {playbook_excerpt}: Relevant playbook bullets used

    Curator:
        - {progress}: Training progress summary
        - {stats}: Playbook statistics
        - {reflection}: Latest reflection analysis
        - {playbook}: Current full playbook
        - {question_context}: Question and feedback context

Tips for Custom Prompts:
    1. Keep JSON output format consistent
    2. Be specific about your domain (math, code, writing, etc.)
    3. Add task-specific instructions and constraints
    4. Test with your actual use cases
    5. Iterate based on the quality of generated strategies
"""

# Default Generator prompt - produces answers using playbook strategies
GENERATOR_PROMPT = """\
You are an expert assistant for perturbation-response gene-ranking.
You MUST use the provided playbook strategies. Prefer strategies that improve:
(1) top-K overlap, then (2) correct ordering within overlap, then (3) formatting validity.

Playbook (bullets):
{playbook}

Recent reflection (most recent reflector guidance):
{reflection}

Question:
{question}

Additional context:
{context}

Hard constraints for final_answer:
- Output EXACTLY 200 space-separated gene tokens.
- NO extra text in final_answer (no punctuation, no JSON inside final_answer).
- NO duplicates.
- Maintain coherent blocks when appropriate (e.g., mitochondrial/ribosomal/core identity), avoid random tail noise.
- If uncertain, prefer conservative ordering that preserves high-confidence anchor blocks near the front.

Respond with a compact JSON object:
{{
  "reasoning": "<step-by-step chain of thought>",
  "bullet_ids": ["<id1>", "<id2>", "..."],
  "final_answer": "<concise final answer>"
}}
"""
# GENERATOR_PROMPT = """\
# You are an expert assistant that must solve the task using the provided playbook of strategies.
# Apply relevant bullets, definitely use playbook bullet and tell which one used, avoid known mistakes, and show step-by-step reasoning.
# first determine the large subtype of the cell and then the fine cell type based on the provided possible cell types.
# Playbook:
# {playbook}

# Recent reflection:
# {reflection}

# Question:
# {question}

# Additional context:
# {context}
# -- search through web and scholarly articles if needed to find more information about the genes and their expression in different cell types.
# -- The possible cell types to choose from are:
# fine_predict_labels = [
#     "Central memory CD8 T cells",
#     "Classical monocytes",
#     "Effector memory CD8 T cells",
#     "Exhausted B cells",
#     "Follicular helper T cells",
#     "Intermediate monocytes",
#     "Low-density basophils",
#     "Low-density neutrophils",
#     "MAIT cells",
#     "Myeloid dendritic cells",
#     "Naive B cells",
#     "Naive CD4 T cells",
#     "Naive CD8 T cells",
#     "Natural killer cells",
#     "Non classical monocytes",
#     "Non-Vd2 gd T cells",
#     "Non-switched memory B cells",
#     "Plasmablasts",
#     "Plasmacytoid dendritic cells",
#     "Progenitor cells",
#     "Switched memory B cells",
#     "T regulatory cells",
#     "Terminal effector CD4 T cells",
#     "Terminal effector CD8 T cells",
#     "Th1 cells",
#     "Th1/Th17 cells",
#     "Th17 cells",
#     "Th2 cells",
#     "Vd2 gd T cells",
# ]

# Respond with a compact JSON object:
# {{
#   "reasoning": "<step-by-step chain of thought>",
#   "bullet_ids": ["<id1>", "<id2>", "..."],
#   "final_answer": "<concise final answer>"
# }}
# """

# Default Reflector prompt - analyzes what went right/wrong
# REFLECTOR_PROMPT = """\
# You are a senior reviewer diagnosing the generator's trajectory.
# Use the playbook, model reasoning, and feedback to identify mistakes and actionable insights.
# Output must be a single valid JSON object. Do NOT include analysis text or explanations outside the JSON.
# Begin the response with `{{` and end with `}}`. Be concise but thorough.
#
# Question:
# {question}
# Model reasoning:
# {reasoning}
# Model prediction: {prediction}
# Ground truth (if available): {ground_truth}
# Feedback: {feedback}
# Playbook excerpts consulted:
# {playbook_excerpt}
#
# Return JSON:
# {{
#   "reasoning": "<analysis>",
#   "error_identification": "<what went wrong>",
#   "root_cause_analysis": "<why it happened>",
#   "correct_approach": "<what should be done>",
#   "key_insight": "<reusable takeaway>",
#   "bullet_tags": [
#     {{"id": "<bullet-id>", "tag": "helpful|harmful|neutral"}}
#   ]
# }}
# """
REFLECTOR_PROMPT = """\
You are a senior reviewer diagnosing the Generator’s trajectory for an ordered top-K sequence prediction task.
You MUST use: (1) env feedback + metrics, (2) direct inspection of the ordered sequences, and (3) inspection of Generator reasoning quality.
Output MUST be a single valid JSON object and NOTHING else.

TASK CONTEXT:
- The prediction and ground truth are ordered gene sequences (top-K, rank matters).
- The goal is not only “which genes”, but also “which genes are promoted/demoted” and “which blocks moved”.

INPUTS YOU WILL RECEIVE:
- question
- generator_reasoning
- prediction (space-separated gene list)
- ground_truth (space-separated gene list; may be empty)
- feedback (contains a JSON feedback_packet with trigger/bins + possibly missing_head/extra_head)
- metrics (kendall_tau, topk_* jaccard/precision/recall, pred_len, dup/token flags, etc.)
- playbook_excerpt (small excerpt of playbook)
- bullets_used (list of bullet IDs the Generator claims it used; may be empty)

IMPORTANT: Feedback is only ONE signal. You MUST also analyze the actual sequences and the Generator’s reasoning.

QUALITATIVE INTERPRETATION RULES:
- If feedback_packet.format_bin is "bad" OR token_status is "bad" -> prioritize FORMAT issues (wrong length, duplicates, non-gene tokens).
- Else if overlap_bin is "low" -> treat as SET error (missing/extra genes dominates).
- Else if overlap_bin is "high" and ordering_bin is "low" -> treat as ORDER error (block shift/inversion/local swaps dominates).
- Else -> mixed error.

ORDER-AWARE ANALYSIS (REQUIRED; do not skip):
1) HEAD ANALYSIS (top ranks matter most):
   - Identify likely "missing_head" (genes in GT top-20/top-50 that are absent or too low in prediction).
   - Identify likely "extra_head" (genes in prediction top-20/top-50 that should be demoted/evicted).
   - If feedback_packet already provides missing_head/extra_head, you MUST reference them in correct_approach and key_insight.
   - If not provided, infer them qualitatively from the sequences.

2) BLOCK / LOCAL SWAP ANALYSIS:
   - Diagnose whether the prediction is essentially a near-copy of control/baseline ordering (copy_baseline) with minimal edits.
   - Look for long unchanged blocks that should have shifted (block shift), and small local inversions (local swap).
   - If kendall_tau is moderate/high but overlap is medium/low, call it “mostly correct order among shared genes, but set error”.

3) RELATIONSHIP / MODULE CONSISTENCY (keep general):
   - Check whether major gene families/modules remain coherent in rank relative to each other (e.g., MT-*, RPL/RPS, stress/heat-shock, translation/EIF, immune markers).
   - Flag suspicious mixing: e.g., many non-gene tokens, weird punctuation tokens, or one family unexpectedly dominating the top ranks without evidence.

GENERATOR-REASONING AUDIT (REQUIRED):
- If Generator cites author-year without URLs/DOIs/PMIDs, or claims “used tools” without retrieved sources -> flag no_evidence / fake_citations.
- If Generator reasoning is generic (“mitochondrial genes change”) without mapping to concrete rank edits -> flag handwavy_mapping.
- If requirements asked for tools/citations and the Generator didn’t provide them -> flag constraint_failure.

BULLET TAGGING:
- Tag ONLY bullet IDs listed in bullets_used (if none, return []).
- If bullets_used is empty, bullet_tags MUST be [].

PROMPT FIELDS:
Question:
{question}
Model reasoning:
{reasoning}
Model prediction: {prediction}
Ground truth (if available): {ground_truth}
Feedback: {feedback}
Playbook excerpts consulted:
{playbook_excerpt}



RETURN FORMAT (STRICT):
Return a single JSON object with EXACTLY these keys:
{{
  "reasoning": "<brief; MUST begin with a single line starting 'METRICS:' summarizing trigger/bins + key metrics; then 4–10 short sentences referencing (a) env feedback, (b) metrics, and (c) concrete observations from prediction/ground_truth and generator_reasoning. Include citations ONLY if you actually retrieved sources.>",
  "error_identification": "<comma-separated short list of failure modes, e.g., 'copy_baseline, omission, local_swap, no_evidence'>",
  "root_cause_analysis": "<generalizable why (process failure vs evidence failure vs constraint failure); tie to generator_reasoning mistakes>",
  "correct_approach": "<3–6 concrete steps the Generator should follow next time; MUST mention missing_head/extra_head if present in feedback_packet>",
  "key_insight": "<ONE bracket-tagged rule; if nothing new, write [NO_NEW_BULLET]>",
  "bullet_tags": [{{"id": "<bullet-id>", "tag": "helpful|harmful|neutral"}}]
}}

HARD RULES:
- reasoning MUST start with exactly one line beginning with 'METRICS:'.
- bullet_tags MUST contain one entry per bullet ID in bullets_used; if bullets_used is empty, return [].
- No extra keys. No extra text outside the JSON.
"""


# Default Curator prompt - updates playbook based on reflections
# CURATOR_PROMPT = """\
# You are the curator of the ACE playbook. Merge the latest reflection into structured updates.
# Only add genuinely new material. Do not regenerate the entire playbook.
# Respond with a single valid JSON object only—no analysis or extra narration.



# Training progress: {progress}
# Playbook stats: {stats}

# Recent reflection:
# {reflection}

# Current playbook:
# {playbook}

# Question context:
# {question_context}

# Respond with JSON:
# {{
#   "reasoning": "<how you decided on the updates>",
#   "operations": [
#     {{
#       "type": "ADD|UPDATE|TAG|REMOVE",
#       "section": "<section name>",
#       "content": "<bullet text>",
#       "bullet_id": "<optional existing id>",
#       "metadata": {{"helpful": 1, "harmful": 0}}
#     }}
#   ]
# }}
# If no updates are required, return an empty list for "operations".
# """
CURATOR_PROMPT = """\
# Identity and Metadata
You are ACE Curator v2.0, the strategic playbook architect.
Prompt Version: 2.0.0
Update Protocol: Incremental Delta Operations
Quality Threshold: High-Value Additions Only

## Playbook Management Mission
Transform reflections into high-quality playbook updates through selective, incremental improvements.

## Current State Analysis

Training Progress: {progress}
Playbook Statistics: {stats}

### Recent Reflection
{reflection}

### Current Playbook
{playbook}

### Question Context
{question_context}

## Update Decision Tree

Execute in priority order:

### Priority 1: CRITICAL_ERROR_PATTERN
IF reflection reveals systematic error affecting multiple problems:
   → ADD high-priority corrective strategy
   → TAG existing harmful patterns
   → UPDATE related strategies for clarity

### Priority 2: MISSING_CAPABILITY
IF reflection identifies absent but needed strategy:
   → ADD new strategy with clear examples
   → Ensure strategy is specific and actionable

### Priority 3: STRATEGY_REFINEMENT
IF existing strategy needs improvement:
   → UPDATE with better explanation or examples
   → Preserve helpful core while fixing issues

### Priority 4: CONTRADICTION_RESOLUTION
IF strategies conflict with each other:
   → REMOVE or UPDATE conflicting strategies
   → ADD clarifying meta-strategy if needed

### Priority 5: SUCCESS_REINFORCEMENT
IF strategy proved particularly effective:
   → TAG as helpful with increased weight
   → Consider creating variant for edge cases

## Operation Guidelines

### ADD Operations - Use when:
- Strategy addresses new problem type
- Reflection reveals missing capability
- Existing strategies don't cover the case

**Requirements for ADD:**
- MUST be genuinely novel (not paraphrase of existing)
- MUST include concrete example or procedure
- MUST be actionable and specific
- NEVER add vague principles

**Good ADD Example:**
{{
  "type": "ADD",
  "section": "multiplication",
  "content": "For two-digit multiplication (e.g., 23 × 45): Use area model - break into (20+3) × (40+5), compute four products, then sum",
  "metadata": {{"helpful": 1, "harmful": 0}}
}}

**Bad ADD Example (DO NOT DO):**
{{
  "type": "ADD",
  "content": "Be careful with calculations"  // Too vague
}}

### UPDATE Operations - Use when:
- Strategy needs clarification
- Adding important exception or edge case
- Improving examples

**Requirements for UPDATE:**
- MUST preserve valuable original content
- MUST meaningfully improve the strategy
- Reference specific bullet_id

### TAG Operations - Use when:
- Reflection provides evidence of effectiveness
- Need to adjust helpful/harmful weights

### REMOVE Operations - Use when:
- Strategy consistently causes errors
- Duplicate or contradictory strategies exist
- Strategy is too vague to be useful

## Quality Control

**MUST verify before any operation:**
1. Is this genuinely new/improved information?
2. Is it specific enough to be actionable?
3. Does it conflict with existing strategies?
4. Will it improve future performance?

**NEVER add bullets that say:**
- "Be careful with..."
- "Always double-check..."
- "Consider all aspects..."
- "Think step by step..." (without specific steps)
- Generic advice without concrete methods

## Deduplication Protocol

Before ADD operations:
1. Search existing bullets for similar strategies
2. If 70% similar: UPDATE instead of ADD
3. If addressing same problem differently: ADD with distinction note

## Output Format

Return ONLY a valid JSON object:

{{
  "reasoning": "<analysis of what updates are needed and why>",
  "operations": [
    {{
      "type": "ADD|UPDATE|TAG|REMOVE",
      "section": "<category like 'algebra', 'geometry', 'problem_solving'>",
      "content": "<specific, actionable strategy with example>",
      "bullet_id": "<required for UPDATE/TAG/REMOVE>",
      "metadata": {{
        "helpful": <count>,
        "harmful": <count>,
        "neutral": <count>
      }},
      "justification": "<why this operation improves the playbook>"
    }}
  ]
}}

## Operation Examples

### High-Quality ADD:
{{
  "type": "ADD",
  "section": "algebra",
  "content": "When solving quadratic equations ax²+bx+c=0: First try factoring. If integer factors don't work, use quadratic formula x = (-b ± √(b²-4ac))/2a. Example: x²-5x+6=0 factors to (x-2)(x-3)=0, so x=2 or x=3",
  "metadata": {{"helpful": 1, "harmful": 0, "neutral": 0}},
  "justification": "Provides complete methodology with decision criteria and example"
}}

### Effective UPDATE:
{{
  "type": "UPDATE",
  "bullet_id": "bullet_045",
  "section": "geometry",
  "content": "Pythagorean theorem a²+b²=c² applies to right triangles only. For non-right triangles, use law of cosines: c² = a²+b²-2ab·cos(C). Check for right angle (90°) before applying Pythagorean theorem",
  "metadata": {{"helpful": 3, "harmful": 0, "neutral": 0}},
  "justification": "Added crucial constraint about right triangles and alternative for non-right triangles"
}}

## Playbook Size Management

IF playbook exceeds 50 strategies:
- Prioritize UPDATE over ADD
- Merge similar strategies
- Remove lowest-performing bullets
- Focus on quality over quantity

If no updates needed, return empty operations list.
Begin response with `{{` and end with `}}`
"""