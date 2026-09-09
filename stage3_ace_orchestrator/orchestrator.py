from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
try:
    from langchain_litellm import ChatLiteLLM
except ImportError:
    ChatLiteLLM = None

from .config import find_repo_root, load_repo_dotenv
from .engine import Stage3ExpertRuntime
from .tools import build_stage3_tools

REPO_ROOT = find_repo_root(Path(__file__))
load_repo_dotenv(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ace import Generator  # noqa: E402
from ace.Agent import LangChainAgentLLMClient  # noqa: E402


DEFAULT_SYSTEM_PROMPT = """
You are a single-cell biology orchestration agent.

You have access to local specialist tools, each backed by a trained tissue/domain expert model.
Do not assume in advance which experts should be called.
Infer likely experts from the question, ordered genes, grounded context, expert descriptions, and the playbook.
Prefer the most directly relevant expert(s) and avoid spending calls on weakly matched experts unless they are needed to resolve ambiguity.
If the task depends on tissue-specific annotation or cross-tissue discrimination, use the expert catalog and call one or multiple relevant expert tools before answering.
When the evidence is ambiguous or multiple tissues/cell identities are plausible, compare outputs from multiple relevant experts and synthesize them before answering.
If the question contains multiple cell/profile gene lists, separate the profiles before tool use and never reuse one profile's genes for another.
For multi-cell/profile questions, every expert-tool call must include exactly one `profile_id` and must leave `genes_csv` empty; the runtime will inject the canonical 200-gene list. Calls without `profile_id` or with `genes_csv` are invalid and will fail.
Treat tool use as internal reasoning: do not tell the user that an expert, tool, model, or routing system was used.
Return one natural-language final answer grounded in marker evidence and explicit exclusions.

When the user prompt asks for JSON, respond with a single JSON object with keys:
reasoning, bullet_ids, bullet_ids_used_from_playbook, answer_fields, final_answer.

Set final_answer to a grounded prose answer whenever the question asks for a biology answer.
For structured biology tasks, put machine-readable values in answer_fields, then write final_answer like a direct human-facing answer: concise paragraphs that name the best label, supporting genes, excluded alternatives/negative markers, tissue context when relevant, confidence, and abstention only if needed.
Do not include extra text outside the top-level JSON object.
""".strip()


STAGE3_GENERATOR_PROMPT = """\
You are an expert assistant for single-cell biology orchestration.
Use the playbook when it contains useful routing or synthesis guidance.
Use the recent reflection to avoid repeating mistakes.

Playbook (bullets):
{playbook}

Recent reflection (most recent reflector guidance):
{reflection}

Question:
{question}

Additional context:
{context}

Tool-use policy:
- Before choosing tools or writing the answer, identify which playbook bullets apply to routing and which apply to synthesis.
- Do not rely on a priori expert hints; infer the best expert tools from the question, genes, grounded context, expert descriptions, and playbook bullets.
- Treat choosing the right expert model(s) as a primary objective: prefer the most directly tissue/task-matched expert first, and add more experts only when they materially reduce ambiguity.
- If routing is uncertain, use `list_stage3_experts` before choosing an expert.
- If the task is cell annotation, tissue inference, differential diagnosis, cross-expert annotation, or OOD abstention, prefer one or multiple relevant local expert tools before answering.
- If there is doubt, conflicting evidence, cross-tissue ambiguity, or several plausible candidate labels, call multiple relevant experts and synthesize their outputs.
- For multi-cell/profile questions, handle each profile separately: extract its genes, call experts with that profile's `profile_id`, then compare or synthesize only after per-profile evidence is clear.
- If `Additional context` lists available profile IDs, expert-tool calls must use one of those exact `profile_id` values.
- For multi-cell/profile questions, never make an expert-tool call without `profile_id`, even if genes appear in the question text.
- For multi-cell/profile questions, do not send `genes_csv` to expert tools. The runtime will supply the canonical 200-gene list from the active sample, and any `genes_csv` will be rejected.
- If a tool returns a JSON payload with `tool_error.recoverable=true`, do not treat it as evidence. Correct the call shape immediately and retry. For multi-cell/profile errors, retry with exactly one valid `profile_id` from `available_profile_ids` and leave `genes_csv` empty.
- Local Stage-2 experts are always prompted with the trained 200-gene C2S cell-type template, regardless of the Stage3 question type.
- Expert outputs may include LABEL/FINAL, NEGATIVE_MARKERS, EVIDENCE, CONTEXT, and CONFIDENCE. Ignore POSITIVE_MARKERS if present; tool payloads intentionally omit them. Use the remaining structured fields when present; if an expert returns only a label, derive supporting_genes and negative_markers from the ordered input genes and grounded context.
- If one or more expert tools are called, treat their returned label/evidence/negative-marker/confidence fields as the primary substrate for answer_fields and final_answer. Do not bypass called expert outputs or invent a final answer independently of them.
- answer_fields must be the normalized synthesis of the called expert outputs plus grounded input-gene checks. final_answer must then be written from answer_fields, not as a separate unsupported answer.
- Pass through the ordered genes, task family, task type, candidate labels, and grounded context when calling a tool. For single-cell questions, ordered genes may be passed directly. For multi-cell questions, pass profile_id plus the other metadata, do not send genes_csv, and do not mix markers across profiles.
- Choose final_label from candidate labels unless abstain is true or the context explicitly permits an outside label.
- Use supporting_genes only from the ordered input genes or expert evidence that appears in the ordered input genes.
- Use negative_markers to explain which alternatives are not supported. If negative markers are provided in context or by experts, cite the relevant ones in final_answer.
- expert_models_used must list exactly the expert tools that were actually called.
- Treat expert/tool calls as internal checks. Do not mention expert names, tool names, model names, routing, or that an expert was consulted in reasoning or final_answer.
- Write final_answer in natural language, as if answering the biology question directly. Do not format it as a field-by-field schema and do not use labels like "Final label:", "Supporting genes:", "Negative markers:", "Confidence:", or "Abstain:" unless the user explicitly asks for that exact format.
- The rationale must briefly connect supporting gene evidence, negative-marker exclusions, and the final label/confidence.

Return a compact JSON object:
{{
    "reasoning": "<brief reasoning>",
    "bullet_ids": ["<id1>", "<id2>"],
    "bullet_ids_used_from_playbook": ["<id1>", "<id2>"],
    "answer_fields": {{
        "final_label": "<best candidate label, or unknown if abstaining>",
        "alternative_labels": ["<other plausible or rejected labels>"],
        "predicted_tissue": "<tissue/context, or unknown if abstaining>",
        "supporting_genes": ["<input gene>", "<input gene>"],
        "negative_markers": ["<marker used to rule out alternatives>"],
        "rationale": "<compact evidence-based explanation using supporting genes and negative markers>",
        "expert_models_used": ["<internal expert tool names actually called>"],
        "confidence": "<high|moderate|low>",
        "abstain": <true|false>
    }},
    "final_answer": "<short prose answer based on answer_fields and evidence>"
}}

When a structured answer schema is requested, answer_fields must contain every required schema field. final_answer must still be prose and should not expose internal expert/tool names.
"""


def build_stage3_system_prompt(runtime: Stage3ExpertRuntime, base_prompt: str | None = None) -> str:
    expert_lines = runtime.format_expert_catalog(limit=100)
    prompt = base_prompt or DEFAULT_SYSTEM_PROMPT
    return prompt + "\n\nAvailable local experts:\n" + expert_lines


def _uses_litellm_backend(model: str) -> bool:
    model_name = model.lower()
    return "claude" in model_name or model_name.startswith("anthropic/")


def _normalize_litellm_model(model: str) -> str:
    if "claude" in model.lower() and "/" not in model:
        return f"anthropic/{model}"
    return model


def _build_stage3_llm(
    model: str,
    *,
    temperature: float | None = None,
    seed: int | None = None,
):
    llm_kwargs: dict[str, Any] = {}
    if temperature is not None:
        llm_kwargs["temperature"] = temperature

    if _uses_litellm_backend(model):
        if ChatLiteLLM is None:
            raise ImportError(
                "Claude stage3 orchestration requires langchain-litellm. "
                "Install it with `pip install langchain-litellm` or from the benchmark requirements."
            )
        if seed is not None:
            llm_kwargs["seed"] = seed
        return ChatLiteLLM(model=_normalize_litellm_model(model), **llm_kwargs)

    if seed is not None:
        llm_kwargs["model_kwargs"] = {"seed": seed}
    return ChatOpenAI(model=model, **llm_kwargs)


def build_stage3_langchain_agent(
    runtime: Stage3ExpertRuntime,
    *,
    model: str = "gpt-5.1",
    system_prompt: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
):
    llm = _build_stage3_llm(model, temperature=temperature, seed=seed)
    tools = build_stage3_tools(runtime)
    return create_agent(
        llm,
        tools=tools,
        system_prompt=build_stage3_system_prompt(runtime, base_prompt=system_prompt),
    )


def build_stage3_generator(
    runtime: Stage3ExpertRuntime,
    *,
    model: str = "gpt-5.1",
    system_prompt: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> Generator:
    agent = build_stage3_langchain_agent(
        runtime,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        seed=seed,
    )
    return Generator(
        LangChainAgentLLMClient(agent),
        prompt_template=STAGE3_GENERATOR_PROMPT,
    )
