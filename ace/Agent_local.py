from typing import Any

from langchain.tools import tool
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI  # OpenAI-compatible LLM; supports local servers
from langchain_tavily import TavilySearch
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from dotenv import load_dotenv
import os

DEFAULT_SYSTEM_PROMPT = """
You are a biology researcher. Use Semantic Scholar for scholarly papers and Tavily for broader web context. Cite sources concisely.
When the user prompt asks for JSON (e.g., from ACE Generator), respond with a single JSON object including keys: reasoning, bullet_ids, bullet_ids_used_from_playbook, final_answer. Do not add extra text outside JSON.
"""

from ace.llm import LLMClient, LLMResponse

load_dotenv()  # Load environment variables from .env file


def require_env(name: str, aliases: list[str] | tuple[str, ...] = ()) -> str:
    """Fetch an env var, trying aliases, or raise with a clear message."""
    for key in (name, *aliases):
        value = os.getenv(key)
        if value:
            return value
    raise RuntimeError(f"Missing required environment variable: {name} (aliases: {aliases})")


def _resolve_openai_env() -> tuple[str | None, str | None, str | None]:
    """Resolve OpenAI-compatible config, allowing local servers without API keys."""
    api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OpenAI_KEY")
    model = os.getenv("OPENAI_MODEL")
    if api_base and not api_key:
        api_key = "EMPTY"
    return api_key, api_base, model


def build_tools() -> list:
    """Create the shared tool set (Tavily + Semantic Scholar)."""
    openai_key, openai_base, _ = _resolve_openai_env()
    if not openai_key and not openai_base:
        openai_key = require_env("OPENAI_API_KEY", ["OpenAI_KEY"])
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key  # Ensure downstream libs see it
    if openai_base:
        os.environ["OPENAI_API_BASE"] = openai_base
        os.environ["OPENAI_BASE_URL"] = openai_base

    require_env("TAVILY_API_KEY")
    web_search = TavilySearch(max_results=5)

    require_env("SEMANTIC_SCHOLAR_API_KEY", ["S2_API_KEY"])
    semantic_scholar = SemanticScholarQueryRun(
        api_wrapper=SemanticScholarAPIWrapper()
    )

    return [web_search, semantic_scholar]


class _ToolLogger:
    """Lightweight proxy to log tool calls and results for debugging.

    Wraps a LangChain tool object and forwards attribute access while
    intercepting callable invocations via `__call__` and `run`.
    """

    def __init__(self, tool):
        self._tool = tool

    def __call__(self, *args, **kwargs):
        try:
            print(f"[ToolLogger] Calling tool {self._tool.__class__.__name__} with args={args} kwargs={kwargs}")
        except Exception:
            print(f"[ToolLogger] Calling tool {self._tool} ...")
        result = None
        try:
            # Prefer `run` if available, otherwise call the tool directly
            if hasattr(self._tool, "run"):
                result = self._tool.run(*args, **kwargs)
            else:
                result = self._tool(*args, **kwargs)
            print(f"[ToolLogger] Result from {self._tool.__class__.__name__}: {repr(result)[:400]}")
        except Exception as exc:  # pragma: no cover - runtime/network errors
            print(f"[ToolLogger] Tool {self._tool.__class__.__name__} raised: {exc}")
            raise
        return result

    # Some tools may be invoked via `run` directly by LangChain; forward that.
    def run(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tool, name)


def build_langchain_agent(
    model: str = "gpt-5.1",
    system_prompt: str | None = None,
    *,
    temperature: float | None = None,
    seed: int | None = None,
):
    """Create the LangChain agent with tools and ACE-friendly system prompt."""
    tools = build_tools()
    _, _, env_model = _resolve_openai_env()
    if env_model:
        model = env_model
    # Instrument tools by wrapping their `run` method to log invocations
    # while keeping the original tool objects (avoids LangChain tool-type checks).
    try:
        for t in tools:
            run_attr = getattr(t, "run", None)
            if callable(run_attr):
                orig_run = run_attr

                def _make_run(orig, tool_obj):
                    def _run(*args, **kwargs):
                        print(f"[ToolLogger] Calling tool {tool_obj.__class__.__name__} with args={args} kwargs={kwargs}")
                        try:
                            result = orig(*args, **kwargs)
                            print(f"[ToolLogger] Result from {tool_obj.__class__.__name__}: {repr(result)[:400]}")
                            return result
                        except Exception as exc:  # pragma: no cover - runtime/network errors
                            print(f"[ToolLogger] Tool {tool_obj.__class__.__name__} raised: {exc}")
                            raise

                    return _run

                try:
                    setattr(t, "run", _make_run(orig_run, t))
                except Exception:
                    # If we can't monkeypatch, ignore and continue with raw tool
                    pass
    except Exception:
        pass
    llm_kwargs: dict[str, Any] = {}
    if temperature is not None:
        llm_kwargs["temperature"] = temperature
    if seed is not None:
        llm_kwargs["model_kwargs"] = {"seed": seed}
    llm = ChatOpenAI(model=model, **llm_kwargs)
    system_prompt_text = system_prompt or DEFAULT_SYSTEM_PROMPT
    return create_agent(llm, tools=tools, system_prompt=system_prompt_text)


class LangChainAgentLLMClient(LLMClient):
    """
    Lightweight adapter so a LangChain agent can be used as an ACE LLMClient.

    The Generator will call `complete(prompt)`, which we forward into the agent
    as a single HumanMessage. The agent's final message content is returned as
    the LLMResponse text so ACE can parse the expected JSON.
    """

    def __init__(self, agent: Any) -> None:
        super().__init__(model="langchain-agent")
        self.agent = agent

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        response = self.agent.invoke({"messages": [HumanMessage(content=prompt)]})
        messages = response.get("messages") if isinstance(response, dict) else None
        final_msg = messages[-1] if messages else None
        content = getattr(final_msg, "content", "") if final_msg else ""
        return LLMResponse(text=content, raw=response)


if __name__ == "__main__":
    # Example: run the agent directly
    agent = build_langchain_agent()
    # response = agent.invoke({
    #     "messages": [
    #         HumanMessage(content="what are the fine grained gene makers for th2 t cells? give me your step by step reasoning.")
    #     ]
    # })
    # final_msg = response["messages"][-1]
    # print("Agent final message:", final_msg.content)

    # Example: plug into ACE Generator with minimal code changes elsewhere
    try:
        from ace import Generator, Playbook

        generator = Generator(LangChainAgentLLMClient(agent))
        playbook = Playbook()
        gen_output = generator.generate(
            question="what are the fine grained gene makers for th2 t cells. give me you step by step reasoning with references?",
            context="",
            playbook=playbook,
        )
        print("\nACE Generator final_answer:", gen_output.final_answer)
        print("\nACE Generator reasoning:", gen_output.reasoning)

    except Exception as exc:  # pragma: no cover - optional demo
        print("ACE integration demo failed:", exc)
