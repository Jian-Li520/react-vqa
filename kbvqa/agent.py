"""ReAct-style agent for knowledge-based visual question answering."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .image import image_to_data_url
from .kb import KnowledgeBase
from .llm import LLMClient
from .retriever import Evidence, KBRetriever


VALID_ACTIONS = {"inspect_image", "search_kb", "final_answer"}
CONFIDENCE_LEVELS = {"low", "medium", "high"}


class ChatModel(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        ...


class ActionParseError(ValueError):
    """Raised when the ReAct controller returns an invalid action."""


@dataclass(frozen=True)
class ReActAction:
    thought: str
    action: str
    action_input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceStep:
    step: int
    thought: str
    action: str
    action_input: dict[str, Any]
    observation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }


@dataclass(frozen=True)
class AgentResult:
    answer: str
    confidence: str
    evidence: list[Evidence]
    trace: list[TraceStep]

    def to_dict(self, include_evidence_text: bool = False) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "evidence": [
                item.to_dict(include_text=include_evidence_text)
                for item in self.evidence
            ],
            "trace": [step.to_dict() for step in self.trace],
        }


class ReActAgent:
    """Coordinates visual inspection, KB search, and final answer synthesis."""

    def __init__(
        self,
        llm: ChatModel,
        retriever: KBRetriever,
        *,
        max_steps: int = 4,
        top_k: int = 5,
    ):
        self.llm = llm
        self.retriever = retriever
        self.max_steps = max_steps
        self.top_k = top_k

    def answer(self, image_path: str | Path, question: str) -> AgentResult:
        if not question.strip():
            raise ValueError("Question must be non-empty.")

        data_url = image_to_data_url(image_path)
        trace: list[TraceStep] = []
        observations: list[str] = []
        evidence: list[Evidence] = []

        for step_number in range(1, self.max_steps + 1):
            try:
                action = self._decide_next_action(question, observations, evidence, trace)
            except ActionParseError as exc:
                trace.append(
                    TraceStep(
                        step=step_number,
                        thought="Controller returned an invalid action.",
                        action="invalid_action",
                        action_input={},
                        observation=str(exc),
                    )
                )
                observations.append(f"Invalid controller action: {exc}")
                continue

            if action.action == "inspect_image":
                observation = self._inspect_image(data_url, question)
                observations.append(f"Image observation: {observation}")
            elif action.action == "search_kb":
                query = _string_action_input(
                    action.action_input,
                    "query",
                    fallback=f"{question}\n" + "\n".join(observations),
                )
                found = self.retriever.search(query, top_k=self.top_k)
                evidence = _merge_evidence(evidence, found)
                observation = _format_search_observation(found)
                observations.append(f"KB search: {observation}")
            elif action.action == "final_answer":
                answer, confidence = self._final_answer(
                    data_url=data_url,
                    question=question,
                    observations=observations,
                    evidence=evidence,
                )
                trace.append(
                    TraceStep(
                        step=step_number,
                        thought=action.thought,
                        action=action.action,
                        action_input=action.action_input,
                        observation=f"Final answer generated with confidence={confidence}.",
                    )
                )
                return AgentResult(
                    answer=answer,
                    confidence=confidence,
                    evidence=evidence,
                    trace=trace,
                )
            else:
                observation = f"Unknown action ignored: {action.action}"

            trace.append(
                TraceStep(
                    step=step_number,
                    thought=action.thought,
                    action=action.action,
                    action_input=action.action_input,
                    observation=observation,
                )
            )

        answer, confidence = self._final_answer(
            data_url=data_url,
            question=question,
            observations=observations,
            evidence=evidence,
        )
        return AgentResult(
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            trace=trace
            + [
                TraceStep(
                    step=self.max_steps + 1,
                    thought="Reached max_steps; synthesizing final answer.",
                    action="final_answer",
                    action_input={},
                    observation=f"Final answer generated with confidence={confidence}.",
                )
            ],
        )

    def _decide_next_action(
        self,
        question: str,
        observations: list[str],
        evidence: list[Evidence],
        trace: list[TraceStep],
    ) -> ReActAction:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the controller for a ReAct KB-VQA agent. "
                    "Pick exactly one next action. Valid actions are "
                    "inspect_image, search_kb, and final_answer. "
                    "Use inspect_image when visual context is missing, search_kb "
                    "when external knowledge may help, and final_answer only when "
                    "there is enough visual and knowledge context. Return JSON only "
                    "with keys thought, action, and action_input."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "observations": observations[-5:],
                        "evidence": [
                            {"id": item.id, "title": item.title, "score": item.score}
                            for item in evidence[: self.top_k]
                        ],
                        "previous_actions": [
                            {
                                "step": step.step,
                                "action": step.action,
                                "observation": step.observation[:500],
                            }
                            for step in trace[-5:]
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        raw = self.llm.chat(
            messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return parse_react_action(raw)

    def _inspect_image(self, data_url: str, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You inspect images for a KB-VQA system. Describe only visible "
                    "objects, scene context, readable text, attributes, and details "
                    "that may help answer the user's question. Do not guess external "
                    "facts yet."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Question: {question}\nProvide a concise visual observation.",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        return self.llm.chat(messages, temperature=0.1, max_tokens=500).strip()

    def _final_answer(
        self,
        *,
        data_url: str,
        question: str,
        observations: list[str],
        evidence: list[Evidence],
    ) -> tuple[str, str]:
        evidence_payload = [
            {
                "id": item.id,
                "title": item.title,
                "text": item.text,
                "metadata": item.metadata,
                "score": item.score,
            }
            for item in evidence[: self.top_k]
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer KB-VQA questions using the image, visual observations, "
                    "and retrieved knowledge. If evidence is weak or missing, say so "
                    "briefly and lower confidence. Return JSON only with keys answer "
                    "and confidence, where confidence is low, medium, or high."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "question": question,
                                "observations": observations,
                                "knowledge_evidence": evidence_payload,
                            },
                            ensure_ascii=False,
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        raw = self.llm.chat(
            messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return parse_final_answer(raw)


class KBVQAAgent:
    """SDK facade for creating and running the KB-VQA ReAct agent."""

    def __init__(self, react_agent: ReActAgent):
        self.react_agent = react_agent

    @classmethod
    def from_env(
        cls,
        kb_path: str | Path,
        *,
        max_steps: int = 4,
        top_k: int = 5,
    ) -> "KBVQAAgent":
        kb = KnowledgeBase.load(kb_path)
        retriever = KBRetriever(kb)
        llm = LLMClient.from_env()
        return cls(ReActAgent(llm, retriever, max_steps=max_steps, top_k=top_k))

    def answer(self, image_path: str | Path, question: str) -> dict[str, Any]:
        return self.react_agent.answer(image_path, question).to_dict()


def parse_react_action(text: str) -> ReActAction:
    payload = _loads_json_object(text)
    thought = str(payload.get("thought", "")).strip()
    action = str(payload.get("action", "")).strip()
    action_input = payload.get("action_input", {})
    if not thought:
        raise ActionParseError("Action JSON is missing a non-empty 'thought'.")
    if action not in VALID_ACTIONS:
        raise ActionParseError(
            f"Invalid action {action!r}. Valid actions: {sorted(VALID_ACTIONS)}"
        )
    if action_input is None:
        action_input = {}
    if isinstance(action_input, str):
        action_input = {"query": action_input}
    if not isinstance(action_input, dict):
        raise ActionParseError("'action_input' must be an object or string.")
    return ReActAction(thought=thought, action=action, action_input=action_input)


def parse_final_answer(text: str) -> tuple[str, str]:
    try:
        payload = _loads_json_object(text)
    except ActionParseError:
        return text.strip(), "medium"
    answer = str(payload.get("answer", "")).strip()
    confidence = str(payload.get("confidence", "medium")).strip().lower()
    if not answer:
        answer = text.strip()
    if confidence not in CONFIDENCE_LEVELS:
        confidence = "medium"
    return answer, confidence


def _loads_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(text.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ActionParseError(f"Expected a JSON object, got: {text[:200]!r}")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"Could not parse JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ActionParseError("Expected a JSON object.")
    return payload


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def _string_action_input(
    action_input: dict[str, Any],
    key: str,
    *,
    fallback: str,
) -> str:
    value = action_input.get(key)
    if value is None:
        return fallback
    value = str(value).strip()
    return value or fallback


def _format_search_observation(evidence: list[Evidence]) -> str:
    if not evidence:
        return "No relevant knowledge entries found."
    return "\n".join(
        f"[{item.id}] {item.title} (score={item.score:.4f}): {item.text[:300]}"
        for item in evidence
    )


def _merge_evidence(existing: list[Evidence], new_items: list[Evidence]) -> list[Evidence]:
    by_id = {item.id: item for item in existing}
    for item in new_items:
        current = by_id.get(item.id)
        if current is None or item.score > current.score:
            by_id[item.id] = item
    return sorted(by_id.values(), key=lambda item: (-item.score, item.id))
