from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kbvqa.agent import ReActAgent, parse_final_answer, parse_react_action
from kbvqa.kb import KnowledgeBase
from kbvqa.retriever import KBRetriever


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
        )
        if not self.responses:
            raise AssertionError("FakeLLM has no response left.")
        return self.responses.pop(0)


class AgentTests(unittest.TestCase):
    def test_parse_react_action_from_json_fence(self):
        action = parse_react_action(
            '```json\n{"thought":"look","action":"inspect_image","action_input":{}}\n```'
        )

        self.assertEqual(action.thought, "look")
        self.assertEqual(action.action, "inspect_image")

    def test_parse_final_answer_falls_back_for_plain_text(self):
        answer, confidence = parse_final_answer("Plain answer")

        self.assertEqual(answer, "Plain answer")
        self.assertEqual(confidence, "medium")

    def test_agent_runs_inspect_search_final_answer(self):
        kb = KnowledgeBase.from_dicts(
            [
                {
                    "id": "apple",
                    "title": "Apple",
                    "text": "Apples are edible fruits that can be red.",
                }
            ]
        )
        fake_llm = FakeLLM(
            [
                '{"thought":"Need visual facts","action":"inspect_image","action_input":{}}',
                "The image shows a red apple.",
                '{"thought":"Need fruit knowledge","action":"search_kb","action_input":{"query":"red apple edible fruit"}}',
                '{"thought":"Ready","action":"final_answer","action_input":{}}',
                '{"answer":"The image likely shows an apple, an edible fruit.","confidence":"high"}',
            ]
        )
        agent = ReActAgent(fake_llm, KBRetriever(kb), max_steps=4, top_k=3)

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "pixel.png"
            image.write_bytes(PNG_1X1)
            result = agent.answer(image, "What fruit is shown?")

        self.assertEqual(result.answer, "The image likely shows an apple, an edible fruit.")
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.evidence[0].id, "apple")
        self.assertEqual([step.action for step in result.trace], ["inspect_image", "search_kb", "final_answer"])
        self.assertEqual(len(fake_llm.calls), 5)

    def test_agent_records_invalid_action_and_recovers(self):
        kb = KnowledgeBase.from_dicts(
            [{"id": "x", "title": "X", "text": "Knowledge text"}]
        )
        fake_llm = FakeLLM(
            [
                '{"thought":"Oops","action":"dance","action_input":{}}',
                '{"thought":"Answer now","action":"final_answer","action_input":{}}',
                '{"answer":"I cannot determine it from the available context.","confidence":"low"}',
            ]
        )
        agent = ReActAgent(fake_llm, KBRetriever(kb), max_steps=3)

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "pixel.png"
            image.write_bytes(PNG_1X1)
            result = agent.answer(image, "What is shown?")

        self.assertEqual(result.trace[0].action, "invalid_action")
        self.assertEqual(result.confidence, "low")


if __name__ == "__main__":
    unittest.main()
