"""Command-line interface for the KB-VQA agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .agent import KBVQAAgent
from .image import ImageInputError
from .kb import KnowledgeBaseError
from .llm import LLMConfigurationError, LLMError


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "answer":
        return _answer(args)
    parser.print_help()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kbvqa",
        description="Run a ReAct KB-VQA agent over an image, question, and local KB.",
    )
    subparsers = parser.add_subparsers(dest="command")

    answer = subparsers.add_parser("answer", help="answer one visual question")
    answer.add_argument("--image", required=True, help="path to an input image")
    answer.add_argument("--question", required=True, help="question about the image")
    answer.add_argument("--kb", required=True, help="path to a JSON or CSV knowledge base")
    answer.add_argument("--max-steps", type=int, default=4, help="maximum ReAct steps")
    answer.add_argument("--top-k", type=int, default=5, help="number of KB entries to return")
    answer.add_argument("--show-trace", action="store_true", help="print ReAct trace")
    answer.add_argument("--json", action="store_true", help="emit full result as JSON")
    return parser


def _answer(args: argparse.Namespace) -> int:
    try:
        agent = KBVQAAgent.from_env(
            Path(args.kb),
            max_steps=args.max_steps,
            top_k=args.top_k,
        )
        result = agent.answer(Path(args.image), args.question)
    except (LLMConfigurationError, KnowledgeBaseError, ImageInputError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        print(
            "Check that KBVQA_MODEL supports image input and that KBVQA_BASE_URL is OpenAI-compatible.",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    if result["evidence"]:
        print("Evidence:")
        for item in result["evidence"]:
            print(f"- {item['id']}: {item['title']} (score={item['score']:.4f})")
    else:
        print("Evidence: none")

    if args.show_trace:
        print("Trace:")
        for step in result["trace"]:
            print(f"- Step {step['step']} | {step['action']}")
            print(f"  Thought: {step['thought']}")
            if step["action_input"]:
                print(
                    "  Action input: "
                    + json.dumps(step["action_input"], ensure_ascii=False)
                )
            print(f"  Observation: {step['observation']}")
    return 0
