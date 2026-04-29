# KB-VQA ReAct Agent

This project implements a small Python agent for knowledge-based visual question answering.
It accepts an image and a question, uses a ReAct loop to inspect the image, searches a local
JSON or CSV knowledge base, and calls an OpenAI-compatible chat completions API to produce
the final answer.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Configure

```bash
export KBVQA_API_KEY="your-api-key"
export KBVQA_BASE_URL="https://api.openai.com/v1"
export KBVQA_MODEL="your-vision-capable-model"
```

`KBVQA_MODEL` must point to a model that accepts image input.

## Knowledge Base Format

JSON can be a list of entries or an object with an `entries`, `items`, or `documents` list.

```json
[
  {
    "id": "apple",
    "title": "Apple",
    "text": "Apples are round edible fruits that can be red, green, or yellow.",
    "metadata": {"source": "sample"}
  }
]
```

CSV requires `id`, `title`, and `text` columns. A `metadata` column is optional and may
contain a JSON object string. Extra columns are preserved as metadata.

## CLI

```bash
python -m kbvqa answer \
  --image path/to/image.jpg \
  --question "What object is shown, and what is it commonly used for?" \
  --kb data/sample_kb.json \
  --show-trace
```

Use `--json` to emit the full result as JSON.

## Python SDK

```python
from kbvqa import KBVQAAgent

agent = KBVQAAgent.from_env("data/sample_kb.json")
result = agent.answer("path/to/image.jpg", "What object is shown?")
print(result["answer"])
```

## Tests

The default test suite uses only the standard library and fake LLM clients.

```bash
python3 -m unittest discover -s tests
```

Optional real API integration tests are skipped unless `KBVQA_RUN_INTEGRATION=1`,
`KBVQA_API_KEY`, and `KBVQA_MODEL` are set.
