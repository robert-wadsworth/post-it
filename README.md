# Post-It

An autonomous posting agent that generates social media content (text + image) using LangChain, LangGraph, and OpenAI.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

1. Install dependencies and create the virtual environment:

   ```bash
   uv sync
   ```

   This creates `.venv/` in the project root and installs everything from `pyproject.toml`.

2. Copy the example env file and fill in your keys:

   ```bash
   cp .env.example .env
   ```

   Required variables:

   | Variable | Purpose |
   | --- | --- |
   | `OPENAI_API_KEY` | OpenAI API key (used for GPT and DALL-E) |
   | `LANGCHAIN_API_KEY` | LangSmith API key (only needed if tracing is on) |
   | `LANGCHAIN_TRACING_V2` | `true` to enable LangSmith tracing, `false` to disable |
   | `LANGCHAIN_PROJECT` | LangSmith project name |
   | `API_TOKEN` | Bearer token required by the FastAPI `/generate` endpoint (optional in local dev) |

3. In Cursor / VS Code, select the interpreter at `./.venv/bin/python` (Cmd+Shift+P → **Python: Select Interpreter**).

## Usage

### Run the agent directly

```bash
uv run python src/main.py
```

### Run the FastAPI server

```bash
uv run uvicorn api:app --reload --host 0.0.0.0 --port 8080 --app-dir src
```

Then send a request:

```bash
curl -X POST http://localhost:8080/generate \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a post about agentic AI workflows"}'
```

Endpoints:

- `GET /` — service health check
- `GET /health` — Cloud Run health check
- `POST /generate` — generate a post (text + optional image URL)

## Project Structure

```
src/
├── main.py        # LangGraph agent definition + CLI entry point
├── api.py         # FastAPI wrapper exposing the agent over HTTP
├── agents/        # Agent-level orchestration
├── nodes/         # LangGraph node implementations (draft, review, image)
├── schemas/       # Pydantic schemas
└── state/         # Graph state definitions
```

## Development

Lint and format with [Ruff](https://docs.astral.sh/ruff/):

```bash
uv run ruff check .
uv run ruff format .
```

Add a runtime dependency:

```bash
uv add <package>
```

Add a dev-only dependency:

```bash
uv add --dev <package>
```

## Features

- Drafts post text with GPT and iteratively reviews/refines it
- Generates a matching image via DALL-E
- Exposes the agent through a FastAPI HTTP endpoint with bearer-token auth
- Built with LangChain and LangGraph
