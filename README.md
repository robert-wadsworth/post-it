# Post-It

An autonomous posting agent that generates social media content using AI.

## Setup

1. Install dependencies:

```bash
poetry install
```

2. Create a `.env` file in the root directory and add your OpenAI API key:

```text
OPENAI_API_KEY=your-api-key-here
```

## Usage

Run the application:

```bash
poetry run python src/main.py
```

## Features

- Generate social media posts using GPT-4o-mini and DALL-E
- Built with LangChain and LangGraph
