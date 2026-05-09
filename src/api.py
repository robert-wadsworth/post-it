import logging
from typing import Optional
from uuid import uuid4

import anyio
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from main import agent  # noqa: E402

app = FastAPI(title="Post-It API", version="0.1.0")
logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)


class GenerateResponse(BaseModel):
    text: str
    image_url: Optional[str] = None
    llm_calls: int
    request_id: str


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_post(request: GenerateRequest):
    """Generate a social media post with text and image."""
    request_id = str(uuid4())
    logger.info("Agent invocation started", extra={"request_id": request_id})

    try:
        inputs = {"messages": [HumanMessage(content=request.prompt)]}
        result = await anyio.to_thread.run_sync(lambda: agent.invoke(inputs))

        text = ""
        for message in reversed(result.get("messages", [])):
            if hasattr(message, "content"):
                text = message.content
                break

        logger.info("Agent invocation completed", extra={"request_id": request_id})
        return GenerateResponse(
            text=text,
            image_url=result.get("image_url"),
            llm_calls=result.get("llm_calls", 0),
            request_id=request_id,
        )

    except Exception:
        logger.exception("Agent invocation failed", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="Failed to generate post")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
