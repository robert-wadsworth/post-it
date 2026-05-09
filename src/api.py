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
    image_prompt: str
    review_feedback: str
    revision_count: int
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

        logger.info("Agent invocation completed", extra={"request_id": request_id})
        return GenerateResponse(
            text=result.get("draft_text", ""),
            image_url=result.get("image_url"),
            image_prompt=result.get("image_prompt", ""),
            review_feedback=result.get("review_feedback", ""),
            revision_count=result.get("revision_count", 0),
            llm_calls=result.get("llm_calls", 0),
            request_id=request_id,
        )

    except Exception:
        logger.exception("Agent invocation failed", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="Failed to generate post")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
