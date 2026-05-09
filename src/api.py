from dotenv import load_dotenv

# Load environment variables BEFORE any other imports
load_dotenv()

import logging
import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from main import agent

app = FastAPI(title="Post-It API", version="0.1.0")

if not os.getenv("API_TOKEN"):
    logging.warning("API_TOKEN not set — all requests will be accepted. Do not deploy without setting this.")


class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    text: str
    image_url: Optional[str] = None
    llm_calls: int


# Simple token validation (you'll enhance this later)
def validate_token(authorization: Optional[str] = Header(None)) -> bool:
    """Validate API token from Authorization header"""
    if not authorization:
        return False

    # Extract token from "Bearer <token>" format
    try:
        token = authorization.split(" ")[1] if " " in authorization else authorization
    except IndexError:
        return False

    # Get valid token from environment
    valid_token = os.getenv("API_TOKEN")
    if not valid_token:
        # If no token set, allow requests (for testing)
        return True

    return token == valid_token


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "post-it-api"}


@app.get("/health")
async def health():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_post(
    request: GenerateRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Generate a social media post with text and image.
    Requires Authorization header with Bearer token.
    """
    # Validate token
    if not validate_token(authorization):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authorization token",
        )

    try:
        # Invoke the agent
        inputs = {"messages": [HumanMessage(content=request.prompt)]}
        result = agent.invoke(inputs)

        # Extract the last text message (the final draft)
        text = ""
        for message in reversed(result.get("messages", [])):
            if hasattr(message, "content"):
                text = message.content
                break

        return GenerateResponse(
            text=text,
            image_url=result.get("image_url"),
            llm_calls=result.get("llm_calls", 0),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating post: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
