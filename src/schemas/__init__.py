from pydantic import BaseModel


class ReviewDecision(BaseModel):
    feedback: str
    approved: bool
