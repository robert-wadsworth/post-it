from main import should_continue
from nodes.constants import GENERATE_IMAGE_PROMPT, REVIEW_DRAFT


def make_state(approved: bool = False, llm_calls: int = 0) -> dict:
    return {
        "messages": [],
        "llm_calls": llm_calls,
        "approved": approved,
        "image_url": None,
    }


def test_routes_to_image_prompt_when_approved():
    assert should_continue(make_state(approved=True)) == GENERATE_IMAGE_PROMPT


def test_routes_to_image_prompt_via_escape_hatch():
    assert should_continue(make_state(llm_calls=6)) == GENERATE_IMAGE_PROMPT


def test_routes_to_review_when_not_approved():
    assert should_continue(make_state(approved=False, llm_calls=2)) == REVIEW_DRAFT
