from main import should_continue
from nodes.constants import DRAFT_TEXT, GENERATE_IMAGE_PROMPT


def make_state(approved: bool = False, revision_count: int = 0) -> dict:
    return {
        "messages": [],
        "llm_calls": 0,
        "approved": approved,
        "image_url": None,
        "draft_text": "",
        "image_prompt": "",
        "review_feedback": "",
        "revision_count": revision_count,
    }


def test_routes_to_image_prompt_when_approved():
    assert should_continue(make_state(approved=True)) == GENERATE_IMAGE_PROMPT


def test_routes_to_image_prompt_via_escape_hatch():
    # max_revisions defaults to 3; revision_count >= 3 triggers escape
    assert should_continue(make_state(revision_count=3)) == GENERATE_IMAGE_PROMPT


def test_routes_to_draft_when_not_approved():
    assert should_continue(make_state(approved=False, revision_count=1)) == DRAFT_TEXT
