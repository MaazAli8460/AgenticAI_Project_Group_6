import os
import time
import pytest
from dotenv import load_dotenv

from agents.edit_agent.intent_classifier import IntentClassifier, EditTarget

# Load environment variables (GROQ_API_KEY)
load_dotenv()

@pytest.fixture(scope="module")
def classifier():
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set. Skipping LLM intent classifier tests.")
    return IntentClassifier()

# 10 Test cases covering various targets, intents, and scopes
TEST_QUERIES = [
    (
        "Change the narrator's voice tone to whispered",
        EditTarget.AUDIO,
        "character:narrator",
    ),
    (
        "Make the background in scene 2 look like a dark forest",
        EditTarget.VIDEO_FRAME,
        "scene:2",
    ),
    (
        "Add upbeat background music to the whole video",
        EditTarget.AUDIO,
        "global",
    ),
    (
        "Remove the subtitles from the video",
        EditTarget.VIDEO,
        "global",
    ),
    (
        "Make character 1 have bright red hair and punk clothes",
        EditTarget.VIDEO_FRAME,
        "character:1",
    ),
    (
        "Speed up scene 3 to be twice as fast",
        EditTarget.VIDEO,
        "scene:3",
    ),
    (
        "Rewrite the script to end on a sad note instead",
        EditTarget.SCRIPT,
        "global",
    ),
    (
        "Change the lighting in scene 1 to be cinematic and moody",
        EditTarget.VIDEO_FRAME,
        "scene:1",
    ),
    (
        "Use a zoom-out effect for the entire video",
        EditTarget.VIDEO,
        "global",
    ),
    (
        "Change Alice's dialogue in scene 4 to sound angrier",
        EditTarget.SCRIPT,
        "scene:4",
    ),
]

@pytest.mark.parametrize("query, expected_target, expected_scope_hint", TEST_QUERIES)
def test_intent_classification(classifier, query, expected_target, expected_scope_hint):
    time.sleep(2.0)  # Avoid Groq rate limits
    try:
        intent_obj = classifier.classify(query)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "rate limit" in message or "429" in message:
            pytest.skip("Groq rate limit reached. Skipping intent classifier tests.")
        raise
    
    # 1. Check if the target is correctly identified
    assert intent_obj.target == expected_target, \
        f"Failed target on query: '{query}'. Expected {expected_target}, got {intent_obj.target}"
    
    # 2. Check if the scope contains the right hint
    assert expected_scope_hint.lower() in intent_obj.scope.lower() or intent_obj.scope.lower() in expected_scope_hint.lower(), \
        f"Failed scope on query: '{query}'. Expected hint '{expected_scope_hint}', got '{intent_obj.scope}'"
    
    # 3. Check that intent and parameters are populated
    assert len(intent_obj.intent) > 0
    assert isinstance(intent_obj.parameters, dict)
    
    print(f"\nQuery: {query}\nResult: {intent_obj.model_dump_json(indent=2)}")
