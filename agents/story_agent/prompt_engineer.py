from __future__ import annotations


def build_phase1_prompt(user_prompt: str) -> str:
    prompt_text = user_prompt.strip() or "A mysterious world"
    return (
        "You are a story planning agent for a short animated film. "
        "Expand the user concept into a concise, cinematic story with a clear arc: "
        "hook, discovery, dilemma, decision, and resolution. "
        "Keep the tone consistent, ensure character motivations are clear, and "
        "make scenes visually specific for image generation.\n\n"
        f"User concept: {prompt_text}"
    )
