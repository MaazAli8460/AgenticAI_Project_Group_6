from __future__ import annotations

from shared.constants.bgm_moods import BGM_MOODS


def build_phase1_prompt(user_prompt: str) -> str:
    prompt_text = user_prompt.strip() or "A mysterious world"
    mood_list = ", ".join(BGM_MOODS)
    return (
        "You are a story planning agent for a short animated film. "
        "Expand the user concept into a concise, cinematic story with a clear arc: "
        "hook, discovery, dilemma, decision, and resolution. "
        "Keep the tone consistent, ensure character motivations are clear, and "
        "make scenes visually specific for image generation.\n\n"
        "For each scene, choose bgm_mood strictly from this list: "
        f"{mood_list}.\n\n"
        f"User concept: {prompt_text}"
    )
