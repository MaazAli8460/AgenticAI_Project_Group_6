from __future__ import annotations

import re
from typing import Optional

BGM_MOODS = [
    "agressive",
    "atmospheric",
    "carefree",
    "confident",
    "disturbing",
    "dramatic",
    "eerie",
    "fun",
    "happy",
    "holiday",
    "hopefull",
    "mysterious",
    "nightmarish",
    "party",
    "proud",
    "relaxed",
    "romantic",
    "sad",
    "scary",
    "sensual",
    "sexy",
    "smooth",
    "triumphant",
    "upbeat",
    "uplifting",
    "urban",
    "weird",
    "young",
]

BGM_DEFAULT_MOOD = "atmospheric"

BGM_ALIASES = {
    "aggressive": "agressive",
    "hopeful": "hopefull",
    "hopefull": "hopefull",
    "wonder": "mysterious",
    "wonderful": "uplifting",
    "introspective": "relaxed",
    "reflective": "relaxed",
    "determination": "confident",
    "determined": "confident",
    "apprehension": "eerie",
    "tense": "eerie",
}


def normalize_bgm_mood(value: Optional[str]) -> str:
    if not value:
        return BGM_DEFAULT_MOOD
    slug = _slugify(value)
    if slug in BGM_ALIASES:
        return BGM_ALIASES[slug]
    for mood in BGM_MOODS:
        if _slugify(mood) == slug:
            return mood
    return BGM_DEFAULT_MOOD


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())
