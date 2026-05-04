from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from mcp.tools.llm_tools.groq_client import GroqClient
from shared.schemas.state import (
    Character,
    DialogueLine,
    PhaseStatus,
    PhaseStatusMap,
    PipelineState,
    ProjectMeta,
    Scene,
    Story,
)
from shared.constants.bgm_moods import BGM_MOODS, normalize_bgm_mood

from .prompt_engineer import build_phase1_prompt


class Phase1Output(BaseModel):
    story: Story
    scenes: list[Scene] = Field(default_factory=list)
    characters: list[Character] = Field(default_factory=list)


class StoryPlanner:
    def __init__(
        self,
        schema_version: str = "1.0",
        llm_client: Optional[GroqClient] = None,
        model: Optional[str] = None,
    ) -> None:
        self.schema_version = schema_version
        self._llm = llm_client or self._create_llm(model)

    def _create_llm(self, model: Optional[str]) -> GroqClient:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for Phase 1 LLM generation.")
        model_name = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return GroqClient(api_key=api_key, model=model_name)

    def build_state(
        self,
        prompt: str,
        project_id: str | None = None,
        seed: int | None = None,
    ) -> PipelineState:
        prompt_text = prompt.strip() or "A mysterious world"
        phase1 = self._generate_phase1(prompt_text)
        phase1 = self._normalize_phase1(phase1, prompt_text)

        now = datetime.now(timezone.utc)
        project_id = project_id or f"proj_{uuid4().hex[:8]}"
        total_duration = sum(scene.duration_s for scene in phase1.scenes)

        if phase1.story.target_duration_s is None or phase1.story.target_duration_s <= 0:
            phase1.story.target_duration_s = total_duration

        meta = ProjectMeta(
            project_id=project_id,
            prompt=prompt_text,
            created_at=now,
            updated_at=now,
            version=1,
            schema_version=self.schema_version,
            seed=seed,
            total_duration_s=total_duration,
            tags=["phase1", "story"],
        )
        phases = PhaseStatusMap(
            story=PhaseStatus(status="complete", updated_at=now, message="Story generated")
        )

        return PipelineState(
            meta=meta,
            story=phase1.story,
            scenes=phase1.scenes,
            characters=phase1.characters,
            phases=phases,
        )

    def _generate_phase1(self, prompt: str) -> Phase1Output:
        messages = self._build_messages(prompt)
        raw = self._llm.chat_json(messages, temperature=0.7, max_tokens=2500)
        return self._parse_phase1(raw, messages)

    def _parse_phase1(self, raw: dict[str, Any], messages: list[dict[str, str]]) -> Phase1Output:
        try:
            return Phase1Output.model_validate(raw)
        except ValidationError as exc:
            fixed = self._repair_output(raw, exc, messages)
            return Phase1Output.model_validate(fixed)

    def _repair_output(
        self,
        raw: dict[str, Any],
        exc: ValidationError,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You repair JSON to match the required schema. "
                    "Return only a JSON object with keys story, scenes, characters."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Fix the JSON below to match the schema constraints. "
                    "Do not add extra keys.\n\n"
                    f"Validation errors:\n{exc}\n\n"
                    f"JSON to fix:\n{json.dumps(raw, ensure_ascii=True)}"
                ),
            },
        ]
        return self._llm.chat_json(repair_messages, temperature=0.0, max_tokens=2000)

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        schema_hint = (
            "Return a JSON object with keys: story, scenes, characters.\n"
            "story: {logline, synopsis, genre, theme, tone, style, target_duration_s}\n"
            "characters: array of {id, name, role, description, voice, visual}\n"
            "voice: {style, gender, age, accent, model, params}\n"
            "visual: {description, style, palette[], references[]}\n"
            "scenes: array of {id, order, title, description, setting, mood, style, "
            "duration_s, visual_prompt, camera, motion, bgm_mood, bgm_style, "
            "character_ids[], dialogue[]}\n"
            "dialogue: array of {id, character_id, text, emotion, direction}\n"
        )
        mood_list = ", ".join(BGM_MOODS)
        rules = (
            "Rules:\n"
            "- 3 to 6 scenes, total duration 15-45 seconds.\n"
            "- 2 to 4 characters.\n"
            "- Use short ids like char_1, scene_1, line_1.\n"
            "- Ensure dialogue character_id references a character id.\n"
            f"- Set bgm_mood using only this list: {mood_list}.\n"
            "- Keep text concise for a short animated film.\n"
            "- Every character must have voice.gender set to male, female, or neutral.\n"
            "- Visual descriptions must match voice.gender.\n"
            "- Each scene dialogue ends with a clear concluding line; avoid trailing ellipses.\n"
        )
        engineered_prompt = build_phase1_prompt(prompt)
        return [
            {
                "role": "system",
                "content": (
                    "You are a story planning agent for an animated short. "
                    "Return only JSON; no markdown or commentary."
                ),
            },
            {"role": "user", "content": f"{engineered_prompt}\n\n{schema_hint}\n{rules}"},
        ]

    def _normalize_phase1(self, output: Phase1Output, prompt: str) -> Phase1Output:
        prompt_seed = prompt or "A mysterious world"

        character_ids: dict[str, str] = {}
        used_ids: set[str] = set()
        for idx, char in enumerate(output.characters, start=1):
            new_id = self._sanitize_id(char.id or char.name, "char", idx)
            new_id = self._unique_id(new_id, used_ids)
            character_ids[char.id] = new_id
            char.id = new_id
            char.voice.gender = self._normalize_gender(char.voice.gender)
            if not char.voice.gender:
                char.voice.gender = self._infer_gender_from_text(
                    f"{char.description} {char.visual.description}"
                )
            if char.voice.gender:
                char.visual.description = self._align_visual_gender(
                    char.visual.description, char.voice.gender
                )
            used_ids.add(new_id)

        if not output.characters:
            raise ValueError("Phase 1 output must include at least one character.")

        for scene_idx, scene in enumerate(output.scenes, start=1):
            scene.id = self._sanitize_id(scene.id or f"scene_{scene_idx}", "scene", scene_idx)
            scene.order = scene.order or scene_idx
            if not scene.duration_s or scene.duration_s <= 0:
                scene.duration_s = 6.0
            if not scene.visual_prompt:
                scene.visual_prompt = f"{prompt_seed}. Cinematic wide shot."
            if not scene.dialogue:
                scene.dialogue = []

            scene.bgm_mood = normalize_bgm_mood(scene.bgm_mood or scene.mood)

            referenced_ids: list[str] = []
            scene.character_ids = [
                character_ids.get(character_id, character_id)
                for character_id in scene.character_ids
            ]
            for line_idx, line in enumerate(scene.dialogue, start=1):
                line.id = line.id or f"line_{scene_idx}_{line_idx}"
                if line.character_id in character_ids:
                    line.character_id = character_ids[line.character_id]
                elif line.character_id in used_ids:
                    line.character_id = line.character_id
                else:
                    line.character_id = output.characters[0].id
                referenced_ids.append(line.character_id)
            if scene.dialogue:
                last_line = scene.dialogue[-1]
                if last_line.text:
                    last_line.text = self._ensure_conclusion(last_line.text)

            if not scene.character_ids:
                scene.character_ids = referenced_ids
            if not scene.character_ids:
                scene.character_ids = [output.characters[0].id]
            scene.character_ids = list(dict.fromkeys(scene.character_ids))

        if not output.scenes:
            raise ValueError("Phase 1 output must include at least one scene.")

        if not output.story.logline:
            output.story.logline = prompt_seed
        if not output.story.synopsis:
            output.story.synopsis = f"A short animated tale inspired by: {prompt_seed}."

        return output

    @staticmethod
    def _ensure_conclusion(text: str) -> str:
        trimmed = text.rstrip()
        if trimmed.endswith("..."):
            trimmed = trimmed[:-3].rstrip()
        if not trimmed.endswith((".", "!", "?")):
            trimmed = f"{trimmed}."
        return trimmed

    @staticmethod
    def _normalize_gender(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        slug = value.strip().lower()
        if slug in {"male", "man", "boy", "masculine", "m"}:
            return "male"
        if slug in {"female", "woman", "girl", "feminine", "f"}:
            return "female"
        if slug in {"neutral", "nonbinary", "non-binary", "androgynous"}:
            return "neutral"
        return None

    @staticmethod
    def _infer_gender_from_text(text: str) -> Optional[str]:
        lowered = text.lower()
        if re.search(r"\b(man|male|boy|masculine)\b", lowered):
            return "male"
        if re.search(r"\b(woman|female|girl|feminine)\b", lowered):
            return "female"
        if re.search(r"\b(nonbinary|non-binary|androgynous|neutral)\b", lowered):
            return "neutral"
        return None

    @staticmethod
    def _align_visual_gender(description: str, gender: str) -> str:
        if not description:
            return description
        updated = description
        if gender == "male":
            updated = re.sub(r"\b(woman|female|girl|lady)\b", "man", updated, flags=re.I)
        elif gender == "female":
            updated = re.sub(r"\b(man|male|boy|gentleman)\b", "woman", updated, flags=re.I)
        elif gender == "neutral":
            if not re.search(r"\b(androgynous|neutral|nonbinary|non-binary)\b", updated, flags=re.I):
                updated = f"androgynous {updated}"
        if not re.search(r"\b(male|female|man|woman|boy|girl|androgynous|neutral|nonbinary|non-binary)\b", updated, flags=re.I):
            updated = f"{gender} {updated}"
        return updated

    @staticmethod
    def _sanitize_id(value: str, prefix: str, index: int) -> str:
        if not value:
            return f"{prefix}_{index}"
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
        if not slug:
            return f"{prefix}_{index}"
        if not slug.startswith(prefix):
            slug = f"{prefix}_{slug}"
        return slug

    @staticmethod
    def _unique_id(value: str, used: set[str]) -> str:
        if value not in used:
            return value
        counter = 2
        while f"{value}_{counter}" in used:
            counter += 1
        return f"{value}_{counter}"
