from enum import Enum
import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

from mcp.tools.llm_tools.groq_client import GroqClient


class EditTarget(str, Enum):
    AUDIO = "audio"
    VIDEO_FRAME = "video_frame"
    VIDEO = "video"
    SCRIPT = "script"


class EditIntent(BaseModel):
    intent: str = Field(..., description="A short, snake_case descriptor of the user's intent. E.g. 'change_voice_tone'")
    target: EditTarget = Field(..., description="The architectural target of the edit.")
    scope: str = Field(..., description="What the edit applies to. Can be 'global', 'scene:<id>', 'character:<id>'.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Key-value parameters extracted from the query.")


class IntentClassifier:
    """
    LLM-powered classification agent that receives a raw edit query and 
    outputs a structured Intent object.
    """
    
    def __init__(self, llm_client: Optional[GroqClient] = None, model: Optional[str] = None):
        self._llm = llm_client or self._create_llm(model)
        
    def _create_llm(self, model: Optional[str]) -> GroqClient:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for Intent Classification.")
        model_name = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return GroqClient(api_key=api_key, model=model_name)

    def classify(self, query: str) -> EditIntent:
        """
        Classifies a natural language query into a structured EditIntent.
        """
        system_prompt = """
You are an expert intent classifier for an AI video generation pipeline.
Your job is to parse a natural language edit request and map it to structured JSON.

Targets:
- "audio": Targets TTS output or background music. (e.g. changing voice tone, adding/changing music, muting).
- "video_frame": Targets still image generation. (e.g. changing visual aesthetics, character looks, backgrounds, scene lighting).
- "video": Targets full compositing/export step. (e.g. speed up a scene, remove subtitles, change zoom effect).
- "script": Targets the story/script. (e.g. regenerate the script, rewrite dialogue, change storyline).

Scope formatting:
- For a specific character: "character:<name_or_id>"
- For a specific scene: "scene:<scene_id_or_number>"
- For the whole project: "global"

Return EXACTLY a JSON object matching this schema:
{
  "intent": "snake_case_description",
  "target": "audio" | "video_frame" | "video" | "script",
  "scope": "target scope",
  "parameters": {"key": "value"}
}
"""
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"Query: {query}"}
        ]
        
        try:
            raw_json = self._llm.chat_json(messages, temperature=0.0)
            return EditIntent.model_validate(raw_json)
        except ValidationError as exc:
            # Fallback repair logic if the LLM output doesn't match the schema
            repair_messages = [
                {"role": "system", "content": "Fix this JSON to match the strict schema. Only return the JSON."},
                {"role": "user", "content": f"JSON: {raw_json}\nError: {exc}"}
            ]
            fixed_json = self._llm.chat_json(repair_messages, temperature=0.0)
            return EditIntent.model_validate(fixed_json)
