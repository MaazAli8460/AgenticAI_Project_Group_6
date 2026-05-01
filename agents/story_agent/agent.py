from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from shared.schemas.state import PipelineState

from .planner import StoryPlanner


class StoryAgent:
	def __init__(self, planner: Optional[StoryPlanner] = None) -> None:
		self._planner = planner or StoryPlanner()

	def generate_state(
		self,
		prompt: str,
		project_id: Optional[str] = None,
		seed: Optional[int] = None,
	) -> PipelineState:
		return self._planner.build_state(prompt, project_id=project_id, seed=seed)


def build_story_state(
	prompt: str,
	project_id: Optional[str] = None,
	seed: Optional[int] = None,
) -> PipelineState:
	return StoryAgent().generate_state(prompt, project_id=project_id, seed=seed)


if __name__ == "__main__":
	prompt_text = " ".join(sys.argv[1:]).strip() or "A traveler finds a hidden valley."
	state = build_story_state(prompt_text)
	print(json.dumps(state.model_dump(mode="json"), indent=2))
