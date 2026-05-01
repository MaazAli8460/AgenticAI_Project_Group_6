from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from agents.story_agent.agent import StoryAgent
from agents.story_agent.planner import StoryPlanner


class FakeLLM:
    def chat_json(self, messages, temperature=0.7, max_tokens=2000):
        return {
            "story": {
                "logline": "An explorer finds a hidden ocean on Mars.",
                "synopsis": "A short tale of discovery and wonder.",
                "genre": "adventure",
                "theme": "discovery",
                "tone": "uplifting",
                "style": "cinematic",
                "target_duration_s": 18
            },
            "characters": [
                {
                    "id": "char_1",
                    "name": "Ari",
                    "role": "Protagonist",
                    "description": "Curious astronaut.",
                    "voice": {"style": "bright", "gender": "neutral", "params": {}},
                    "visual": {"description": "Explorer suit", "style": "2D", "palette": ["teal"]}
                },
                {
                    "id": "char_2",
                    "name": "Narrator",
                    "role": "Narrator",
                    "description": "Guides the story.",
                    "voice": {"style": "calm", "gender": "neutral", "params": {}},
                    "visual": {"description": "Silhouette", "style": "2D", "palette": ["gold"]}
                }
            ],
            "scenes": [
                {
                    "id": "scene_1",
                    "order": 1,
                    "title": "Arrival",
                    "description": "A landing at dawn.",
                    "setting": "Martian plain",
                    "mood": "anticipatory",
                    "style": "cinematic",
                    "duration_s": 6,
                    "visual_prompt": "Wide shot, soft light.",
                    "camera": "wide",
                    "motion": "slow pan",
                    "bgm_mood": "ambient",
                    "bgm_style": "cinematic",
                    "character_ids": ["char_1", "char_2"],
                    "dialogue": [
                        {
                            "id": "line_1",
                            "character_id": "char_2",
                            "text": "Our explorer steps into the unknown.",
                            "emotion": "gentle"
                        }
                    ]
                },
                {
                    "id": "scene_2",
                    "order": 2,
                    "title": "Discovery",
                    "description": "A hidden ocean glows.",
                    "setting": "Cavern opening",
                    "mood": "wonder",
                    "style": "cinematic",
                    "duration_s": 12,
                    "visual_prompt": "Glowing reflections.",
                    "camera": "medium",
                    "motion": "slow push in",
                    "bgm_mood": "mystical",
                    "bgm_style": "ethereal",
                    "character_ids": ["char_1"],
                    "dialogue": [
                        {
                            "id": "line_2",
                            "character_id": "char_1",
                            "text": "There is a sea beneath the surface.",
                            "emotion": "amazed"
                        }
                    ]
                }
            ]
        }


def test_story_agent_generates_state() -> None:
    prompt = "A young astronaut discovers a hidden ocean on Mars"
    planner = StoryPlanner(llm_client=FakeLLM())
    agent = StoryAgent(planner=planner)
    state = agent.generate_state(prompt, project_id="proj_test", seed=7)

    assert state.meta.schema_version == "1.0"
    assert state.meta.prompt == prompt
    assert state.meta.project_id == "proj_test"
    assert state.meta.total_duration_s == sum(scene.duration_s for scene in state.scenes)

    assert len(state.characters) >= 2
    assert len(state.scenes) >= 2

    character_ids = {char.id for char in state.characters}
    for scene in state.scenes:
        assert scene.order is not None
        assert scene.character_ids
        assert set(scene.character_ids).issubset(character_ids)
        for line in scene.dialogue:
            assert line.character_id in character_ids

    assert state.phases is not None
    assert state.phases.story is not None
    assert state.phases.story.status == "complete"
