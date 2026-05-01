# Agentic AI Project Group 6

AI-powered animated video generation system that turns a single prompt into a short film via a multi-phase agent pipeline.

## Phase Overview
1. Story and script (Phase 1): prompt expansion, scene-by-scene script, and character roster.
2. Audio generation (Phase 2): TTS per dialogue line plus background music and a timing manifest.
3. Video generation (Phase 3): per-scene imagery, light animation, A/V sync, and final MP4.
4. Web interface (Phase 4): full-stack UI for orchestration, progress, and re-run controls.
5. Edit agent and undo (Phase 5): intent classification, targeted re-runs, and versioned state history.

## Shared Data Contract
All phases read/write a shared PipelineState JSON object.
- JSON Schema: shared/schemas/pipeline_state.schema.json
- Pydantic models: shared/schemas/state.py

Required top-level fields: meta, story, scenes, characters. Later phases append audio, video, and edits.

## Repository Layout (Current)
- agents/: phase agents and orchestration graphs
- backend/: API server and orchestration endpoints
- frontend/: UI for prompt input, progress, and previews
- mcp/: tool abstraction layer and tool implementations
- shared/: schemas and shared utilities
- state_manager/: snapshot, history, and storage for undo

## Setup
Prerequisites:
- Python 3.10+ and Node.js 18+
- FFmpeg installed and on PATH
- API keys or local models for LLM, TTS, and image generation

Backend environment:
```bash
python -m venv .venv
# Windows
\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Frontend environment:
```bash
cd frontend
npm install
```

## Configuration
Create a .env file at the repo root with any required keys. Example:
```
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
TTS_PROVIDER=auto
TTS_STRICT=0
TTS_DEBUG=0
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=
ELEVENLABS_VOICE_ID_FEMALE=
ELEVENLABS_VOICE_ID_MALE=
ELEVENLABS_VOICE_ID_NEUTRAL=
ELEVENLABS_MODEL=eleven_multilingual_v2
ELEVENLABS_OUTPUT_FORMAT=wav_22050
BGM_LIBRARY_DIR=
IMG_API_KEY=
BGM_API_KEY=
```

## Run (Planned)
These commands will be used once the backend and frontend entrypoints are implemented:
- Backend: uvicorn backend.app:app --reload
- Frontend: npm run dev (from frontend/)

## Run (Current)
Phase 1 can be executed directly:
```bash
python run_phase1.py "A young astronaut discovers a hidden ocean on Mars and must decide whether to reveal it."
```
Outputs are saved to data/outputs/phase1/<project_id>.json. Use --stdout to also print JSON.

Phase 2 can be executed using a Phase 1 JSON output:
```bash
python run_phase2.py --input data/outputs/phase1/<project_id>.json
```
Audio outputs and the updated state are saved under data/outputs/phase2/<project_id>/.

If ELEVENLABS_VOICE_ID is not set, the TTS tool will fetch your available ElevenLabs voices and
select one based on character gender, style/tone keywords, accent, and age labels when possible.

## Testing (Planned)
- Backend/unit tests: pytest
- Phase tests: per-agent tests under agents/*/tests

## Status
Completed:
- Shared JSON schema + Pydantic models
- Phase 1 story generation with Groq LLM
- Prompt engineering helper for Phase 1
- Phase 1 CLI runner with file output
- Phase 2 audio generation with TTS fallback and timing manifest

In progress:
- Phase 2 audio generation
- Phase 3 video composition
- Phase 4 web interface
- Phase 5 edit agent and undo

## Next Steps
1. Phase 2: build TTS pipeline and timing manifest writer.
2. Phase 3: add image generation + FFmpeg compositing.
3. Phase 4: create FastAPI endpoints + React UI for orchestration.
4. Phase 5: implement intent classifier, edit executor, and state history.