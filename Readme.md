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
TTS_API_KEY=
IMG_API_KEY=
BGM_API_KEY=
```

## Run (Planned)
These commands will be used once the backend and frontend entrypoints are implemented:
- Backend: uvicorn backend.app:app --reload
- Frontend: npm run dev (from frontend/)

## Testing (Planned)
- Backend/unit tests: pytest
- Phase tests: per-agent tests under agents/*/tests

## Status
Skeleton scaffold with shared schema and contracts defined. Phase implementations and endpoints will be added next.