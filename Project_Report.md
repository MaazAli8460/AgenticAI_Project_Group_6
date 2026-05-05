# AI-Powered Animated Video Generation System

## Abstract
This report describes an end-to-end agentic pipeline that turns a single natural-language prompt into a polished animated short video. The system is organized into five phases: story and script generation, audio synthesis and mixing, video composition, a full-stack web interface, and an edit agent with undo support. Each phase operates over a shared JSON state, enabling modular execution, re-runs, and versioned edits.

## System Architecture
The pipeline is orchestrated as a phase-aware workflow with a shared state object passed forward and updated by each module. The web interface coordinates execution, monitors progress, and exposes edit/undo controls.

```
User Prompt
   |
   v
Phase 1: Story + Script + Characters (LLM)
   |
   v
Phase 2: Dialogue TTS + BGM + Timing (Audio)
   |
   v
Phase 3: Scene Images + Animation + Sync (Video)
   |
   v
Phase 4: Web UI + Orchestration (FastAPI + React)
   |
   v
Phase 5: Edit Agent + Undo (LangGraph + State Manager)
```

## Shared JSON Schema Design
The JSON state is the contract between phases. It is validated and extended at each stage.

Core fields:
- story: full narrative text
- characters: list of character profiles with name, gender, voice_id, and visual description
- scenes: list of scene entries with dialogue lines, setting, mood, duration, and per-speaker metadata
- scenes.character_overrides: optional per-scene visual overrides for targeted edits
- audio: per-line audio file references and a timing manifest for synchronization
- visuals: per-scene image prompts, generated image paths, and animation metadata
- outputs: assembled audio/video artifacts and final MP4 location

The schema enables:
- deterministic re-runs of individual phases
- consistent character and voice usage across scenes
- precise synchronization between audio and visuals

## Phase-wise Implementation Details

### Phase 1: Story and Script Generation
- Uses a Groq-hosted LLM to expand a user prompt into a structured story with scene-by-scene dialogue.
- Enforces speaker-driven segmentation and gender alignment between voices and visual prompts.
- Produces a validated JSON state with characters and scenes suitable for downstream phases.

### Phase 2: Audio Generation and Integration
- Uses ElevenLabs TTS for dialogue lines with per-character voice selection.
- Builds a timing manifest from dialogue durations.
- Applies randomized background music selection with a configurable seed for reproducibility.
- Outputs audio files and a combined dialogue track for synchronization.

### Phase 3: Video Generation and Composition
- Generates per-speaker, camera-facing images using the Pollinations image API.
- Applies simple animation (Ken Burns style) and scene concatenation via FFmpeg.
- Optionally runs Wav2Lip for lip-sync if configured.
- Creates subtitles and exports final MP4.

### Phase 4: Web Interface and Orchestration
- FastAPI backend exposes phase endpoints and a Server-Sent Events stream.
- React frontend provides prompt input, project tracking, phase-level reruns, and preview.
- Status cards and event logs surface real-time progress for each phase.

### Phase 5: Edit Agent with Undo
- LangGraph-based edit agent interprets natural-language edit commands.
- Tool routing decides whether to adjust story text, audio parameters, or visuals.
- A versioned state manager snapshots changes and supports undo via history restoration.
- Scene-scoped overrides allow targeted edits without re-rendering unrelated scenes.
- Subtitle toggles switch between pre-rendered videos without re-running Phase 3.

## Tools and APIs Used
- LLM: Groq (llama-3.3-70b-versatile)
- TTS: ElevenLabs
- Images: Pollinations
- Video: FFmpeg
- Orchestration: LangChain, LangGraph
- Backend: FastAPI
- Frontend: React + Vite

## Challenges and Mitigations
- Dialogue segmentation drift: forced speaker-turn segmentation to align audio and visuals.
- Gender consistency: enforced in story generation and visual prompt engineering.
- Provider reliability: standardized on Pollinations for image generation.
- Audio mixing balance: tuned BGM and dialogue gain controls with debug mode.
- Lip-sync resilience: Wav2Lip failures fall back to raw clips without blocking the pipeline.
- Undo correctness: centralized state snapshots with safe reversion logic.

## Results
- End-to-end pipeline generates a full animated short video from a single prompt.
- Phase-level reruns complete without full pipeline restarts.
- Edit agent supports iterative improvement and undo across outputs.

## Individual Contributions
- Member 1: Phase 1 story and script generation, schema definition.
- Member 2: Phase 2 audio synthesis, timing manifest, BGM mixing.
- Member 3: Phase 3 image generation, animation, video composition.
- Member 4: Phase 4 UI/backend, Phase 5 edit agent integration.

## Limitations and Future Work
- Expand visual variety with multi-image blending and transitions.
- Add more robust lip-sync handling and face detection fallbacks.
- Improve edit agent reasoning with richer tool feedback and tests.
- Package a turnkey installer or Docker setup for easier deployment.

## Conclusion
The system demonstrates a modular, agentic approach to automated video generation. The shared JSON schema and phase-wise orchestration enable flexible iteration, while the edit agent provides practical post-generation control with undo support.