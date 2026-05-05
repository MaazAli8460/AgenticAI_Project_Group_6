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
- FFmpeg installed (normally on `PATH`; if your terminal was open before `winget install Gyan.FFmpeg`, restart it or set `FFMPEG_PATH` to `ffmpeg.exe`. On Windows the project also auto-detects WinGet’s `…\WinGet\Packages\…\bin\ffmpeg.exe`.)
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
BGM_GAIN=0.2
DIALOGUE_GAIN=0.9
BGM_DEBUG=0
BGM_STRICT=0
BGM_START_OFFSET_S=20
BGM_RANDOM_SEED=
IMAGE_PROVIDER=pollinations
POLLINATIONS_BASE_URL=https://image.pollinations.ai
POLLINATIONS_MODEL=
IMAGE_SEED=
VIDEO_RESOLUTION=1280x720
VIDEO_FPS=24
VIDEO_EFFECT=zoom_in
SUBTITLES_ENABLED=1
IMG_API_KEY=
BGM_API_KEY=
LIP_SYNC_ENABLED=0
LIP_SYNC_STRICT=0
LIP_SYNC_DEBUG=0
WAV2LIP_DIR=
WAV2LIP_CHECKPOINT=
WAV2LIP_PYTHON=
WAV2LIP_PADS=0,10,0,0
WAV2LIP_RESIZE_FACTOR=1
WAV2LIP_NOSMOOTH=0
WAV2LIP_BATCH_SIZE=128
WAV2LIP_FACE_DET_BATCH_SIZE=16
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

Phase 3 can be executed using a Phase 2 JSON output:
```bash
python run_phase3.py --input data/outputs/phase2/<project_id>/state.json
```
Video outputs and the updated state are saved under data/outputs/phase3/<project_id>/.

Phase 3 behavior:
- Scene segments are created only on speaker changes (no fixed splitting).
- Single-speaker scenes render as one continuous clip.
- Per-speaker images are generated and stitched per segment.
- Dialogue speaker images are prompted as camera-facing, photorealistic portraits
	with sharp focus to support lip-sync.

If ELEVENLABS_VOICE_ID is not set, the TTS tool will fetch your available ElevenLabs voices and
select one based on character gender, style/tone keywords, accent, and age labels when possible.
Note: this requires an ElevenLabs API key with voices_read permission. If your key does not
include voices_read, set ELEVENLABS_VOICE_ID_FEMALE/MALE/NEUTRAL explicitly.

BGM_LIBRARY_DIR can point to a folder that contains mood-named subfolders. The current supported
moods are: agressive, atmospheric, carefree, confident, disturbing, dramatic, eerie, fun, happy,
holiday, hopefull, mysterious, nightmarish, party, proud, relaxed, romantic, sad, scary, sensual,
sexy, smooth, triumphant, upbeat, uplifting, urban, weird, young.
The BGM library can contain .wav, .mp3, .m4a, .flac, or .ogg files. Non-wav formats require FFmpeg
to convert to WAV for mixing.
Set BGM_DEBUG=1 to log which BGM track is selected per scene. Set BGM_STRICT=1 to fail if BGM
conversion is not possible.
BGM_START_OFFSET_S trims the first N seconds from each BGM track before mixing.
BGM tracks are selected randomly per mood without repeats until the pool is exhausted. Set
BGM_RANDOM_SEED for deterministic selection.
BGM is layered under the full dialogue timeline per scene (background layer = BGM, foreground = dialogue).

IMAGE_PROVIDER controls which image generator to use. Supported values are pollinations.
POLLINATIONS_BASE_URL and POLLINATIONS_MODEL configure Pollinations.
VIDEO_RESOLUTION, VIDEO_FPS, and VIDEO_EFFECT control clip rendering. SUBTITLES_ENABLED toggles SRT burn-in.
Dialogue speaker images are prompted as camera-facing, photorealistic portraits with sharp focus to
support lip-sync.

## Lip-sync with Wav2Lip (optional)
Phase 3 can drive Wav2Lip to lip-sync each speaker segment against its dialogue audio.
The integration runs Wav2Lip per segment (one speaker at a time), then concatenates the
synced clips and muxes the full mixed audio (dialogue + BGM) at the end. Lip-sync is
applied only to segments that have dialogue; non-speaking fallback segments are passed
through unchanged.

Setup (one-time):
The upstream Wav2Lip repo gitignores its own model weights, and its pinned dependencies
target Python 3.7 / torch 1.1.0. The project includes an automated setup script that
clones the repo, downloads the model weights from public Hugging Face mirrors (with
SHA-256 verification), patches Wav2Lip's `audio.py` / `inference.py` /
`face_detection/detection/sfd/sfd_detector.py` for compatibility with modern PyTorch
(>=2.6), librosa (>=0.10), and numpy (>=2), and writes the required keys into your
`.env`:
```bash
python scripts/setup_wav2lip.py --install-deps
```
This:
1. Clones https://github.com/Rudrabha/Wav2Lip into `third_party/Wav2Lip/`
   (`third_party/` is gitignored).
2. Downloads `wav2lip_gan.pth` (~436 MB) and `s3fd.pth` (~90 MB) into the right
   subdirectories of `third_party/Wav2Lip/`.
3. Applies idempotent patches (re-runs are safe).
4. Pip-installs `librosa` (the only Wav2Lip dep that's not already present in a
   typical PyTorch install). Drop `--install-deps` if you want to manage that
   yourself or use a separate venv.
5. Adds `LIP_SYNC_ENABLED=1`, `WAV2LIP_DIR`, and `WAV2LIP_CHECKPOINT` to `.env`
   (creating it from `.example.env` if needed).

Other Wav2Lip dependencies (torch, torchvision, opencv-python, numpy, numba, scipy,
tqdm) are normally already installed alongside PyTorch. If any are missing, install
them into the same Python environment as this project, or set up a dedicated venv
and point at it via `WAV2LIP_PYTHON` in `.env`.

You also need ffmpeg on `PATH` (the rest of Phase 3 already requires it). On Windows:
```powershell
winget install Gyan.FFmpeg
# restart the shell so the PATH update is picked up
```

Configure (auto-set by the setup script, shown here for reference):
```
LIP_SYNC_ENABLED=1
WAV2LIP_DIR=<repo>/third_party/Wav2Lip
WAV2LIP_CHECKPOINT=<repo>/third_party/Wav2Lip/checkpoints/wav2lip_gan.pth
WAV2LIP_PYTHON=                  # leave empty to reuse this project's interpreter
# Optional tuning:
WAV2LIP_PADS=0,10,0,0
WAV2LIP_RESIZE_FACTOR=1
WAV2LIP_NOSMOOTH=0
WAV2LIP_BATCH_SIZE=128
WAV2LIP_FACE_DET_BATCH_SIZE=16
LIP_SYNC_STRICT=0
LIP_SYNC_DEBUG=0
```

Behavior:
- If `LIP_SYNC_ENABLED=1` but `WAV2LIP_DIR` / `WAV2LIP_CHECKPOINT` are missing or
	invalid, Phase 3 disables lip-sync and renders normally. Set `LIP_SYNC_STRICT=1`
	to fail loudly instead.
- If Wav2Lip fails on a single segment (face not detected, etc.), that segment falls
	back to the raw image clip while other segments still get lip-synced. Set
	`LIP_SYNC_STRICT=1` to abort the run instead.
- Set `LIP_SYNC_DEBUG=1` to log per-segment lip-sync decisions and the underlying
	Wav2Lip command.
- Per-scene metadata about which segments were lip-synced (or fell back, with reason)
	is written to `data/outputs/phase3/<project_id>/prompts.json` under each scene's
	`lip_sync_segments` entry. Intermediate files (segment dialogue WAV, raw clip, and
	Wav2Lip output) are kept under `data/outputs/phase3/<project_id>/lip_sync/`.

## Testing (Planned)
- Backend/unit tests: pytest
- Phase tests: per-agent tests under agents/*/tests

## Status
Completed:
- Shared JSON schema + Pydantic models
- Phase 1 story generation with Groq LLM
- Prompt engineering helper for Phase 1
- Phase 1 CLI runner with file output
- Phase 2 audio generation with timing manifest, BGM layering, and file output
- Phase 3 video generation with per-scene assets and final MP4

In progress:
- Phase 4 web interface
- Phase 5 edit agent and undo

## Next Steps
1. Phase 4: create FastAPI endpoints + React UI for orchestration.
2. Phase 5: implement intent classifier, edit executor, and state history.