"""
Wav2Lip Remote Inference Server — for Google Colab
===================================================

Paste the cells below into a Colab notebook (Runtime → Change runtime type → GPU).
The server exposes a single POST /lip_sync endpoint that accepts a face video
and audio file, runs Wav2Lip inference on Colab's GPU, and returns the
lip-synced video.

A public ngrok tunnel is printed to the console — copy it into your local
project's .env as WAV2LIP_REMOTE_URL.

------- CELL 1: Setup (run once) -------
"""

CELL_1_SETUP = r'''
# ============================================================
# CELL 1: Install dependencies and download Wav2Lip
# ============================================================

# 1a. Clone Wav2Lip
!git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git /content/Wav2Lip

# 1b. Download model weights
!mkdir -p /content/Wav2Lip/checkpoints
!wget -q -O /content/Wav2Lip/checkpoints/wav2lip_gan.pth \
    "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true"

# 1c. Download s3fd face detector weights
!wget -q -O /content/Wav2Lip/face_detection/detection/sfd/s3fd.pth \
    "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/s3fd-619a316812.pth"

# 1d. Install Python deps
!pip install -q flask pyngrok librosa==0.10.2 numpy scipy opencv-python-headless

# 1e. Patch Wav2Lip for modern PyTorch / librosa / numpy
import re, pathlib

def patch_file(path, replacements):
    text = pathlib.Path(path).read_text()
    for old, new in replacements:
        text = text.replace(old, new, 1)
    pathlib.Path(path).write_text(text)

# audio.py: librosa.filters.mel needs keyword sr=
patch_file("/content/Wav2Lip/audio.py", [
    ("return librosa.filters.mel(hp.sample_rate, hp.n_fft,",
     "return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,"),
])

# inference.py: torch.load needs weights_only=False
patch_file("/content/Wav2Lip/inference.py", [
    ("\t\tcheckpoint = torch.load(checkpoint_path)\n",
     "\t\tcheckpoint = torch.load(checkpoint_path, weights_only=False)\n"),
    ("\t\tcheckpoint = torch.load(checkpoint_path,\n"
     "\t\t\t\t\t\t\t\tmap_location=lambda storage, loc: storage)\n",
     "\t\tcheckpoint = torch.load(checkpoint_path,\n"
     "\t\t\t\t\t\t\t\tmap_location=lambda storage, loc: storage,\n"
     "\t\t\t\t\t\t\t\tweights_only=False)\n"),
])

# sfd_detector.py: torch.load needs weights_only=False
sfd_path = "/content/Wav2Lip/face_detection/detection/sfd/sfd_detector.py"
patch_file(sfd_path, [
    ("import os\nimport cv2\nfrom torch.utils.model_zoo import load_url",
     "import os\nimport cv2\nimport torch\nfrom torch.utils.model_zoo import load_url"),
    ("            model_weights = torch.load(path_to_detector)",
     "            model_weights = torch.load(path_to_detector, weights_only=False)"),
])

# Create temp directory
!mkdir -p /content/Wav2Lip/temp

print("✅ Wav2Lip setup complete!")
import torch
print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
'''

CELL_2_SERVER = r'''
# ============================================================
# CELL 2: Start the inference server with ngrok tunnel
# ============================================================

# ⚠️  SET YOUR NGROK AUTH TOKEN BELOW
# Get a free token at https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = ""  # <--- PASTE YOUR TOKEN HERE

import os, sys, uuid, subprocess, shutil, threading, time
from pathlib import Path

# Add Wav2Lip to path
sys.path.insert(0, "/content/Wav2Lip")
os.chdir("/content/Wav2Lip")

from flask import Flask, request, send_file, jsonify
from pyngrok import ngrok, conf

# ── Pre-load model once ──────────────────────────────────────
import torch
import numpy as np
import cv2
import audio
import face_detection
from models import Wav2Lip

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[server] Using device: {device}")

def _load_wav2lip_model(checkpoint_path):
    model = Wav2Lip()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    s = ckpt["state_dict"]
    new_s = {k.replace("module.", ""): v for k, v in s.items()}
    model.load_state_dict(new_s)
    del ckpt, s, new_s
    torch.cuda.empty_cache()
    return model.to(device).eval()

MODEL_PATH = "/content/Wav2Lip/checkpoints/wav2lip_gan.pth"
wav2lip_model = _load_wav2lip_model(MODEL_PATH)
print("[server] Wav2Lip model loaded on GPU ✅")

# ── Inference helpers ─────────────────────────────────────────
mel_step_size = 16

def _face_detect(images, pads, nosmooth, batch_size=16):
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device=device
    )
    while True:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(
                    detector.get_detections_for_batch(np.array(images[i : i + batch_size]))
                )
        except RuntimeError:
            if batch_size == 1:
                raise
            batch_size //= 2
            continue
        break
    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            raise ValueError("Face not detected in one or more frames.")
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])
    boxes = np.array(results)
    if not nosmooth:
        for i in range(len(boxes)):
            window = boxes[max(0, i - 4) : i + 5] if i + 5 <= len(boxes) else boxes[len(boxes) - 5 :]
            boxes[i] = np.mean(window, axis=0)
    results = [
        [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
        for image, (x1, y1, x2, y2) in zip(images, boxes)
    ]
    del detector
    torch.cuda.empty_cache()
    return results


def run_inference(face_path, audio_path, outfile, pads=(0, 10, 0, 0),
                  resize_factor=1, wav2lip_batch_size=128,
                  face_det_batch_size=16, nosmooth=False):
    """Full Wav2Lip inference, using the pre-loaded model."""
    img_size = 96

    # ── Read frames ──
    if face_path.lower().endswith((".jpg", ".png", ".jpeg")):
        full_frames = [cv2.imread(face_path)]
        fps = 25.0
        static = True
    else:
        cap = cv2.VideoCapture(face_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if resize_factor > 1:
                frame = cv2.resize(
                    frame,
                    (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor),
                )
            full_frames.append(frame)
        cap.release()
        static = False

    print(f"[inference] {len(full_frames)} frames @ {fps:.1f} fps")

    # ── Mel spectrogram ──
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    mel_chunks = []
    mel_idx_multiplier = 80.0 / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    full_frames = full_frames[: len(mel_chunks)]

    # ── Face detection ──
    if static:
        face_det_results = _face_detect([full_frames[0]], pads, nosmooth, face_det_batch_size)
    else:
        face_det_results = _face_detect(full_frames, pads, nosmooth, face_det_batch_size)

    # ── Batched inference ──
    frame_h, frame_w = full_frames[0].shape[:2]
    out = cv2.VideoWriter(
        "/content/Wav2Lip/temp/result.avi",
        cv2.VideoWriter_fourcc(*"DIVX"),
        fps,
        (frame_w, frame_h),
    )
    total_batches = int(np.ceil(len(mel_chunks) / wav2lip_batch_size))

    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    for idx, m in enumerate(mel_chunks):
        fidx = 0 if static else idx % len(full_frames)
        frame_to_save = full_frames[fidx].copy()
        face, coords = face_det_results[fidx if not static else 0].copy()
        face = cv2.resize(face, (img_size, img_size))
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size or idx == len(mel_chunks) - 1:
            ib = np.asarray(img_batch)
            mb = np.asarray(mel_batch)
            ib_masked = ib.copy()
            ib_masked[:, img_size // 2 :] = 0
            ib = np.concatenate((ib_masked, ib), axis=3) / 255.0
            mb = mb.reshape(len(mb), mb.shape[1], mb.shape[2], 1)

            ib_t = torch.FloatTensor(ib.transpose(0, 3, 1, 2)).to(device)
            mb_t = torch.FloatTensor(mb.transpose(0, 3, 1, 2)).to(device)

            with torch.no_grad():
                pred = wav2lip_model(mb_t, ib_t)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            del ib_t, mb_t
            torch.cuda.empty_cache()

            for p, f, c in zip(pred, frame_batch, coords_batch):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)

            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    out.release()

    # ── Mux audio ──
    cmd = f'ffmpeg -y -i "{audio_path}" -i /content/Wav2Lip/temp/result.avi -strict -2 -q:v 1 "{outfile}"'
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return os.path.exists(outfile)


# ── Flask App ─────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device, "model": "wav2lip_gan"})

@app.route("/lip_sync", methods=["POST"])
def lip_sync():
    job_id = uuid.uuid4().hex[:8]
    work_dir = Path(f"/content/jobs/{job_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Save uploaded files
        face_file = request.files.get("face")
        audio_file = request.files.get("audio")
        if not face_file or not audio_file:
            return jsonify({"error": "Both 'face' and 'audio' files are required."}), 400

        face_path = str(work_dir / face_file.filename)
        audio_path = str(work_dir / audio_file.filename)
        out_path = str(work_dir / "output.mp4")

        face_file.save(face_path)
        audio_file.save(audio_path)

        # Parse optional parameters
        pads = [int(x) for x in request.form.get("pads", "0,10,0,0").split(",")]
        resize_factor = int(request.form.get("resize_factor", "1"))
        wav2lip_batch_size = int(request.form.get("wav2lip_batch_size", "128"))
        face_det_batch_size = int(request.form.get("face_det_batch_size", "16"))
        nosmooth = request.form.get("nosmooth", "0") in {"1", "true", "yes"}

        print(f"[job {job_id}] face={face_file.filename} audio={audio_file.filename}")

        ok = run_inference(
            face_path, audio_path, out_path,
            pads=pads,
            resize_factor=resize_factor,
            wav2lip_batch_size=wav2lip_batch_size,
            face_det_batch_size=face_det_batch_size,
            nosmooth=nosmooth,
        )

        if not ok:
            return jsonify({"error": "Wav2Lip produced no output."}), 500

        return send_file(out_path, mimetype="video/mp4", as_attachment=True,
                         download_name=f"lip_synced_{job_id}.mp4")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up job files after response
        def _cleanup():
            time.sleep(5)
            shutil.rmtree(work_dir, ignore_errors=True)
        threading.Thread(target=_cleanup, daemon=True).start()


# ── Start ngrok tunnel and Flask ──────────────────────────────
if NGROK_AUTH_TOKEN:
    conf.get_default().auth_token = NGROK_AUTH_TOKEN
else:
    print("⚠️  No NGROK_AUTH_TOKEN set! Using ngrok without auth (rate-limited).")

public_url = ngrok.connect(5000, "http").public_url

print()
print("=" * 60)
print("🚀 Wav2Lip Remote Server is LIVE!")
print(f"   Public URL: {public_url}")
print()
print("   Copy this into your local .env:")
print(f"   WAV2LIP_REMOTE_URL={public_url}")
print("=" * 60)
print()

# Run Flask (blocking — keeps the cell running)
app.run(port=5000)
'''


# ─── Print instructions ──────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Wav2Lip Colab Remote Inference Server")
    print("=" * 70)
    print()
    print("Open a Google Colab notebook and create TWO cells:")
    print()
    print("─── CELL 1: Setup ───────────────────────────────────")
    print(CELL_1_SETUP)
    print()
    print("─── CELL 2: Start Server ────────────────────────────")
    print(CELL_2_SERVER)
    print()
    print("After running Cell 2, copy the printed ngrok URL into")
    print("your local .env as WAV2LIP_REMOTE_URL=<url>")
