"""Idempotent setup script for the Wav2Lip lip-sync integration.

Performs the steps that the upstream Wav2Lip repo does NOT ship out of the box:

1. Clone https://github.com/Rudrabha/Wav2Lip into ``third_party/Wav2Lip``.
2. Download ``wav2lip_gan.pth`` and ``s3fd.pth`` from public Hugging Face mirrors
   (the upstream repo gitignores ``*.pth``).
3. Verify SHA-256 digests match the values published on Hugging Face.
4. Apply small in-place patches that make the upstream code compatible with
   modern PyTorch (>=2.6) / librosa (>=0.10) / numpy (>=2). These patches are
   idempotent and can be safely re-run.
5. Write / update the project ``.env`` with ``WAV2LIP_DIR``,
   ``WAV2LIP_CHECKPOINT``, and ``LIP_SYNC_ENABLED=1`` so Phase 3 is ready to go.
6. Optionally pip-install ``librosa`` (the only Wav2Lip dep that is not already
   in this project's environment for typical PyTorch installs).

Re-running is safe: existing files are reused unless ``--force`` is passed,
patches detect their own marker so they apply at most once, and ``.env`` keys
are updated in-place without touching unrelated lines.

Usage::

    python scripts/setup_wav2lip.py                  # full setup, leaves deps to user
    python scripts/setup_wav2lip.py --install-deps   # also pip install librosa
    python scripts/setup_wav2lip.py --force          # re-download checkpoints
    python scripts/setup_wav2lip.py --no-env         # skip .env updates
    python scripts/setup_wav2lip.py --no-patch       # skip Wav2Lip code patches
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
WAV2LIP_DIR = THIRD_PARTY_DIR / "Wav2Lip"
WAV2LIP_GIT_URL = "https://github.com/Rudrabha/Wav2Lip.git"

CHECKPOINTS_DIR = WAV2LIP_DIR / "checkpoints"
S3FD_DIR = WAV2LIP_DIR / "face_detection" / "detection" / "sfd"
WAV2LIP_GAN_PATH = CHECKPOINTS_DIR / "wav2lip_gan.pth"
S3FD_PATH = S3FD_DIR / "s3fd.pth"

WAV2LIP_GAN_URL = "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true"
WAV2LIP_GAN_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"

S3FD_URL = "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/s3fd-619a316812.pth"
S3FD_SHA256 = "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543"

ENV_PATH = REPO_ROOT / ".env"
EXAMPLE_ENV_PATH = REPO_ROOT / ".example.env"


def log(msg: str) -> None:
	print(f"[setup_wav2lip] {msg}", flush=True)


def ensure_git() -> None:
	if shutil.which("git") is None:
		raise SystemExit(
			"git is required to clone the Wav2Lip repository but was not found on PATH."
		)


def clone_wav2lip(force: bool) -> None:
	if WAV2LIP_DIR.exists() and not force:
		log(f"Wav2Lip already cloned at {WAV2LIP_DIR}")
		return
	if WAV2LIP_DIR.exists() and force:
		log(f"Removing existing Wav2Lip clone at {WAV2LIP_DIR}")
		shutil.rmtree(WAV2LIP_DIR)
	THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)
	log(f"Cloning Wav2Lip into {WAV2LIP_DIR}")
	subprocess.run(
		["git", "clone", "--depth", "1", WAV2LIP_GIT_URL, str(WAV2LIP_DIR)],
		check=True,
	)


def sha256_of(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _human_bytes(num: int) -> str:
	for unit in ("B", "KB", "MB", "GB"):
		if num < 1024:
			return f"{num:.1f} {unit}"
		num /= 1024
	return f"{num:.1f} TB"


def download_with_progress(url: str, target: Path) -> None:
	target.parent.mkdir(parents=True, exist_ok=True)
	log(f"Downloading {url}")
	log(f"          -> {target}")

	def _hook(block_num: int, block_size: int, total_size: int) -> None:
		if total_size <= 0:
			return
		downloaded = min(block_num * block_size, total_size)
		pct = downloaded * 100.0 / total_size
		end = "\n" if downloaded >= total_size else "\r"
		print(
			f"  {pct:6.2f}%  {_human_bytes(downloaded)} / {_human_bytes(total_size)}",
			end=end,
			flush=True,
		)

	tmp_target = target.with_suffix(target.suffix + ".part")
	try:
		urllib.request.urlretrieve(url, tmp_target, reporthook=_hook)
	except Exception:
		if tmp_target.exists():
			tmp_target.unlink()
		raise
	tmp_target.replace(target)


def download_checkpoint(url: str, target: Path, expected_sha256: str, force: bool) -> None:
	if target.exists() and not force:
		actual = sha256_of(target)
		if actual == expected_sha256:
			log(f"Already present and verified: {target.name}")
			return
		log(
			f"Existing {target.name} hash mismatch (got {actual[:12]}..., "
			f"expected {expected_sha256[:12]}...); re-downloading."
		)
		target.unlink()
	download_with_progress(url, target)
	actual = sha256_of(target)
	if actual != expected_sha256:
		target.unlink(missing_ok=True)
		raise SystemExit(
			f"SHA-256 mismatch for {target.name}: got {actual}, expected {expected_sha256}"
		)
	log(f"Verified {target.name} (sha256 ok)")


PATCH_MARKER = "# patched-by-setup_wav2lip"


def _patch_file(path: Path, replacements: list[tuple[str, str]]) -> bool:
	"""Apply a list of (old, new) replacements to a file. Idempotent.

	Each replacement is applied at most once. Patches that are already in place
	(``old`` no longer present) are silently skipped, so re-runs after the
	script learns about new compatibility issues will still upgrade the file.
	Returns True iff the file was modified.
	"""
	if not path.exists():
		log(f"Skip patch (missing): {path}")
		return False
	original = path.read_text(encoding="utf-8")
	updated = original
	for old, new in replacements:
		# Replacement is naturally idempotent: once ``old`` is replaced it no
		# longer matches on subsequent runs. We don't gate on ``new`` being
		# absent because the new snippet may legitimately appear elsewhere
		# in the file (e.g. shell=True is used by other call sites too).
		if old in updated:
			updated = updated.replace(old, new, 1)
	if updated == original:
		return False
	header = f"{PATCH_MARKER} {path.name}\n"
	if not updated.startswith(header):
		updated = header + updated
	path.write_text(updated, encoding="utf-8")
	return True


def apply_patches() -> None:
	"""Patch upstream Wav2Lip for modern librosa / numpy / PyTorch compatibility."""
	log("Applying Wav2Lip compatibility patches")

	audio_py = WAV2LIP_DIR / "audio.py"
	audio_replacements = [
		(
			"return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,\n"
			"                               fmin=hp.fmin, fmax=hp.fmax)",
			"return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,\n"
			"                               fmin=hp.fmin, fmax=hp.fmax)",
		),
	]
	if _patch_file(audio_py, audio_replacements):
		log(f"  patched {audio_py.name} (librosa.filters.mel kwargs)")
	else:
		log(f"  {audio_py.name} already patched or unchanged")

	inference_py = WAV2LIP_DIR / "inference.py"
	inference_replacements = [
		(
			"\t\tcheckpoint = torch.load(checkpoint_path)\n",
			"\t\tcheckpoint = torch.load(checkpoint_path, weights_only=False)\n",
		),
		(
			"\t\tcheckpoint = torch.load(checkpoint_path,\n"
			"\t\t\t\t\t\t\t\tmap_location=lambda storage, loc: storage)\n",
			"\t\tcheckpoint = torch.load(checkpoint_path,\n"
			"\t\t\t\t\t\t\t\tmap_location=lambda storage, loc: storage,\n"
			"\t\t\t\t\t\t\t\tweights_only=False)\n",
		),
		# Windows bug in upstream: passing a string command with shell=False fails.
		# Force shell=True so the same command works on all platforms.
		(
			"subprocess.call(command, shell=platform.system() != 'Windows')",
			"subprocess.call(command, shell=True)",
		),
	]
	if _patch_file(inference_py, inference_replacements):
		log(f"  patched {inference_py.name} (torch.load weights_only)")
	else:
		log(f"  {inference_py.name} already patched or unchanged")

	sfd_py = WAV2LIP_DIR / "face_detection" / "detection" / "sfd" / "sfd_detector.py"
	sfd_replacements = [
		(
			"import os\nimport cv2\nfrom torch.utils.model_zoo import load_url",
			"import os\nimport cv2\nimport torch\nfrom torch.utils.model_zoo import load_url",
		),
		(
			"            model_weights = torch.load(path_to_detector)",
			"            model_weights = torch.load(path_to_detector, weights_only=False)",
		),
	]
	if _patch_file(sfd_py, sfd_replacements):
		log(f"  patched {sfd_py.name} (import torch + weights_only)")
	else:
		log(f"  {sfd_py.name} already patched or unchanged")

	# Wav2Lip's inference writes intermediate files to ``temp/`` relative to its repo.
	(WAV2LIP_DIR / "temp").mkdir(exist_ok=True)


def install_deps() -> None:
	log("Installing Wav2Lip Python dependencies (librosa)")
	cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "librosa"]
	log("  " + " ".join(cmd))
	subprocess.run(cmd, check=True)


def update_env(write_env: bool) -> None:
	if not write_env:
		log("Skipping .env update (--no-env)")
		return
	if not ENV_PATH.exists() and EXAMPLE_ENV_PATH.exists():
		log(f"Creating .env from {EXAMPLE_ENV_PATH.name}")
		ENV_PATH.write_text(EXAMPLE_ENV_PATH.read_text(encoding="utf-8"), encoding="utf-8")

	values = {
		"LIP_SYNC_ENABLED": "1",
		"WAV2LIP_DIR": str(WAV2LIP_DIR.resolve()),
		"WAV2LIP_CHECKPOINT": str(WAV2LIP_GAN_PATH.resolve()),
	}

	if ENV_PATH.exists():
		lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
	else:
		lines = []

	updated_keys: set[str] = set()
	for index, line in enumerate(lines):
		stripped = line.lstrip()
		if not stripped or stripped.startswith("#"):
			continue
		if "=" not in line:
			continue
		key = line.split("=", 1)[0].strip()
		if key in values:
			lines[index] = f"{key}={values[key]}"
			updated_keys.add(key)

	for key, value in values.items():
		if key not in updated_keys:
			lines.append(f"{key}={value}")

	ENV_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
	log(f"Updated {ENV_PATH.name} with WAV2LIP_DIR / WAV2LIP_CHECKPOINT / LIP_SYNC_ENABLED=1")


def post_install_notes() -> None:
	print()
	log("Setup complete.")
	print()
	print("Next steps:")
	print("  Wav2Lip needs torch, torchvision, librosa, opencv-python, numpy, numba,")
	print("  scipy, and tqdm. Most are already present in a typical PyTorch install.")
	print("  If the smoke test (or run_phase3.py) reports 'No module named librosa', run:")
	print()
	print(f"    {sys.executable} -m pip install librosa")
	print("  (or re-run this script with --install-deps).")
	print()
	print("  If you'd rather keep Wav2Lip's deps isolated, create a separate venv and")
	print("  set WAV2LIP_PYTHON in .env to that interpreter. Example:")
	print()
	print("    python -m venv third_party\\wav2lip-venv")
	print("    third_party\\wav2lip-venv\\Scripts\\activate")
	print("    pip install torch torchvision librosa opencv-python numpy numba scipy tqdm")
	print(f"    # then in .env: WAV2LIP_PYTHON={(THIRD_PARTY_DIR / 'wav2lip-venv' / 'Scripts' / 'python.exe').resolve()}")
	print()
	print("  Once deps are in place, run Phase 3 as usual:")
	print("    python run_phase3.py --input data/outputs/phase2/<project_id>/state.json")


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Re-clone Wav2Lip and re-download checkpoints even if present.",
	)
	parser.add_argument(
		"--no-env",
		action="store_true",
		help="Do not create or modify the project .env file.",
	)
	parser.add_argument(
		"--no-patch",
		action="store_true",
		help="Do not apply Wav2Lip code compatibility patches.",
	)
	parser.add_argument(
		"--install-deps",
		action="store_true",
		help="Pip install librosa (the only Wav2Lip dep typically not already present).",
	)
	args = parser.parse_args()

	ensure_git()
	clone_wav2lip(force=args.force)
	download_checkpoint(WAV2LIP_GAN_URL, WAV2LIP_GAN_PATH, WAV2LIP_GAN_SHA256, args.force)
	download_checkpoint(S3FD_URL, S3FD_PATH, S3FD_SHA256, args.force)
	if not args.no_patch:
		apply_patches()
	if args.install_deps:
		install_deps()
	update_env(write_env=not args.no_env)
	post_install_notes()
	return 0


if __name__ == "__main__":
	sys.exit(main())
