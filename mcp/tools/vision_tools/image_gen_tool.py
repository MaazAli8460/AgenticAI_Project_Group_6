from __future__ import annotations

import base64
import os
import urllib.parse
from pathlib import Path
from typing import Optional

import httpx


class ImageGenTool:
	def __init__(
		self,
		provider: Optional[str] = None,
		base_url: Optional[str] = None,
		timeout_s: float = 120.0,
	) -> None:
		self._provider = (provider or os.getenv("IMAGE_PROVIDER", "auto")).lower()
		self._client = httpx.Client(timeout=timeout_s)
		self._pollinations_base = base_url or os.getenv(
			"POLLINATIONS_BASE_URL", "https://image.pollinations.ai"
		)
		self._pollinations_model = os.getenv("POLLINATIONS_MODEL")
		self._zsky_api_url = os.getenv(
			"ZSKY_API_URL", "https://zsky.ai/api/v1/image/generate"
		)
		self._zsky_api_urls = [
			item.strip()
			for item in os.getenv("ZSKY_API_URLS", "").split(",")
			if item.strip()
		]
		self._zsky_api_key = os.getenv("ZSKY_API_KEY")
		self._zeskit_api_url = os.getenv("ZESKIT_API_URL")
		self._zeskit_api_key = os.getenv("ZESKIT_API_KEY")

	def generate(
		self,
		prompt: str,
		output_path: Path,
		width: int,
		height: int,
		seed: Optional[int] = None,
	) -> Path:
		if self._provider == "pollinations":
			return self._generate_pollinations(prompt, output_path, width, height, seed)
		if self._provider in {"zeskit", "zeskit.ai", "auto"}:
			try:
				return self._generate_zeskit(prompt, output_path, width, height, seed)
			except Exception as exc:  # noqa: BLE001 - intentional fallback
				try:
					return self._generate_pollinations(
						prompt, output_path, width, height, seed
					)
				except Exception as fallback_exc:  # noqa: BLE001 - report combined
					raise RuntimeError(
						"Zeskit image generation failed; Pollinations fallback failed."
					) from fallback_exc
		if self._provider == "zsky":
			return self._generate_zsky(prompt, output_path, width, height, seed)
		raise RuntimeError(f"Unsupported IMAGE_PROVIDER: {self._provider}")

	def _generate_zeskit(
		self,
		prompt: str,
		output_path: Path,
		width: int,
		height: int,
		seed: Optional[int],
	) -> Path:
		if not self._zeskit_api_url:
			raise RuntimeError("ZESKIT_API_URL is required for IMAGE_PROVIDER=zeskit")
		headers = {}
		if self._zeskit_api_key:
			headers["Authorization"] = f"Bearer {self._zeskit_api_key}"
		payload = {
			"prompt": prompt,
			"width": width,
			"height": height,
		}
		if seed is not None:
			payload["seed"] = seed
		response = self._client.post(self._zeskit_api_url, json=payload, headers=headers)
		response.raise_for_status()
		content_type = response.headers.get("content-type", "").lower()
		if "application/json" in content_type:
			data = response.json()
			image_url = data.get("image_url") or data.get("url")
			if image_url:
				image_response = self._client.get(image_url)
				image_response.raise_for_status()
				image_bytes = image_response.content
			else:
				image_b64 = (
					data.get("image_base64")
					or data.get("image")
					or data.get("base64")
				)
				if not image_b64:
					raise RuntimeError("Zeskit response missing image data")
				if isinstance(image_b64, str) and image_b64.startswith("data:"):
					image_b64 = image_b64.split(",", 1)[-1]
				image_bytes = base64.b64decode(image_b64)
		else:
			image_bytes = response.content
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_bytes(image_bytes)
		return output_path

	def _generate_pollinations(
		self,
		prompt: str,
		output_path: Path,
		width: int,
		height: int,
		seed: Optional[int],
	) -> Path:
		encoded = urllib.parse.quote(prompt, safe="")
		url = f"{self._pollinations_base}/prompt/{encoded}"
		params: dict[str, str] = {"width": str(width), "height": str(height)}
		if seed is not None:
			params["seed"] = str(seed)
		if self._pollinations_model:
			params["model"] = self._pollinations_model
		response = self._client.get(url, params=params)
		response.raise_for_status()
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_bytes(response.content)
		return output_path

	def _generate_zsky(
		self,
		prompt: str,
		output_path: Path,
		width: int,
		height: int,
		seed: Optional[int],
	) -> Path:
		api_urls = self._zsky_api_urls or [self._zsky_api_url]
		if not api_urls or not api_urls[0]:
			raise RuntimeError("ZSKY_API_URL is required for IMAGE_PROVIDER=zsky")
		headers = {
			"Accept": "image/*,application/json",
			"Origin": "https://zsky.ai",
			"Referer": "https://zsky.ai/",
			"User-Agent": (
				"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
				"AppleWebKit/537.36 (KHTML, like Gecko) "
				"Chrome/124.0.0.0 Safari/537.36"
			),
		}
		if self._zsky_api_key:
			headers["Authorization"] = f"Bearer {self._zsky_api_key}"
		payload = {
			"prompt": prompt,
			"width": width,
			"height": height,
			"seed": seed,
		}
		last_error: Optional[str] = None
		for api_url in api_urls:
			response = self._client.post(api_url, json=payload, headers=headers)
			if response.status_code >= 400:
				last_error = (
					f"{response.status_code} {response.text[:200]}".strip()
				)
				continue
			content_type = response.headers.get("content-type", "").lower()
			if "application/json" in content_type:
				data = response.json()
				image_url = data.get("image_url") or data.get("url")
				if not image_url:
					raise RuntimeError("ZSKY response missing image URL")
				image_response = self._client.get(image_url, headers=headers)
				image_response.raise_for_status()
				image_bytes = image_response.content
			else:
				image_bytes = response.content
			output_path.parent.mkdir(parents=True, exist_ok=True)
			output_path.write_bytes(image_bytes)
			return output_path
		error_detail = last_error or "Unknown error"
		raise RuntimeError(f"ZSKY image generation failed: {error_detail}")
