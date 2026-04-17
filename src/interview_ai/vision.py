from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from interview_ai.schemas import VisionAnalysis, VisionFrameMetric


def _safe_mean(values: list[float]) -> float:
    return round(mean(values), 2) if values else 0.0


class VisionAnalyzer:
    def __init__(self) -> None:
        self._landmarker = None

    def _ensure_landmarker(self):
        if self._landmarker is not None:
            return self._landmarker
        import mediapipe as mp

        self._landmarker = mp
        return self._landmarker

    def analyze_precomputed(self, json_path: str) -> VisionAnalysis:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        frames = [VisionFrameMetric(**item) for item in payload.get("frames", [])]
        return self._aggregate(frames)

    def analyze_images(self, image_paths: list[str]) -> VisionAnalysis:
        mp = self._ensure_landmarker()
        frames: list[VisionFrameMetric] = []

        for index, image_path in enumerate(image_paths):
            image = mp.Image.create_from_file(image_path)
            # Placeholder skeleton: hook MediaPipe Face Landmarker here when model asset is available.
            brightness_proxy = min(1.0, max(0.0, image.numpy_view().mean() / 255.0))
            frames.append(
                VisionFrameMetric(
                    timestamp=float(index),
                    eye_open_ratio=round(0.5 + 0.4 * brightness_proxy, 2),
                    smile_score=round(0.3 + 0.3 * brightness_proxy, 2),
                    gaze_forward_score=round(0.6, 2),
                    head_down_score=round(0.2, 2),
                    face_present=True,
                )
            )
        return self._aggregate(frames)

    def _aggregate(self, frames: list[VisionFrameMetric]) -> VisionAnalysis:
        eye_contact = [frame.gaze_forward_score for frame in frames if frame.face_present]
        attention = [max(0.0, 1.0 - frame.head_down_score) * frame.eye_open_ratio for frame in frames if frame.face_present]
        smiles = [frame.smile_score for frame in frames if frame.face_present]
        head_down = [frame.head_down_score for frame in frames if frame.face_present]
        presence = [1.0 if frame.face_present else 0.0 for frame in frames]
        return VisionAnalysis(
            frames=frames,
            attention_score=_safe_mean(attention),
            eye_contact_score=_safe_mean(eye_contact),
            smile_frequency=_safe_mean(smiles),
            head_down_ratio=_safe_mean(head_down),
            face_presence_ratio=_safe_mean(presence),
        )
