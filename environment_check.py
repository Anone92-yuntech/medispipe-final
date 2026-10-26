"""Check runtime imports/assets; optionally prepare assets and save provenance."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np
import PIL
import scipy
import filterpy

from person_detector import PersonDetector
from ocsort_adapter import OCSortTracker, OCSORT_COMMIT, OCSORT_REPOSITORY
from tracking_pipeline import BASE_DIR, DEFAULT_MODEL, MODEL_URL, ensure_model
from behavior_engine import BehaviorEngine
from behavior_csv import TimeCsvLogger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Download missing models and save environment metadata')
    args = parser.parse_args()
    if args.prepare:
        pose_path = ensure_model(DEFAULT_MODEL)
        detector = PersonDetector()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(blank)
        assert len(detections) == 0, 'Unexpected person on a black image'
        tracker = OCSortTracker(fps=30)
        assert tracker.update(detections, blank.shape, 0) == []
        with mp.tasks.vision.PoseLandmarker.create_from_options(mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=pose_path.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )) as pose:
            result = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=blank))
            assert not result.pose_landmarks
        packages = {d.metadata['Name']: d.version for d in metadata.distributions() if d.metadata['Name']}
        payload = {
            'python': sys.version, 'executable': sys.executable,
            'packages': dict(sorted(packages.items())),
            'detector': detector.provenance,
            'ocsort': {'repository': OCSORT_REPOSITORY, 'commit': OCSORT_COMMIT,
                       'provenance': 'vendor/ocsort/PROVENANCE.json'},
            'pose_model': {'path': str(pose_path), 'source': MODEL_URL,
                           'sha256': hashlib.sha256(pose_path.read_bytes()).hexdigest()},
            'gui': next((line.strip() for line in cv2.getBuildInformation().splitlines() if 'GUI:' in line), ''),
        }
        (BASE_DIR / 'ENVIRONMENT.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('Environment/assets verified; ENVIRONMENT.json updated. Ready: run_pose.bat')
    else:
        # No downloads or report writes in the launcher's lightweight check.
        assert hasattr(cv2, 'imshow') and hasattr(mp.tasks.vision, 'PoseLandmarker')
        print('Runtime imports OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
