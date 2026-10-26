"""OC-SORT shoulder-selected target with a display-only FHD pose preview."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from functools import lru_cache
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from mediapipe.tasks.python import vision
    from person_detector import PersonDetector, DETECTOR_PRESETS, DEFAULT_DETECTOR_PRESET
    from tracking_pipeline import (
        BASE_DIR, DEFAULT_MODEL, FrameAnalysis, PipelineConfig, TrackingPipeline,
        ensure_model,
    )
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc}\nRun setup_env.bat first.") from exc

FHD_WIDTH, FHD_HEIGHT = 1920, 1080
VIDEO_WIDTH = 1500
WINDOW_NAME = "OC-SORT + MediaPipe - FHD 1920x1080"
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm",
    ".mpg", ".mpeg", ".mts", ".m2ts", ".ts", ".3gp",
}
STATE_LABELS = {
    "waiting": "等待人物", "collecting": "蒐集肩寬樣本",
    "locked": "目標鎖定", "lost": "目標遺失",
    "pose_uncertain": "姿態遮擋／無法判定", "uncertain": "身分不確定／保留 ID",
    "reinitializing": "自動重新初始化",
}
REASONS = {
    "waiting_for_people": "等待穩定的人物偵測",
    "collecting_valid_shoulders": "等待至少一人的肩寬樣本達標",
    "only_qualified_candidate": "僅一人肩寬達標，鎖定該 ID",
    "smallest_qualified_candidate": "選擇合格候選人中肩寬最小者",
    "shoulder_sizes_too_similar": "肩寬差距不足，繼續觀察",
    "target_locked": "依初始化肩寬選定，持續追蹤該 ID",
    "target_lost": "當幀沒有目標的有效偵測，保留 ID",
    "target_identity_uncertain_reset_required": "軌跡跳動，暫停分析；按 R 手動確認重選",
    "identity_jump_detected": "軌跡不連續，暫停姿態分析",
    "pose_anchor_jumped": "頭肩位置跳動，等待可信位置恢復",
    "pose_inference_error": "MediaPipe 推論失敗，將自動恢復",
    "pose_recovery_retry": "重建目標姿態分析，保留原 ID 與坐姿基準",
    "posture_landmarks_missing": "頭部或前臂資料不足，保留 ID 等待恢復",
    "auto_reset_tracking_lost": "目標追蹤遺失，重新蒐集肩寬",
    "auto_reset_pose_failed": "骨架持續失敗，重新蒐集肩寬",
    "auto_reset_identity_uncertain": "身分無法確認，重新蒐集肩寬",
    "not_sampled": "尚未取樣", "low_shoulder_confidence": "肩膀信心不足",
    "missing_shoulders": "缺少雙肩", "nonfinite_shoulder_data": "肩膀資料無效",
    "shoulders_outside_person": "肩膀超出人物範圍",
    "head_outside_person": "頭部超出人物範圍",
    "head_shoulder_geometry": "頭肩位置不合理",
    "implausible_shoulder_width": "肩寬不合理", "shoulders_not_comparable": "肩膀角度不適合比較",
    "No pose detected": "未偵測到骨架", "Invalid crop": "裁切範圍無效",
    "Invalid target crop": "目標裁切範圍無效，保留 ID 等待恢復",
    "pose_unavailable": "沒有可信姿態，等待恢復",
    "head_unavailable": "頭部關鍵點不足",
    "head_geometry_invalid": "頭部位置資料無效",
    "invalid_person_box": "人物框無效",
    "baseline_shoulders_unavailable": "建立基準需要可信雙肩",
    "baseline_shoulders_outside_person": "基準肩膀超出人物框",
    "baseline_shoulder_width_invalid": "基準肩寬無效",
    "upright_baseline_required": "請等待正常坐姿以建立基準",
    "collecting_upright_baseline": "蒐集正常坐姿的頭部與肩寬基準",
    "forearm_unavailable": "沒有可信的肘部至手腕線段",
    "posture_conditions_not_met": "頭部下降或靠近前臂條件尚未同時符合",
    "holding_posture": "頭部已下降且靠近前臂，持續計時中",
    "suspected_table_rest": "符合三項條件：疑似趴桌",
    "Following selected OC-SORT ID": "只分析鎖定目標，不重新選最小者",
}
_SIDEBAR_CACHE = {}
POSTURE_LABELS = {
    "calibrating": "建立坐姿基準", "normal": "未符合趴桌條件",
    "candidate": "疑似趴桌／計時中", "resting": "疑似趴桌／條件成立",
    "unknown": "遮擋／無法判定",
}


def reset_sidebar_cache():
    _SIDEBAR_CACHE.clear()


def find_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")
    return sorted((p for p in input_path.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
                  key=lambda p: p.name.casefold())


def display_geometry(frame_shape):
    source_h, source_w = frame_shape[:2]
    scale = min(VIDEO_WIDTH / source_w, FHD_HEIGHT / source_h)
    width = min(VIDEO_WIDTH, max(1, round(source_w * scale)))
    height = min(FHD_HEIGHT, max(1, round(source_h * scale)))
    return width, height, (VIDEO_WIDTH - width) // 2, (FHD_HEIGHT - height) // 2


def to_display(points, frame_shape):
    width, height, left, top = display_geometry(frame_shape)
    source_h, source_w = frame_shape[:2]
    return np.asarray(points, dtype=float) * [width / source_w, height / source_h] + [left, top]


@lru_cache(maxsize=8)
def _font(size):
    for candidate in (Path("C:/Windows/Fonts/msjh.ttc"), Path("C:/Windows/Fonts/msyh.ttc")):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.truetype("DejaVuSans.ttf", size=size)


def _reason(text):
    if text.startswith("ambiguous_with_id_"):
        return "骨架可能屬於鄰近 ID " + text.rsplit("_", 1)[-1]
    return REASONS.get(text, text)


@lru_cache(maxsize=512)
def _wrapped_lines(text, size, width):
    """Measure a complete line first; cache static labels and long filenames."""
    font = _font(size)
    result = []
    for paragraph in str(text).split('\n'):
        if draw_length := font.getlength(paragraph):
            if draw_length <= width:
                result.append(paragraph)
                continue
        elif not paragraph:
            result.append('')
            continue
        buffer = ''
        for character in paragraph:
            if buffer and font.getlength(buffer + character) > width:
                result.append(buffer)
                buffer = character
            else:
                buffer += character
        result.append(buffer)
    return tuple(result)


def _render_sidebar(analysis, video_path, index, count, frame_index,
                    fps, processing_fps, config):
    panel = Image.new("RGB", (FHD_WIDTH - VIDEO_WIDTH, FHD_HEIGHT), (22, 25, 31))
    draw = ImageDraw.Draw(panel)
    y = 18

    def line(text, color=(225, 230, 235), size=19, space=6):
        nonlocal y
        font = _font(size)
        for row in _wrapped_lines(str(text), size, 382):
            draw.text((18, y), row, fill=color, font=font)
            y += size + 7
        y += space

    line("OC-SORT ＋ MediaPipe", (100, 225, 255), 24)
    line(f"影片 {index}/{count}：{video_path.name}", size=17)
    line(f"時間 {analysis.timestamp_ms / 1000:.2f} 秒　影格 {frame_index + 1}", size=17)
    line(f"來源 {fps:.1f} FPS／處理 {processing_fps:.1f} FPS", size=17)
    if config.detector_label:
        line(f"人物模型：{config.detector_label}", size=16)
    state_color = (115, 255, 140) if analysis.state == 'locked' else (255, 208, 115)
    line(STATE_LABELS.get(analysis.state, analysis.state), state_color, 23)
    line(f"目標 ID：{analysis.target_id if analysis.target_id is not None else '尚未選定'}", size=21)
    line(_reason(analysis.detail), (255, 208, 115), 17)
    line(f"初始化：{config.init_seconds:g} 秒／至少 {config.min_samples} 筆", size=16)
    line(f"信心門檻 {config.landmark_confidence:g}／肩寬差距 {config.min_gap:.0%}", size=16)
    line(f"追蹤遺失 {config.auto_reset_seconds:g} 秒後重選／共 {analysis.auto_reset_count} 次", size=15)
    line(f"姿態不足 {analysis.pose_failure_seconds:.1f}／{config.pose_recovery_seconds:g} 秒；恢復 {analysis.pose_recovery_count} 次", size=15)
    if analysis.auto_reset_reason:
        line(_reason(analysis.auto_reset_reason), (170, 185, 200), 15)
    posture = analysis.posture
    if posture is not None:
        color = (255, 145, 100) if posture.state == 'resting' else (130, 220, 255)
        line("動作：" + POSTURE_LABELS.get(posture.state, posture.state), color, 21)
        if posture.baseline_head_y is None:
            line(f"坐姿基準：{posture.baseline_samples} 筆／需 {config.posture_baseline_seconds:g} 秒", size=15)
        else:
            drop = '--' if posture.head_drop_ratio is None else f'{posture.head_drop_ratio:.2f}'
            distance = '--' if posture.forearm_distance_ratio is None else f'{posture.forearm_distance_ratio:.2f}'
            line(f"頭部下降 {drop}／至少 {config.posture_head_drop_ratio:g}", size=15)
            line(f"前臂距離 {distance}／上限 {config.posture_forearm_distance_ratio:g}", size=15)
            line(f"持續 {posture.held_seconds:.1f}／{config.posture_hold_seconds:g} 秒（以基準肩寬換算）", size=15)
        line(_reason(posture.detail), (170, 185, 200), 15)
    for track in analysis.tracks:
        if y > 775:
            break
        data = analysis.statistics.get(track.track_id, {})
        color = (115, 255, 140) if track.track_id == analysis.target_id else (215, 220, 230)
        line(f"ID {track.track_id}　{'當幀有偵測' if track.observed else '預測框（不做骨架）'}", color, 16, space=2)
        width = data.get("median_width")
        width_text = f'{width:.1f} px' if width is not None else '--'
        line(f"肩寬 {width_text}／{data.get('sample_count', 0)} 筆／{data.get('valid_seconds', 0):.2f} 秒", color, 15)
        if analysis.target_id is None and data.get("rejection_reason"):
            line(_reason(data["rejection_reason"]), (255, 185, 115), 15)
    draw.rectangle((0, 873, panel.width, FHD_HEIGHT), fill=(22, 25, 31))
    y = 890
    line("空白鍵：暫停／繼續", size=18)
    line("追蹤遺失自動重選／姿態不足保留 ID", size=16)
    line("R：手動重選　N：下一部", size=16)
    line("Q／Esc／關閉視窗：離開", size=18)
    line("FHD 1920×1080｜不儲存影片｜無音訊", (150, 160, 170), 15)
    return np.asarray(panel)[:, :, ::-1].copy()


def _draw_sidebar(canvas, analysis, video_path, index, count, frame_index,
                  fps, processing_fps, config):
    # Refresh numeric readouts at 5 Hz of source time. Identity/status changes
    # bypass the timer; R and each new video explicitly clear the cache.
    stats_state = tuple(sorted(
        (track_id, bool(data.get('sample_count')), bool(data.get('ready')),
         data.get('rejection_reason', ''))
        for track_id, data in analysis.statistics.items()
    ))
    posture_state = ((analysis.posture.state, analysis.posture.detail)
                     if analysis.posture is not None else None)
    key = (str(video_path), index, count, analysis.timestamp_ms // 200,
           analysis.target_id, analysis.state, analysis.detail, fps, config,
           analysis.selection_generation, analysis.auto_reset_count,
           analysis.pose_recovery_count, posture_state,
           tuple((t.track_id, t.observed) for t in analysis.tracks), stats_state)
    if key != _SIDEBAR_CACHE.get('key'):
        _SIDEBAR_CACHE['image'] = _render_sidebar(
            analysis, video_path, index, count, frame_index, fps, processing_fps, config
        )
        _SIDEBAR_CACHE['key'] = key
    canvas[:, VIDEO_WIDTH:] = _SIDEBAR_CACHE['image']


def render_fhd(frame, analysis, video_path=Path("video"), index=1, count=1,
               frame_index=0, fps=30.0, processing_fps=0.0, config=None, *, sidebar=True):
    canvas = np.zeros((FHD_HEIGHT, FHD_WIDTH, 3), dtype=np.uint8)
    width, height, left, top = display_geometry(frame.shape)
    canvas[top:top + height, left:left + width] = cv2.resize(frame, (width, height))
    for track in analysis.tracks:
        box = np.rint(to_display(np.asarray(track.bbox).reshape(2, 2), frame.shape)).astype(int)
        box[:, 0] = np.clip(box[:, 0], left, left + width - 1)
        box[:, 1] = np.clip(box[:, 1], top, top + height - 1)
        color = (80, 255, 80) if track.track_id == analysis.target_id else (255, 190, 70)
        if not track.observed:
            color = (120, 120, 120)
        cv2.rectangle(canvas, tuple(box[0]), tuple(box[1]), color, 2, cv2.LINE_AA)
        label = f"ID {track.track_id}" + (" [TARGET]" if track.track_id == analysis.target_id else "")
        if not track.observed:
            label += " [PREDICTED]"
        anchor = (int(box[0, 0]), max(top + 22, int(box[0, 1]) - 8))
        cv2.putText(canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, .65, color, 1, cv2.LINE_AA)
    if analysis.pose is not None:
        pose = analysis.pose
        points = to_display(pose.points, frame.shape)
        source_h, source_w = frame.shape[:2]
        valid = (np.isfinite(points).all(axis=1)
                 & np.isfinite(pose.visibility) & np.isfinite(pose.presence)
                 & (pose.visibility >= .5) & (pose.presence >= .5)
                 & (pose.points[:, 0] >= 0) & (pose.points[:, 0] < source_w)
                 & (pose.points[:, 1] >= 0) & (pose.points[:, 1] < source_h))
        for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
            a, b = connection.start, connection.end
            if a < len(points) and b < len(points) and valid[a] and valid[b]:
                cv2.line(canvas, tuple(np.rint(points[a]).astype(int)),
                         tuple(np.rint(points[b]).astype(int)), (50, 255, 80), 3, cv2.LINE_AA)
        for point in points[valid]:
            position = tuple(np.rint(point).astype(int))
            cv2.circle(canvas, position, 5, (10, 10, 10), -1, cv2.LINE_AA)
            cv2.circle(canvas, position, 3, (60, 255, 255), -1, cv2.LINE_AA)
    if sidebar:
        _draw_sidebar(canvas, analysis, video_path, index, count, frame_index,
                      fps, processing_fps, config or PipelineConfig())
    return canvas


def show_frame(canvas, delay_ms, on_reset=None, on_seat=None):
    """UI waits never advance tracking or source time, including while paused."""
    paused = False
    while True:
        shown = canvas
        if paused:
            shown = canvas.copy()
            cv2.rectangle(shown, (16, 16), (690, 64), (20, 20, 20), -1)
            cv2.putText(shown, "PAUSED - SPACE: resume | R: reinitialize", (28, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, shown)
        key = cv2.waitKey(30 if paused else delay_ms) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            return "quit"
        try:
            visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return "quit"
        if visible < 1:
            return "quit"
        if key in (ord('n'), ord('N')):
            return "next"
        if key in (ord('r'), ord('R')) and on_reset is not None:
            canvas = on_reset()
            continue
        if key in (ord('s'), ord('S')) and on_seat is not None:
            canvas = on_seat()
            continue
        if key == ord(' '):
            paused = not paused
            continue
        if not paused:
            return "continue"


def play_video(video_path, detector, model_path, config, args, video_index, video_count):
    print(f"[{video_index}/{video_count}] {video_path.name}")
    reset_sidebar_cache()
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
            print("[WARN] Invalid source FPS; using 30")
        if args.start_seconds:
            if not capture.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000):
                raise OSError("Video backend cannot seek to the requested start time")
        frame_index = max(0, round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        previous_ms, processed = -1, 0
        states, locks, posture_states = Counter(), [], Counter()
        max_people = 0
        compute_durations = deque(maxlen=30)
        compute_seconds = 0.0
        started = time.perf_counter()
        action = "continue"
        with TrackingPipeline(detector, model_path, fps, config) as pipeline:
            while args.max_frames is None or processed < args.max_frames:
                frame_start = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    if processed == 0:
                        raise OSError("No decodable frames at requested video position")
                    break
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
                if not math.isfinite(timestamp) or timestamp <= previous_ms:
                    timestamp = frame_index * 1000 / fps
                timestamp_ms = max(previous_ms + 1, round(timestamp))
                previous_ms = timestamp_ms
                analysis = pipeline.process(frame, timestamp_ms, frame_index)
                states[analysis.state] += 1
                if analysis.posture is not None:
                    posture_states[analysis.posture.state] += 1
                if analysis.state == 'reinitializing':
                    reset_sidebar_cache()
                    print(f'[AUTO RESET] {analysis.detail} at {timestamp_ms / 1000:.2f}s')
                max_people = max(max_people, sum(t.observed for t in analysis.tracks))
                if analysis.target_id is not None and analysis.target_id not in locks:
                    locks.append(analysis.target_id)
                    print(f"[LOCK] ID {analysis.target_id} at {timestamp_ms / 1000:.2f}s")
                # Display the previous 30 frames' complete decode/inference/render
                # rate, excluding playback waits and user pauses.
                processing_fps = (len(compute_durations) / sum(compute_durations)
                                  if compute_durations else
                                  1 / max(1e-6, time.perf_counter() - frame_start))
                canvas = render_fhd(frame, analysis, video_path, video_index, video_count,
                                    frame_index, fps, processing_fps, config)
                frame_compute = max(1e-6, time.perf_counter() - frame_start)
                compute_durations.append(frame_compute)
                compute_seconds += frame_compute

                def reset_current_frame():
                    pipeline.reset_selection()
                    reset_sidebar_cache()
                    print(f"[RESET] Selection cleared at {timestamp_ms / 1000:.2f}s")
                    fresh = FrameAnalysis(analysis.tracks, None, None, "collecting",
                                          "collecting_valid_shoulders", {}, timestamp_ms,
                                          selection_generation=analysis.selection_generation + 1,
                                          auto_reset_count=analysis.auto_reset_count,
                                          auto_reset_reason=analysis.auto_reset_reason,
                                          pose_recovery_count=analysis.pose_recovery_count)
                    return render_fhd(frame, fresh, video_path, video_index, video_count,
                                      frame_index, fps, processing_fps, config)

                if not args.headless:
                    delay = max(1, math.ceil((1 / fps - (time.perf_counter() - frame_start)) * 1000))
                    action = show_frame(canvas, delay, reset_current_frame)
                processed += 1
                frame_index += 1
                if action in ("quit", "next"):
                    break
            summary = {
                "frames": processed, "maximum_observed_people": max_people,
                "locked_ids": locks, "state_counts": dict(states),
                "final_state": analysis.state, "final_detail": analysis.detail,
                "candidate_statistics": analysis.statistics,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "compute_seconds": round(compute_seconds, 3),
                "processing_fps": round(processed / max(compute_seconds, 1e-6), 2),
                "display_shape": [1080, 1920, 3],
                "auto_reset_count": analysis.auto_reset_count,
                "last_auto_reset_reason": analysis.auto_reset_reason,
                "pose_recovery_count": analysis.pose_recovery_count,
                "posture_state_counts": dict(posture_states),
                "final_posture": asdict(analysis.posture) if analysis.posture else None,
            }
            print("[SUMMARY] " + json.dumps(summary, ensure_ascii=True))
            return action != "quit", summary
    finally:
        capture.release()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=BASE_DIR, help='Video file or nonrecursive folder')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='MediaPipe Pose Landmarker .task')
    parser.add_argument('--detector-model', type=Path, help='Compatible raw COCO YOLOX ONNX model')
    parser.add_argument('--detector-preset', choices=tuple(DETECTOR_PRESETS),
                        default=DEFAULT_DETECTOR_PRESET,
                        help='YOLOX size preset; smaller models favor speed')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--detector-size', type=int,
                        help='Model input size (automatic for bundled presets)')
    parser.add_argument('--det-confidence', type=float, default=.35)
    parser.add_argument('--init-seconds', type=float, default=2.0)
    parser.add_argument('--min-samples', type=int, default=10)
    parser.add_argument('--min-gap', type=float, default=.10, help='(second-smallest - smallest) / second-smallest')
    parser.add_argument('--landmark-confidence', type=float, default=.70)
    parser.add_argument('--sample-stride', type=int, default=3)
    parser.add_argument('--crop-padding', type=float, default=.08)
    parser.add_argument('--max-age-seconds', type=float, default=1.0)
    parser.add_argument('--auto-reset-seconds', type=float, default=.5,
                        help='Continuous missing target detection before reselecting (0=immediate)')
    parser.add_argument('--pose-recovery-seconds', type=float, default=2.0,
                        help='Retry target pose after persistent failure; keep ID and baseline (0=immediate)')
    parser.add_argument('--posture-baseline-seconds', type=float, default=2.0,
                        help='Reliable upright source time for head-height/shoulder baseline')
    parser.add_argument('--posture-head-drop-ratio', type=float, default=.30,
                        help='Minimum head descent divided by calibrated shoulder width')
    parser.add_argument('--posture-forearm-distance-ratio', type=float, default=.35,
                        help='Maximum head-to-forearm distance divided by calibrated shoulder width')
    parser.add_argument('--posture-hold-seconds', type=float, default=1.5,
                        help='Continuous valid source time before suspected table-rest is confirmed')
    parser.add_argument('--posture-max-gap-seconds', type=float, default=.25,
                        help='Break posture evidence when adjacent source timestamps exceed this gap')
    parser.add_argument('--headless', action='store_true', help='Console-only bounded verification; no files saved')
    parser.add_argument('--max-frames', type=int, help='Limit processed frames per video')
    parser.add_argument('--start-seconds', type=float, default=0.0, help='Optional test seek position')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ('init_seconds', 'max_age_seconds', 'posture_baseline_seconds',
                 'posture_head_drop_ratio', 'posture_forearm_distance_ratio',
                 'posture_hold_seconds', 'posture_max_gap_seconds'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and positive')
    for name in ('det_confidence', 'landmark_confidence', 'min_gap'):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value < 1:
            parser.error(f'--{name.replace("_", "-")} must be in [0,1)')
    if args.min_samples < 2 or args.sample_stride < 1:
        parser.error('--min-samples >= 2 and --sample-stride >= 1 required')
    if not math.isfinite(args.crop_padding) or args.crop_padding < 0:
        parser.error('--crop-padding must be finite and nonnegative')
    if not math.isfinite(args.auto_reset_seconds) or args.auto_reset_seconds < 0:
        parser.error('--auto-reset-seconds must be finite and nonnegative')
    if not math.isfinite(args.pose_recovery_seconds) or args.pose_recovery_seconds < 0:
        parser.error('--pose-recovery-seconds must be finite and nonnegative')
    if not math.isfinite(args.start_seconds) or args.start_seconds < 0:
        parser.error('--start-seconds must be finite and nonnegative')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    if args.headless and args.max_frames is None:
        parser.error('--headless requires --max-frames to bound the verification run')
    try:
        videos = find_videos(args.input.expanduser().resolve())
        if not videos:
            print(f'No videos found in {args.input.resolve()}. Place videos there and run again.')
            return 0
        model_path = ensure_model(args.model)
        detector = PersonDetector(args.detector_model, args.device, args.det_confidence,
                                  input_size=args.detector_size, preset=args.detector_preset)
        config = PipelineConfig(
            init_seconds=args.init_seconds, min_samples=args.min_samples,
            min_gap=args.min_gap, landmark_confidence=args.landmark_confidence,
            sample_stride=args.sample_stride, crop_padding=args.crop_padding,
            max_age_seconds=args.max_age_seconds, detection_confidence=args.det_confidence,
            detector_label=f'{detector.provenance["name"]} / {detector.input_size}',
            auto_reset_seconds=args.auto_reset_seconds,
            pose_recovery_seconds=args.pose_recovery_seconds,
            posture_baseline_seconds=args.posture_baseline_seconds,
            posture_head_drop_ratio=args.posture_head_drop_ratio,
            posture_forearm_distance_ratio=args.posture_forearm_distance_ratio,
            posture_hold_seconds=args.posture_hold_seconds,
            posture_max_gap_seconds=args.posture_max_gap_seconds,
        )
        print(f'Detector: {detector.provenance["name"]} / OpenCV DNN / {detector.device}')
        print('Preview only. Missing target: auto reselect. Pose failure: keep ID and retry. Table-rest: lowered head near forearm over time. SPACE pause; R manual reset; N next; Q/ESC quit.')
        if not args.headless:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(WINDOW_NAME, FHD_WIDTH, FHD_HEIGHT)
        failures = 0
        for index, path in enumerate(videos, 1):
            try:
                keep_playing, _ = play_video(path, detector, model_path, config, args, index, len(videos))
                if not keep_playing:
                    break
            except (OSError, RuntimeError, ValueError, cv2.error) as exc:
                failures += 1
                print(f'[ERROR] Skipping {path.name}: {exc}', file=sys.stderr)
        return 1 if failures == len(videos) else 0
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    raise SystemExit(main())
