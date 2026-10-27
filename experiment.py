"""Independent FHD child-subject behavior experiment with source-time CSVs."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

import pose_viewer as preview
from behavior_csv import TimeCsvLogger
from behavior_engine import BehaviorEngine, BehaviorFrame
from motion_detection import MotionConfig
from seat_detection import SeatConfig, SeatResult
from tracking_pipeline import BASE_DIR, FrameAnalysis, PipelineConfig, TrackingPipeline, ensure_model
from person_detector import PersonDetector


BEHAVIOR_NAMES = {
    'head_shake': '搖頭', 'left_wave': '左手快速揮舞', 'right_wave': '右手快速揮舞',
    'both_wave': '雙手快速揮舞', 'table_rest': '疑似趴桌', 'seat_leave': '離開座位區',
}
STATUS_NAMES = {'active': '成立', 'normal': '未成立', 'unknown': '遮擋／無法判定',
                'calibrating': '建立基準', 'candidate': '持續計時中'}
_PANEL_CACHE = {}


def build_parser():
    parser = preview.build_parser()
    parser.description = __doc__
    next(action for action in parser._actions if action.dest == 'headless').help = (
        'No preview window; CSV logging remains enabled (use --max-frames for bounded tests)'
    )
    parser.add_argument('--settings', type=Path, default=BASE_DIR / 'settings.json')
    parser.add_argument('--output', type=Path, default=BASE_DIR / 'logs')
    parser.add_argument('--csv-interval', type=float, default=1.0)
    parser.add_argument('--reselect-on-loss', action=argparse.BooleanOptionalAction, default=True,
                        help='Allow fresh shoulder selection after loss; may select a different person')
    parser.add_argument('--motion-window-seconds', type=float, default=2.0)
    parser.add_argument('--shake-min-turns', type=int, default=4)
    parser.add_argument('--shake-threshold', type=float, default=.12)
    parser.add_argument('--shake-min-interval', type=float, default=.12)
    parser.add_argument('--head-calibration-seconds', type=float, default=.4)
    parser.add_argument('--motion-max-gap-seconds', type=float, default=.2)
    parser.add_argument('--wave-min-hz', type=float, default=2.0)
    parser.add_argument('--wave-min-amplitude', type=float, default=.03)
    parser.add_argument('--wave-max-amplitude', type=float, default=.25)
    parser.add_argument('--wave-min-cycles', type=int, default=3)
    parser.add_argument('--wave-min-seconds', type=float, default=1.5)
    parser.add_argument('--seat-baseline-seconds', type=float, default=2.0)
    parser.add_argument('--seat-iou-threshold', type=float, default=.5,
                        help='Observed track/reference box IoU below this value means departure')
    parser.add_argument('--seat-departure-ratio', type=float, default=1.0)
    parser.add_argument('--seat-hold-seconds', type=float, default=1.0)
    parser.add_argument('--seat-return-seconds', type=float, default=.7)
    for name in ('seat_departure_ratio', 'seat_hold_seconds', 'seat_return_seconds'):
        next(action for action in parser._actions if action.dest == name).help = (
            'Legacy setting accepted for compatibility; not used by the IoU seat rule'
        )
    parser.add_argument('--seat-max-gap-seconds', type=float, default=.25)
    parser.add_argument('--seat-roi', nargs=4, type=float, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'),
                        help='Initial-generation reference person box in source pixels; full reselection cancels it; S selects it interactively')
    return parser


def parse_configuration(argv=None):
    parser = build_parser()
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--settings', type=Path, default=BASE_DIR / 'settings.json')
    settings_path = early.parse_known_args(argv)[0].settings.expanduser().resolve()
    if settings_path.is_file():
        try:
            defaults = json.loads(settings_path.read_text(encoding='utf-8-sig'))
            if not isinstance(defaults, dict):
                raise ValueError('settings must be a JSON object')
            actions = {action.dest: action for action in parser._actions if action.dest != 'help'}
            for name, value in defaults.items():
                if name not in actions or name == 'settings':
                    raise ValueError(f'Unknown settings key: {name}')
                action = actions[name]
                if action.nargs == 0 and not isinstance(value, bool):
                    raise ValueError(f'{name} must be a JSON boolean')
                if value is not None and action.type is not None:
                    if isinstance(action.nargs, int):
                        if not isinstance(value, list) or len(value) != action.nargs:
                            raise ValueError(f'{name} needs {action.nargs} numbers')
                        value = [action.type(item) for item in value]
                    else:
                        value = action.type(value)
                if action.choices and value not in action.choices:
                    raise ValueError(f'Invalid {name}: {value}')
                defaults[name] = value
            parser.set_defaults(**defaults)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(f'Cannot read {settings_path}: {exc}')
    else:
        parser.error(f'Settings file missing: {settings_path}')
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.csv_interval) or args.csv_interval < .001:
            raise ValueError('--csv-interval must be finite and at least 0.001 seconds')
        if not math.isfinite(args.start_seconds) or args.start_seconds < 0:
            raise ValueError('--start-seconds must be finite and nonnegative')
        if args.max_frames is not None and args.max_frames < 1:
            raise ValueError('--max-frames must be positive')
        if args.min_samples < 2 or args.sample_stride < 1:
            raise ValueError('--min-samples >= 2 and --sample-stride >= 1 required')
        for name in ('det_confidence', 'landmark_confidence', 'min_gap'):
            value = getattr(args, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f'--{name.replace("_", "-")} must be in [0,1)')
        for name in ('init_seconds', 'max_age_seconds'):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(args.crop_padding) or args.crop_padding < 0:
            raise ValueError('crop_padding must be finite and nonnegative')
        pipeline = PipelineConfig(
            init_seconds=args.init_seconds, min_samples=args.min_samples, min_gap=args.min_gap,
            landmark_confidence=args.landmark_confidence, sample_stride=args.sample_stride,
            crop_padding=args.crop_padding, max_age_seconds=args.max_age_seconds,
            detection_confidence=args.det_confidence, auto_reset_seconds=args.auto_reset_seconds,
            pose_recovery_seconds=args.pose_recovery_seconds,
            pose_reselect_seconds=args.pose_reselect_seconds,
            posture_baseline_seconds=args.posture_baseline_seconds,
            posture_head_drop_ratio=args.posture_head_drop_ratio,
            posture_forearm_distance_ratio=args.posture_forearm_distance_ratio,
            posture_hold_seconds=args.posture_hold_seconds,
            posture_max_gap_seconds=args.posture_max_gap_seconds,
            retain_target_on_loss=not args.reselect_on_loss,
        )
        motion = MotionConfig(
            window_seconds=args.motion_window_seconds, shake_min_turns=args.shake_min_turns,
            shake_threshold=args.shake_threshold, shake_min_interval=args.shake_min_interval,
            wave_min_hz=args.wave_min_hz, wave_min_amplitude=args.wave_min_amplitude,
            wave_max_amplitude=args.wave_max_amplitude, wave_min_cycles=args.wave_min_cycles,
            wave_min_seconds=args.wave_min_seconds, confidence=args.landmark_confidence,
            max_gap_seconds=args.motion_max_gap_seconds,
            head_calibration_seconds=args.head_calibration_seconds,
        )
        seat = SeatConfig(baseline_seconds=args.seat_baseline_seconds,
                          iou_threshold=args.seat_iou_threshold,
                          departure_ratio=args.seat_departure_ratio, hold_seconds=args.seat_hold_seconds,
                          return_seconds=args.seat_return_seconds, max_gap_seconds=args.seat_max_gap_seconds,
                          confidence=args.landmark_confidence)
        BehaviorEngine(motion, seat, args.seat_roi)  # Validate any supplied rectangle.
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    return args, pipeline, motion, seat


def _draw_panel(canvas, analysis, behavior, path, index, count, frame_index, fps, speed,
                pipeline_config, motion_config, csv_interval):
    status_key = tuple(behavior.statuses.items()) if behavior else ()
    key = (str(path), index, analysis.timestamp_ms // 200, analysis.target_id,
           analysis.selection_generation, analysis.state, status_key,
           analysis.pose_recovery_count, analysis.auto_reset_count)
    if _PANEL_CACHE.get('key') != key:
        panel = Image.new('RGB', (420, 1080), (22, 25, 31))
        draw = ImageDraw.Draw(panel)
        y = 16

        def line(text, size=17, color=(220, 226, 233), gap=5):
            nonlocal y
            for row in preview._wrapped_lines(str(text), size, 384):
                draw.text((16, y), row, fill=color, font=preview._font(size))
                y += size + 6
            y += gap

        line('兒童受試者行為實驗', 23, (100, 225, 255))
        line(f'影片 {index}/{count}：{path.name}', 16)
        line(f'時間 {analysis.timestamp_ms / 1000:.2f} 秒／影格 {frame_index + 1}', 16)
        line(f'來源 {fps:.1f} FPS／處理 {speed:.1f} FPS', 16)
        line(pipeline_config.detector_label, 15)
        line(f'目標 ID：{analysis.target_id if analysis.target_id is not None else "等待選人"}', 21)
        line(preview.STATE_LABELS.get(analysis.state, analysis.state), 18,
             (110, 255, 140) if analysis.state == 'locked' else (255, 208, 115))
        if analysis.state == 'lost' and pipeline_config.retain_target_on_loss:
            line('保留原 ID；無法確認新 ID 時按 R 重選', 15, (255, 208, 115))
        if analysis.state == 'reinitializing':
            line(preview._reason(analysis.detail), 15, (255, 208, 115))
        line('行為可同時成立；資料不足不視為未發生', 15)
        for name, label in BEHAVIOR_NAMES.items():
            state = behavior.statuses[name] if behavior else 'unknown'
            color = ((255, 155, 100) if state == 'active' else
                     (150, 190, 210) if state in ('unknown', 'calibrating') else (220, 226, 233))
            line(f'{label}：{STATUS_NAMES[state]}', 18, color)
        if behavior:
            line(f'搖頭：{motion_config.window_seconds:g} 秒內 {behavior.motion.turn_count}／{motion_config.shake_min_turns} 次', 15)
            for side, label in [('left', '左手'), ('right', '右手')]:
                hz = behavior.metrics.get(f'{side}_frequency_hz')
                amplitude = behavior.metrics.get(f'{side}_amplitude')
                hz_text = '--' if hz is None else f'{hz:.1f}'
                amp_text = '--' if amplitude is None else f'{amplitude:.3f}'
                line(f'{label}：{hz_text} Hz／擺幅 {amp_text} 倍肩寬', 14)
            if analysis.posture:
                line(f'趴桌持續：{analysis.posture.held_seconds:.1f}／{pipeline_config.posture_hold_seconds:g} 秒', 15)
            iou = behavior.seat.iou
            iou_text = '--' if iou is None else f'{iou:.0%}'
            threshold = behavior.metrics.get('seat_iou_threshold', .5)
            line(f'座位框 IoU：{iou_text}／門檻 {threshold:.0%}', 15)
            if behavior.seat.detail in ('awaiting_return_pose', 'awaiting_box_center_return',
                                        'baseline_head_shoulders_unavailable'):
                line(preview._reason(behavior.seat.detail), 14, (255, 208, 115), gap=1)
        line(f'CSV：每 {csv_interval:g} 秒取樣＋事件起訖', 15, (130, 220, 255))
        line(f'姿態不足 {analysis.pose_failure_seconds:.1f}／{pipeline_config.pose_reselect_seconds:g} 秒後重選', 14, gap=1)
        line(f'每 {pipeline_config.pose_recovery_seconds:g} 秒重建姿態／恢復 {analysis.pose_recovery_count} 次', 14, gap=1)
        line(f'自動重選 {analysis.auto_reset_count} 次', 14, gap=1)
        draw.rectangle((0, 900, 420, 1080), fill=(22, 25, 31))
        y = 916
        line('空白鍵：暫停／繼續', 17)
        line('S：設定座位區　R：重新選人', 17)
        line('N：下一部　Q／Esc：離開', 17)
        line('FHD｜不儲存處理影片｜無音訊', 15, (150, 160, 170))
        _PANEL_CACHE.update(key=key, panel=np.asarray(panel)[:, :, ::-1].copy())
    canvas[:, 1500:] = _PANEL_CACHE['panel']


def render_preview(frame, analysis, behavior, path, index, count, frame_index, fps, speed,
                   pipeline_config, motion_config, csv_interval):
    canvas = preview.render_fhd(frame, analysis, config=pipeline_config, sidebar=False)
    if behavior and behavior.seat.seat_roi:
        box = np.rint(preview.to_display(np.asarray(behavior.seat.seat_roi).reshape(2, 2), frame.shape)).astype(int)
        width, height, left, top = preview.display_geometry(frame.shape)
        box[:, 0] = np.clip(box[:, 0], left, left + width - 1)
        box[:, 1] = np.clip(box[:, 1], top, top + height - 1)
        cv2.rectangle(canvas, tuple(box[0]), tuple(box[1]), (220, 110, 210), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'SEAT REGION', (int(box[0, 0])+5, int(box[0, 1])+25),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (220, 110, 210), 2, cv2.LINE_AA)
    _draw_panel(canvas, analysis, behavior, path, index, count, frame_index, fps, speed,
                pipeline_config, motion_config, csv_interval)
    return canvas


def display_roi_to_source(rect, frame_shape):
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return None
    width, height, left, top = preview.display_geometry(frame_shape)
    if x < left or y < top or x + w > left + width or y + h > top + height:
        return None
    source_h, source_w = frame_shape[:2]
    return ((x-left)*source_w/width, (y-top)*source_h/height,
            (x+w-left)*source_w/width, (y+h-top)*source_h/height)


def play_video(path, detector, model_path, pipeline_config, motion_config, seat_config,
               args, index=1, count=1):
    capture = cv2.VideoCapture(str(path))
    logger = None
    last_timestamp = None
    last_analyzed_ms = None
    end_reason = 'error'
    _PANEL_CACHE.clear()
    try:
        if not capture.isOpened():
            raise OSError(f'Cannot open video: {path}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        if args.start_seconds and not capture.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds*1000):
            raise OSError('Video backend cannot seek to requested position')
        frame_position = capture.get(cv2.CAP_PROP_POS_FRAMES)
        frame_index = max(0, round(frame_position)) if math.isfinite(frame_position) else 0
        if args.start_seconds and frame_index == 0:
            frame_index = round(args.start_seconds * fps)
        engine = BehaviorEngine(motion_config, seat_config, args.seat_roi)
        logger = TimeCsvLogger(args.output, path, args.csv_interval,
                               metadata={'pipeline': asdict(pipeline_config), 'motion': asdict(motion_config),
                                         'seat': asdict(seat_config), 'manual_seat_roi': args.seat_roi,
                                         'source_fps': fps, 'detector': detector.provenance,
                                         'environment_record': str(BASE_DIR / 'ENVIRONMENT.json'),
                                         'head_turn_definition': 'Alternating extreme arrivals; first extreme counts as one',
                                         'seat_definition': 'Calibrate from continuous observed OC-SORT boxes with valid face/shoulders. After calibration, box/reference IoU below threshold means departure independently of pose. Return needs overlap, center inside and valid pose; full reselection cancels the reference'})
        print('[CSV] ' + str(logger.timeline_path))
        durations, counts = deque(maxlen=30), Counter()
        processed, compute_seconds = 0, 0.0
        action = 'continue'
        with TrackingPipeline(detector, model_path, fps, pipeline_config) as pipeline:
            while args.max_frames is None or processed < args.max_frames:
                started = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    if not processed:
                        raise OSError('No decodable video frames')
                    end_reason = 'end_video'
                    break
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
                if (not math.isfinite(timestamp)
                        or (last_timestamp is None and frame_index > 0 and timestamp <= 0)
                        or (last_timestamp is not None and timestamp <= last_timestamp)):
                    timestamp = frame_index * 1000 / fps
                timestamp = max(0 if last_timestamp is None else last_timestamp+1, round(timestamp))
                last_timestamp = timestamp
                analysis = pipeline.process(frame, timestamp, frame_index)
                behavior = engine.update(analysis, frame.shape)
                logger.update(timestamp, analysis.target_id, analysis.selection_generation,
                              behavior.statuses, metrics=behavior.metrics)
                last_analyzed_ms = timestamp
                counts.update(name for name, state in behavior.statuses.items() if state == 'active')
                speed = len(durations)/sum(durations) if durations else 0.0
                canvas = render_preview(frame, analysis, behavior, path, index, count, frame_index,
                                        fps, speed, pipeline_config, motion_config, args.csv_interval)
                elapsed = max(1e-6, time.perf_counter()-started)
                durations.append(elapsed)
                compute_seconds += elapsed

                def reset_current():
                    nonlocal analysis, behavior, canvas
                    logger.boundary(timestamp, 'manual_reset')
                    pipeline.reset_selection()
                    engine.reset()
                    _PANEL_CACHE.clear()
                    analysis = FrameAnalysis(analysis.tracks, None, None, 'collecting',
                                          'collecting_valid_shoulders', {}, timestamp,
                                          selection_generation=pipeline.selection_generation,
                                          auto_reset_count=pipeline.auto_reset_count,
                                          pose_recovery_count=pipeline.pose_recovery_count)
                    behavior = None
                    canvas = render_preview(frame, analysis, behavior, path, index, count, frame_index, fps,
                                            speed, pipeline_config, motion_config, args.csv_interval)
                    return canvas

                def choose_seat():
                    nonlocal behavior, canvas
                    name = 'Select seat anchor region - ENTER confirm, ESC cancel'
                    try:
                        rect = cv2.selectROI(name, canvas, showCrosshair=True, fromCenter=False)
                    finally:
                        try:
                            cv2.destroyWindow(name)
                        except cv2.error:
                            pass
                    roi = display_roi_to_source(rect, frame.shape)
                    if roi is not None:
                        logger.boundary(timestamp, 'seat_roi_changed')
                        engine.set_seat_roi(roi)
                        _PANEL_CACHE.clear()
                        if behavior is not None:
                            seat_preview = SeatResult('calibrating', 'seat_roi_changed', seat_roi=roi)
                            statuses = dict(behavior.statuses, seat_leave='calibrating')
                            details = dict(behavior.details, seat_leave='seat_roi_changed')
                            metrics = dict(behavior.metrics, seat=asdict(seat_preview))
                            behavior = replace(behavior, statuses=statuses, details=details,
                                               metrics=metrics, seat=seat_preview)
                        canvas = render_preview(frame, analysis, behavior, path, index, count, frame_index, fps,
                                                speed, pipeline_config, motion_config, args.csv_interval)
                        print('[SEAT ROI] ' + json.dumps(roi))
                    return canvas

                if not args.headless:
                    delay = max(1, math.ceil((1/fps - (time.perf_counter()-started))*1000))
                    action = preview.show_frame(canvas, delay, reset_current, choose_seat)
                processed += 1
                frame_index += 1
                if action in ('quit', 'next'):
                    end_reason = action
                    break
            else:
                end_reason = 'max_frames'
        logger.close(last_analyzed_ms, reason=end_reason)
        summary = {'video': str(path), 'frames': processed, 'source_fps': fps,
                   'processing_fps': round(processed/max(compute_seconds, 1e-6), 2),
                   'confirmed_active_frames': dict(counts), 'final_target_id': analysis.target_id,
                   'final_tracking_state': analysis.state,
                   'final_states': behavior.statuses if behavior else {name: 'unknown' for name in BEHAVIOR_NAMES},
                   'final_metrics': behavior.metrics if behavior else {},
                   'timeline_rows': logger.timeline_row_count, 'events': logger.event_count,
                   'csv_paths': {name: str(value) for name, value in logger.paths.items()},
                   'display_shape': list(canvas.shape), 'end_reason': end_reason}
        print('[SUMMARY] ' + json.dumps(summary, ensure_ascii=True, default=str))
        return action != 'quit', summary
    except KeyboardInterrupt:
        end_reason = 'interrupted'
        raise
    finally:
        try:
            if logger is not None:
                logger.close(last_analyzed_ms, reason=end_reason)
        finally:
            capture.release()


def main(argv=None):
    args, pipeline_config, motion_config, seat_config = parse_configuration(argv)
    try:
        videos = preview.find_videos(args.input.expanduser().resolve())
        if not videos:
            print(f'No videos in {args.input}. Place source videos in {BASE_DIR}, then run again.')
            return 0
        args.output = args.output.expanduser().resolve()
        model_path = ensure_model(args.model)
        detector = PersonDetector(args.detector_model, args.device, args.det_confidence,
                                  input_size=args.detector_size, preset=args.detector_preset)
        pipeline_config = replace(pipeline_config, detector_label=f'{detector.provenance["name"]} / {detector.input_size}')
        if not args.headless:
            cv2.namedWindow(preview.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(preview.WINDOW_NAME, 1920, 1080)
        failures = 0
        for index, path in enumerate(videos, 1):
            try:
                keep, _ = play_video(path, detector, model_path, pipeline_config, motion_config,
                                     seat_config, args, index, len(videos))
                if not keep:
                    break
            except (OSError, ValueError, RuntimeError, cv2.error) as exc:
                failures += 1
                print(f'[ERROR] {path.name}: {exc}', file=sys.stderr)
        return 1 if failures == len(videos) else 0
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    raise SystemExit(main())
