"""
Run a timed emotion recording session from the webcam and export to CSV.

Usage:
    python scripts/run_emotion_session.py                     # 30s default
    python scripts/run_emotion_session.py --duration 60       # 60 seconds
    python scripts/run_emotion_session.py --output my_run.csv # custom filename
"""

import argparse
import time
import sys
import logging

from config import RECORDER_TARGET_FPS
from utils.logger import get_logger
from pipeline.emotion_recorder import EmotionRecorder

logger = get_logger(__name__)

DEFAULT_DURATION_S = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record webcam emotions to a timestamped CSV."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_S,
        help=f"Recording duration in seconds (default: {DEFAULT_DURATION_S}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV filename (saved to outputs/ directory).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=RECORDER_TARGET_FPS,
        help=f"Target frames per second (default: {RECORDER_TARGET_FPS}).",
    )
    return parser.parse_args()


def run_session(duration_s: int, target_fps: int, output_filename: str | None) -> None:
    """Run a timed emotion recording session."""
    frame_interval = 1.0 / target_fps
    recorder = EmotionRecorder()

    logger.info(
        "Starting %ds emotion session at %d FPS target...", duration_s, target_fps
    )
    print(f"\n  Recording for {duration_s}s — look at the camera. Press Ctrl+C to stop early.\n")

    recorder.start()
    frames_processed = 0

    try:
        while recorder.timer.elapsed_ms() < duration_s * 1000:
            loop_start = time.monotonic()

            record = recorder.record_frame()
            if record is not None:
                frames_processed += 1
                # Live feedback every 1 second worth of frames
                if frames_processed % target_fps == 0:
                    elapsed_s = recorder.timer.elapsed_ms() / 1000
                    print(
                        f"  [{elapsed_s:6.1f}s] emotion={record.emotion:<12s} "
                        f"confidence={record.confidence:.2f}  "
                        f"face={'yes' if record.face_detected else 'NO'}",
                    )

            # Sleep to maintain target FPS
            processing_time = time.monotonic() - loop_start
            sleep_time = frame_interval - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n  Session interrupted by user.")
        logger.info("Session interrupted by user at %dms.", recorder.timer.elapsed_ms())

    recorder.stop()

    # Export results
    csv_path = recorder.export_csv(output_filename)
    actual_fps = frames_processed / (recorder.timer.elapsed_ms() / 1000) if recorder.timer.elapsed_ms() > 0 else 0

    print(f"\n  Session complete.")
    print(f"  Duration:  {recorder.timer.elapsed_ms() / 1000:.1f}s")
    print(f"  Frames:    {frames_processed}")
    print(f"  Avg FPS:   {actual_fps:.1f}")
    print(f"  Output:    {csv_path}\n")


def main() -> None:
    args = parse_args()
    run_session(
        duration_s=args.duration,
        target_fps=args.fps,
        output_filename=args.output,
    )


if __name__ == "__main__":
    main()
