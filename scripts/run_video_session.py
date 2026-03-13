"""
Run a synchronized video + emotion recording session.

Plays a video file while simultaneously capturing the viewer's facial
emotions via webcam. Both streams share the same monotonic clock so
timestamps are aligned. Exports a CSV on completion.

Controls:
    SPACE  — Pause / resume
    Q      — Quit early

Usage:
    python scripts/run_video_session.py --video path/to/clip.mp4
    python scripts/run_video_session.py --video clip.mp4 --output my_session.csv
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on the path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import SYNC_DISPLAY_FPS, SYNC_EMOTION_FPS
from utils.logger import get_logger
from pipeline.sync_controller import SyncController

logger = get_logger(__name__)

WINDOW_NAME_VIDEO = "AffectSync — Video"
WINDOW_NAME_WEBCAM = "AffectSync — Webcam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AffectSync — Synchronized video + emotion recording session."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the video file to play.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV filename (saved to outputs/ directory).",
    )
    parser.add_argument(
        "--display-fps",
        type=int,
        default=SYNC_DISPLAY_FPS,
        help=f"Display loop FPS (default: {SYNC_DISPLAY_FPS}).",
    )
    parser.add_argument(
        "--emotion-fps",
        type=int,
        default=SYNC_EMOTION_FPS,
        help=f"Emotion inference FPS (default: {SYNC_EMOTION_FPS}).",
    )
    parser.add_argument(
        "--no-webcam-preview",
        action="store_true",
        help="Hide the webcam preview window.",
    )
    return parser.parse_args()


def draw_emotion_overlay(frame: np.ndarray, emotion: str, confidence: float) -> None:
    """Draw emotion label on the bottom-left of the video frame."""
    label = f"{emotion} ({confidence:.0%})"
    # Black background rectangle for readability
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    y_pos = frame.shape[0] - 10
    cv2.rectangle(
        frame,
        (5, y_pos - text_h - baseline - 5),
        (15 + text_w, y_pos + 5),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        frame, label, (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )


def draw_timestamp_overlay(frame: np.ndarray, timestamp_ms: int) -> None:
    """Draw current video timestamp on the top-left of the frame."""
    seconds = timestamp_ms / 1000.0
    label = f"t={seconds:.1f}s"
    cv2.putText(
        frame, label, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )


def draw_pause_overlay(frame: np.ndarray) -> None:
    """Draw a PAUSED indicator in the center of the frame."""
    h, w = frame.shape[:2]
    cv2.putText(
        frame, "PAUSED", (w // 2 - 60, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
    )


def run_session(
    video_path: str,
    output_filename: str | None,
    display_fps: int,
    emotion_fps: int,
    show_webcam: bool,
) -> None:
    """Run the synchronized video + emotion recording session."""
    controller = SyncController(
        video_path=video_path,
        display_fps=display_fps,
        emotion_fps=emotion_fps,
    )

    # Setup — open video and webcam
    logger.info("Setting up sync session...")
    controller.setup()

    # Set the video window to be resizable and give it a max size (e.g., 1280x720)
    cv2.namedWindow(WINDOW_NAME_VIDEO, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME_VIDEO, 1280, 720) 

    if show_webcam:
        # Set the webcam window to be resizable too (e.g., 640x480)
        cv2.namedWindow(WINDOW_NAME_WEBCAM, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME_WEBCAM, 640, 480)

    video_duration_s = controller.video_player.duration_ms / 1000
    print(f"\n  Video: {Path(video_path).name} ({video_duration_s:.1f}s)")
    print(f"  Display FPS: {display_fps} | Emotion FPS: {emotion_fps}")
    print(f"  Controls: SPACE=pause/resume, Q=quit\n")

    frame_interval = controller.frame_interval_s
    controller.start()

    frames_displayed = 0
    inference_count = 0
    is_paused = False

    # Keep a copy of the last video frame for display during pause
    last_video_frame = None

    try:
        for synced in controller.run():
            loop_start = time.monotonic()

            # --- Handle keyboard input ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\n  Quit by user.")
                break

            if key == ord(" "):  # Spacebar
                if is_paused:
                    controller.resume()
                    is_paused = False
                    print("  ▶ Resumed")
                else:
                    controller.pause()
                    is_paused = True
                    print("  ⏸ Paused")
                continue

            # --- Paused state: show frozen frame with overlay ---
            if is_paused:
                if last_video_frame is not None:
                    display = last_video_frame.copy()
                    draw_pause_overlay(display)
                    draw_timestamp_overlay(display, synced.video_timestamp_ms)
                    cv2.imshow(WINDOW_NAME_VIDEO, display)
                # Small sleep to avoid busy-waiting during pause
                time.sleep(0.03)
                continue

            # --- Active playback ---
            if synced.video_frame is not None:
                display = synced.video_frame.copy()
                last_video_frame = synced.video_frame.copy()

                # Draw overlays
                draw_timestamp_overlay(display, synced.video_timestamp_ms)
                if synced.emotion_record is not None:
                    draw_emotion_overlay(
                        display,
                        synced.emotion_record.emotion,
                        synced.emotion_record.confidence,
                    )

                cv2.imshow(WINDOW_NAME_VIDEO, display)
                frames_displayed += 1

            # Track inference count for summary
            if synced.is_inference_frame and synced.emotion_record is not None:
                inference_count += 1

                # Live feedback every 1 second worth of inferences
                if inference_count % emotion_fps == 0:
                    elapsed_s = controller.timer.elapsed_ms() / 1000
                    rec = synced.emotion_record
                    print(
                        f"  [{elapsed_s:6.1f}s] emotion={rec.emotion:<12s} "
                        f"confidence={rec.confidence:.2f}  "
                        f"face={'yes' if rec.face_detected else 'NO'}",
                    )

            # --- Webcam preview (optional) ---
            if show_webcam:
                try:
                    _, webcam_frame = controller.recorder._webcam.read_frame()
                    cv2.imshow(WINDOW_NAME_WEBCAM, webcam_frame)
                except RuntimeError:
                    pass  # Webcam read failed, skip preview

            # Sleep to maintain display FPS
            processing_time = time.monotonic() - loop_start
            sleep_time = frame_interval - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n  Session interrupted (Ctrl+C).")

    # --- Cleanup and export ---
    controller.stop()
    cv2.destroyAllWindows()

    duration_total = controller.timer.elapsed_ms() / 1000
    actual_display_fps = frames_displayed / duration_total if duration_total > 0 else 0
    actual_emotion_fps = inference_count / duration_total if duration_total > 0 else 0

    # Export CSV
    try:
        csv_path = controller.export_session_csv(output_filename)
    except ValueError:
        csv_path = "(no records to export)"

    print(f"\n  Session complete.")
    print(f"  Duration:       {duration_total:.1f}s")
    print(f"  Frames shown:   {frames_displayed}")
    print(f"  Avg display:    {actual_display_fps:.1f} FPS")
    print(f"  Inferences:     {inference_count}")
    print(f"  Avg emotion:    {actual_emotion_fps:.1f} FPS")
    print(f"  Output:         {csv_path}\n")

    controller.teardown()


def main() -> None:
    args = parse_args()
    run_session(
        video_path=args.video,
        output_filename=args.output,
        display_fps=args.display_fps,
        emotion_fps=args.emotion_fps,
        show_webcam=not args.no_webcam_preview,
    )


if __name__ == "__main__":
    main()
