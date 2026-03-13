"""
Synchronized video + emotion recording controller.

The SyncController is the brain of Phase 2. It runs a single main loop
that on each tick:
  1. Asks the SessionTimer "what time is it?"
  2. Reads the next video frame sequentially (fast decode, no seeking).
  3. Captures a webcam frame and runs emotion inference (at a lower rate).
  4. Pairs the video timestamp with the emotion data.

Both the video player and emotion recorder share the SAME SessionTimer,
so pausing the timer freezes everything simultaneously — no drift.

Design decisions:
- Single-threaded loop. The main loop targets ~30 FPS for smooth video
  display, while emotion inference runs at ~10 FPS (every Nth iteration).
  This avoids threading complexity for the MVP.
- The controller does NOT own the display window. It yields SyncedFrame
  objects that the CLI script uses to render the UI. This keeps the
  controller testable without any GUI dependency.
- Pause/resume is delegated entirely to the SessionTimer. When paused,
  elapsed_ms() freezes, so the video player keeps showing the same frame
  and the emotion recorder stops recording.
- Video frames are read SEQUENTIALLY with read_next_frame() during
  normal playback. Seeking (get_frame_at / seek_to) is only used after
  a resume from pause to reposition to the correct timestamp. Sequential
  reads are ~100x faster than seeking on compressed codecs like H.264.
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.timer import SessionTimer
from pipeline.video_player import VideoPlayer
from pipeline.webcam_capture import WebcamCapture
from pipeline.face_detector import FaceDetector
from pipeline.emotion_classifier import EmotionClassifier
from pipeline.emotion_recorder import EmotionRecorder, EmotionRecord

logger = logging.getLogger(__name__)


@dataclass
class SyncedFrame:
    """
    One tick of the sync loop: a video frame paired with emotion data.

    The video_timestamp_ms and emotion_timestamp_ms should be very close
    (within one loop tick) since they come from the same timer. They are
    stored separately so downstream analysis can verify sync quality.
    """

    video_timestamp_ms: int
    video_frame: np.ndarray | None
    emotion_record: EmotionRecord | None
    is_inference_frame: bool


class SyncController:
    """
    Coordinates video playback and emotion recording on a shared timer.

    Components are injected so tests can replace hardware with mocks.

    Usage:
        controller = SyncController(video_path="clip.mp4")
        controller.setup()
        for synced in controller.run():
            # synced.video_frame  — display this
            # synced.emotion_record — log or overlay this
            pass
        controller.teardown()
    """

    def __init__(
        self,
        video_path: str | Path,
        webcam: WebcamCapture | None = None,
        face_detector: FaceDetector | None = None,
        emotion_classifier: EmotionClassifier | None = None,
        timer: SessionTimer | None = None,
        display_fps: int = 30,
        emotion_fps: int = 10,
    ):
        self._video_player = VideoPlayer(video_path)
        self._timer = timer or SessionTimer()

        # Build the EmotionRecorder with the SAME timer the controller owns
        self._recorder = EmotionRecorder(
            webcam=webcam or WebcamCapture(),
            face_detector=face_detector or FaceDetector(),
            emotion_classifier=emotion_classifier or EmotionClassifier(),
            timer=self._timer,
        )

        self._display_fps = display_fps
        self._emotion_fps = emotion_fps

        # How many display ticks between emotion inferences
        # e.g., 30 display / 10 emotion = every 3rd tick
        self._emotion_every_n = max(1, display_fps // emotion_fps)

        self._is_setup = False
        self._synced_records: list[SyncedFrame] = []

    @property
    def timer(self) -> SessionTimer:
        return self._timer

    @property
    def video_player(self) -> VideoPlayer:
        return self._video_player

    @property
    def recorder(self) -> EmotionRecorder:
        return self._recorder

    @property
    def synced_records(self) -> list[SyncedFrame]:
        return list(self._synced_records)

    @property
    def display_fps(self) -> int:
        return self._display_fps

    @property
    def frame_interval_s(self) -> float:
        """Seconds between display loop ticks."""
        return 1.0 / self._display_fps

    def setup(self) -> None:
        """
        Open video and webcam, prepare for recording.

        Call this before run(). Separated from __init__ so the caller
        can construct the controller, inspect metadata, then decide
        whether to proceed.
        """
        self._video_player.open()
        self._synced_records = []

        # Open webcam (recorder.start() handles this, but we need the
        # webcam ready before the timer starts)
        if not self._recorder._webcam.is_open:
            self._recorder._webcam.open()

        self._is_setup = True

        logger.info(
            "SyncController ready — video: %s (%.1fs) | display: %d FPS | emotion: %d FPS",
            self._video_player.video_path.name,
            self._video_player.duration_ms / 1000,
            self._display_fps,
            self._emotion_fps,
        )

    def teardown(self) -> None:
        """Release all resources."""
        if self._timer.is_running or self._timer.is_paused:
            self._timer.stop()
        self._video_player.release()
        self._recorder._webcam.release()
        self._is_setup = False
        logger.info(
            "SyncController torn down — %d synced frames collected.",
            len(self._synced_records),
        )

    def start(self) -> None:
        """
        Start the shared timer and the emotion recorder.

        The recorder's start() normally starts its own timer, but since
        we injected our shared timer, calling start() on the recorder
        will start our shared timer. We call it directly on the timer
        to be explicit, then mark the recorder as ready.
        """
        if not self._is_setup:
            raise RuntimeError("Call setup() before start().")

        self._synced_records = []
        self._timer.start()
        # The recorder checks is_running on record_frame(), and since
        # we share the timer, it's now running. We just need to clear
        # its internal records list.
        self._recorder._records = []

        logger.info("Sync session started.")

    def pause(self) -> None:
        """Pause both video and emotion recording by pausing the timer."""
        self._timer.pause()
        logger.info("Sync session paused at %dms.", self._timer.elapsed_ms())

    def resume(self) -> None:
        """
        Resume from pause.

        After resuming the timer, seek the video player to the current
        elapsed position so sequential reads continue from the right spot.
        """
        self._timer.resume()
        current_ms = self._timer.elapsed_ms()
        self._video_player.seek_to(current_ms)
        logger.info("Sync session resumed at %dms.", current_ms)

    def stop(self) -> None:
        """Stop the session and freeze the timer."""
        self._timer.stop()
        logger.info(
            "Sync session stopped at %dms. %d synced frames.",
            self._timer.elapsed_ms(),
            len(self._synced_records),
        )

    def run(self) -> Generator[SyncedFrame, None, None]:
        """
        Generator that yields one SyncedFrame per display tick.

        Video frames are read SEQUENTIALLY (no seeking) for performance.
        The timer controls playback speed — the caller sleeps between
        yields to maintain the target display FPS, so the video advances
        at approximately real-time speed.

        The caller is responsible for:
        - Sleeping to maintain display_fps between yields.
        - Rendering the video frame and emotion overlay.
        - Handling keyboard input (pause, quit).
        - Breaking out of the loop when done.

        The generator exits when:
        - The video ends (read_next_frame returns None).
        - The timer elapsed past the video duration.
        - The timer is stopped externally.
        """
        if not self._timer.is_running:
            raise RuntimeError("Call start() before run().")

        tick_count = 0
        last_emotion_record: EmotionRecord | None = None
        last_video_frame: np.ndarray | None = None

        while True:
            # If paused, yield a paused frame (no new data)
            if self._timer.is_paused:
                yield SyncedFrame(
                    video_timestamp_ms=self._timer.elapsed_ms(),
                    video_frame=None,
                    emotion_record=last_emotion_record,
                    is_inference_frame=False,
                )
                continue

            if self._timer.is_stopped:
                break

            current_ms = self._timer.elapsed_ms()

            # Check if we've elapsed past the video duration
            video_duration = self._video_player.duration_ms
            if video_duration > 0 and current_ms > video_duration:
                logger.info("Video ended at %dms.", current_ms)
                break

            # Read the next frame sequentially — fast, no seeking.
            # The caller's sleep between yields controls playback speed.
            read_result = self._video_player.read_next_frame()

            if read_result is None:
                # Reached end of video file
                logger.info("Video ended (EOF) at %dms.", current_ms)
                break

            _video_pos_ms, video_frame = read_result
            last_video_frame = video_frame

            # Run emotion inference every Nth tick
            is_inference = (tick_count % self._emotion_every_n == 0)
            emotion_record = None

            if is_inference:
                emotion_record = self._recorder.record_frame()
                if emotion_record is not None:
                    last_emotion_record = emotion_record

            synced = SyncedFrame(
                video_timestamp_ms=current_ms,
                video_frame=video_frame,
                emotion_record=last_emotion_record,
                is_inference_frame=is_inference,
            )
            self._synced_records.append(synced)
            tick_count += 1

            yield synced

    def export_session_csv(self, filename: str | None = None) -> Path:
        """
        Export the emotion records from this session to CSV.

        Delegates to the EmotionRecorder's export method, which writes
        the standard CSV format from Phase 1.
        """
        return self._recorder.export_csv(filename)
