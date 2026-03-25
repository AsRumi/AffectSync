# AffectSync

**Affective Data Infrastructure for Emotionally-Aware Generative AI**

AffectSync is a local deep learning pipeline that records a viewer's real-time facial emotions while a video plays, synchronizes those emotion labels to video timestamps, transcribes the audio, and exports a richly annotated JSON dataset. The output is purpose-built as training data for generative models that need a ground-truth human emotional response signal.

---

## Motivation

Generative models for creative content (comedy, drama, advertising) are trained on what content _exists_, not on what content makes us _feel something_. A model trained purely on transcripts learns syntax and structure, but has no feedback signal for what actually lands. AffectSync is the data collection infrastructure that produces this signal: every frame of a video paired with a measured viewer emotional response and the transcript text spoken at that moment.

---

## Pipeline Overview

```
[ Video File ] ──────────────────────────────────────┐
                                                     │
[ Video Player ]                             [ Audio Extractor ]
(cv2.VideoCapture)                             (ffmpeg-python)
       │                                             │
[ Frame Sync Controller ]     ◄── shared ──►   [ SessionTimer ]
       │                           clock             │
[ Webcam Feed ]                              [ Whisper STT Model ]
       │                                             │
[ Face Detector ]                            [ Transcript Aligner ]
(OpenCV Haar Cascade)                                │
       │                                             ▼
[ Emotion Classifier ]         ──────────►  [ Session Assembler ]
  (DeepFace / FER+)                                  │
                                              [ Peak Detector ]
                                                     │
                                             [ JSON Output File ]
```

| Module              | Responsibility                                | Primary Tool            |
| ------------------- | --------------------------------------------- | ----------------------- |
| `VideoPlayer`       | Serve frames sequentially, expose timestamps  | `cv2.VideoCapture`      |
| `WebcamCapture`     | Capture viewer webcam feed                    | `cv2.VideoCapture`      |
| `FaceDetector`      | Locate largest face in frame                  | OpenCV Haar Cascade     |
| `EmotionClassifier` | Classify cropped face into 7 emotion labels   | DeepFace (FER+ weights) |
| `SyncController`    | Coordinate video + emotion on a shared clock  | `SessionTimer`          |
| `AudioExtractor`    | Extract 16kHz mono WAV from video             | `ffmpeg-python`         |
| `Transcriber`       | Transcribe audio to time-aligned segments     | `openai-whisper`        |
| `TranscriptAligner` | Convert Whisper timestamps to milliseconds    | Python                  |
| `SessionAssembler`  | Merge emotion timeline + transcript into JSON | Python                  |
| `PeakDetector`      | Identify emotional peaks in the timeline      | Python                  |

---

## Output Schema

Each session produces a single JSON file:

```json
{
  "session_id": "3f2a1b4c-...",
  "video_source": "clip.mp4",
  "video_duration_ms": 120000,
  "viewer_id": "anonymous",
  "recorded_at": "2025-01-15T14:32:00+00:00",
  "emotion_timeline": [
    {
      "timestamp_ms": 1040,
      "emotion": "happy",
      "confidence": 0.87,
      "face_detected": true,
      "all_scores": { "happy": 0.87, "neutral": 0.08, "surprised": 0.05 }
    }
  ],
  "transcript_segments": [
    {
      "start_ms": 980,
      "end_ms": 3200,
      "text": "So I was at the DMV the other day..."
    }
  ],
  "aligned_annotations": [
    {
      "timestamp_ms": 1040,
      "emotion": "happy",
      "confidence": 0.87,
      "face_detected": true,
      "transcript_segment": "So I was at the DMV the other day...",
      "segment_position": "mid"
    }
  ],
  "annotated_peaks": [
    {
      "peak_type": "onset",
      "emotion": "happy",
      "start_ms": 1040,
      "end_ms": 1040,
      "duration_ms": 0,
      "peak_confidence": 0.87,
      "transcript_segment": "So I was at the DMV the other day..."
    }
  ]
}
```

**Peak types:**

- `onset` - sharp transition from a neutral baseline into a non-neutral emotion.
- `sustained` - non-neutral emotion held continuously for ≥ 500ms.
- `spike` - single frame where confidence exceeds 0.80.

---

## Tech Stack

- **Python 3.10+**
- [DeepFace](https://github.com/serengil/deepface) - emotion recognition (FER+ weights)
- [OpenCV](https://opencv.org/) - webcam capture, face detection, video playback
- [openai-whisper](https://github.com/openai/whisper) - local speech-to-text transcription
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) - audio extraction
- [ffmpeg](https://ffmpeg.org/) - system binary required by ffmpeg-python
- [TensorFlow](https://www.tensorflow.org/) - DeepFace backend
- [pytest](https://pytest.org/) - test suite (169 tests, all hardware-mocked)

---

## Setup

### Prerequisites

- Python 3.10 or higher
- A connected webcam
- [ffmpeg](https://ffmpeg.org/download.html) installed as a system binary

**Install ffmpeg on Windows:**

```powershell
winget install ffmpeg
```

**Install ffmpeg on macOS:**

```bash
brew install ffmpeg
```

**Install ffmpeg on Linux:**

```bash
sudo apt install ffmpeg
```

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/AsRumi/AffectSync.git
cd AffectSync
```

**2. Create and activate a virtual environment**

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m ensurepip --upgrade
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **First run note:** DeepFace will download FER+ model weights (~500 MB) on the first inference call. Whisper will download the `base` model (~140 MB) on the first transcription. Both are cached locally after the initial download.

**4. Verify the environment**

```bash
python -m pytest tests/ -v
```

All 169 tests should pass. No webcam or video file is required to run the test suite, all hardware dependencies are mocked.

---

## Usage

### Run a synchronized video session

Records viewer emotions while a video plays. After playback, extracts audio, transcribes it with Whisper, and exports the full session JSON.

```bash
python scripts/run_video_session.py --video path/to/clip.mp4
```

**With a custom output filename:**

```bash
python scripts/run_video_session.py --video clip.mp4 --output my_session.csv --session-output my_session.json
```

**Skip transcription:**

```bash
python scripts/run_video_session.py --video clip.mp4 --no-transcription
```

**All options:**

```
--video               Path to the video file (required)
--output              CSV filename for emotion timeline (default: emotion_session.csv)
--transcript-output   JSON filename for transcript (default: <video_stem>_transcript.json)
--session-output      JSON filename for full session dataset (default: <video_stem>_session.json)
--display-fps         Main loop tick rate (default: 30)
--emotion-fps         Emotion inference rate (default: 10)
--no-webcam-preview   Hide the webcam preview window
--no-transcription    Skip audio extraction and Whisper transcription
```

**Controls during playback:**

- `SPACE` - pause / resume
- `Q` - quit early

### Run a standalone emotion session without a video

Records emotions from webcam only, exports to CSV. Can be used to test your setup.

```bash
python scripts/run_emotion_session.py --duration 30
```

```
--duration   Recording duration in seconds (default: 30)
--output     Output CSV filename
--fps        Target inference FPS (default: 10)
```

### Test webcam + emotion detection

Quick sanity check that opens a live preview with emotion labels overlaid.

```bash
python scripts/test_webcam_emotion.py
python scripts/test_webcam_emotion.py --no-preview   # log-only, no window
```

## Design Decisions

**Single-threaded sync loop.** Video playback and emotion inference run in one thread. The main loop ticks at 30 FPS for smooth display; emotion inference runs every 3rd tick at 10 FPS. This was done to avoid threading complexity at the MVP stage.

**Shared monotonic clock.** `SessionTimer` is the single source of truth for all timestamps.

**Sequential video reads.** `VideoPlayer.read_next_frame()` reads frames sequentially rather than seeking on every tick. Seeking on H.264 is ~100× slower due to keyframe decode cost. Seeking is reserved for post-resume repositioning only.

**Dependency injection throughout.** Every pipeline component accepts its dependencies as constructor arguments, defaulting to real instances.

**Post-processing separation.** Transcription and peak detection run after video playback ends, not during.

---

## Limitations (MVP Scope)

- Single viewer per session. Multi-viewer aggregation is a post-MVP feature.
- Video clips capped at 5 minutes (`MAX_VIDEO_DURATION_SEC`).
- Emotion inference runs on CPU which is why it was downsampled to 10FPS.
- Whisper `base` model is used by default. Accuracy on overlapping dialogue or heavy accents improves significantly with `small` or `medium` at the cost of transcription time.
- The pipeline assumes a controlled capture environment: reasonable lighting, viewer facing the camera, single face in frame.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**AsRumi** — [github.com/AsRumi](https://github.com/AsRumi)
