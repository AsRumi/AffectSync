# Contributing to AffectSync

Thank you for your interest in contributing. This document covers how to set
up the development environment, run the test suite, and the conventions the
codebase follows.

---

## Development Setup

Follow the same installation steps as in the [README](README.md#setup), then
install the development dependencies:

```bash
python -m pip install pytest
```

No additional dev dependencies are required - the test suite uses only pytest
and the stdlib `unittest.mock` module.

---

## Running the Tests

```bash
python -m pytest tests/ -v
```

The full suite (169 tests) should pass with no hardware present. All webcam,
video file, GPU, and model dependencies are replaced with mocks via dependency
injection.

To run a single test file:

```bash
python -m pytest tests/test_peak_detector.py -v
```

To run a specific test class or function:

```bash
python -m pytest tests/test_peak_detector.py::TestOnsetDetection -v
python -m pytest tests/test_peak_detector.py::TestOnsetDetection::test_detects_onset_after_neutral_baseline -v
```

---

## Project Conventions

These conventions are followed consistently throughout the codebase. New
contributions should match them.

**One module per file.**

**All configuration in `config.py`.**

**Dependency injection throughout.** Every pipeline component accepts its
hardware or model dependencies as optional constructor arguments, defaulting
to creating real instances. This is what makes the test suite hardware-free.
If you are adding a new component that depends on hardware or a model, follow
the same pattern.

```python
# Correct - injectable, testable
class MyComponent:
    def __init__(self, webcam: WebcamCapture | None = None):
        self._webcam = webcam or WebcamCapture()

# Wrong - not testable without real hardware
class MyComponent:
    def __init__(self):
        self._webcam = WebcamCapture()
```

**Use the logging module, not print statements.** Import the logger via:

```python
import logging
logger = logging.getLogger(__name__)
```

Print statements are only acceptable in CLI scripts (`scripts/`) for
user-facing session summaries.

**No bare `except` clauses.** Always catch a specific exception type. If you
genuinely need to catch everything, use `except Exception as exc:` and log
`exc`.

**No stub functions or placeholder comments.**

**Tests for every non-trivial function in `pipeline/` and `utils/`.**

---

## Output Schema

The session JSON schema is defined in the MVP specification. Do not change the
schema of any existing key without opening a discussion first. Downstream
tooling (fine-tuning pipelines, analysis scripts) depends on schema stability.

Adding new top-level keys to the session JSON is acceptable as long as
existing keys are untouched.

---

## Submitting Changes

1. Fork the repository and create a branch from `main`.
2. Make your changes, following the conventions above.
3. Ensure `python -m pytest tests/ -v` passes in full.
4. Open a pull request with a clear description of what changed and why.
