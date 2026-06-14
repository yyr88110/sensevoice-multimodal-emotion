# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2026-06-14

### Added
- **Speaker Baseline System** (`speaker_baseline.py`): per-speaker acoustic feature profiling
  - 11-dimensional feature extraction: RMS energy (3), F0 pitch (3), voiced ratio, speech rate, pause patterns (3)
  - Z-score calibration: compare new audio against speaker's personal baseline
  - `build` / `calibrate` / `show` / `extract` commands
- New dependency: `librosa>=0.10` for acoustic feature extraction

### Use Case
Solves the "soft-spoken speaker" problem: a person who naturally speaks softly won't be
misclassified as "sad" -- the system compares against *their own* baseline, not universal averages.

## [3.0.1] - 2026-06-12

### Fixed
- Suppress funasr version output to stdout (redirected to stderr)
- Add error handling for model loading (SenseVoice + Whisper)
- Add error handling for Whisper transcription

### Changed
- Add `openai-whisper` to requirements.txt

## [3.0.0] - 2026-06-12

### Added
- **Whisper ASR integration**: Dual-engine architecture with Whisper for transcription + SenseVoice for emotion
- `asr.whisper` and `asr.sensevoice` fields in output for ASR comparison
- Examples directory with 3 sample outputs (probing, frustrated, satisfied)
- MIT License
- .gitignore

### Changed
- Architecture changed from single-engine (SenseVoice) to dual-engine (Whisper + SenseVoice)
- Text channel now uses Whisper transcription for better accuracy
- SenseVoice now only used for acoustic emotion extraction (CTC logits)

## [2.0.0] - 2026-06-12

### Added
- Multimodal fusion analysis (acoustic + text channels)
- CTC logits probability extraction from SenseVoice
- Text semantic analysis with pragmatic intent detection
- 6 fusion strategies
- Fine-grained emotion labels
- Agent hint generation for AI integration

## [1.0.0] - 2026-06-12

### Added
- Initial release
- SenseVoice-based emotion recognition
- Basic text-to-speech emotion analysis
