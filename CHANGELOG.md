# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Updated README with v3.0 architecture diagram

## [2.0.0] - 2026-06-12

### Added
- Multimodal fusion analysis (acoustic + text channels)
- CTC logits probability extraction from SenseVoice
- Text semantic analysis with pragmatic intent detection
- 6 fusion strategies (consistent, acoustic-dominant, text-dominant, etc.)
- Fine-grained emotion labels (无奈, 试探, 好奇, etc.)
- Agent hint generation for AI integration
- Human-friendly output mode

### Changed
- Replaced hardcoded rule-based emotion refinement with proper fusion layer
- Added probability distributions instead of single labels

## [1.0.0] - 2026-06-12

### Added
- Initial release
- SenseVoice-based emotion recognition
- Basic text-to-speech emotion analysis
- Simple rule-based emotion refinement
