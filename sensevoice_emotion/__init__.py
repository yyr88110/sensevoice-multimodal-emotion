"""SenseVoice Multimodal Emotion Analysis System"""

__version__ = "4.0.0"
__author__ = "yyr88110"

from .sensevoice_emotion import analyze, extract_acoustic_emotion, analyze_text_emotion, fuse_emotion
from .speaker_baseline import extract_features, build_baseline, calibrate, show_baseline

__all__ = [
    "analyze", "extract_acoustic_emotion", "analyze_text_emotion", "fuse_emotion",
    "extract_features", "build_baseline", "calibrate", "show_baseline",
]
