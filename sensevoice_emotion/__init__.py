"""SenseVoice 多模态情绪融合分析系统"""

__version__ = "3.0.1"
__author__ = "yyr88110"

from .sensevoice_emotion import analyze, extract_acoustic_emotion, analyze_text_emotion, fuse_emotion

__all__ = ["analyze", "extract_acoustic_emotion", "analyze_text_emotion", "fuse_emotion"]
