#!/usr/bin/env python3
"""
说话人声学基线系统 v1.0

功能：
  1. 从音频文件提取声学特征（RMS能量、F0基频、语速、停顿模式）
  2. 建立说话人个人基线 profile（均值+标准差）
  3. 对新音频做 Z-score 校准（偏离基线多少个标准差）

用法：
  # 建基线（从多条"正常状态"音频）
  python3 speaker_baseline.py build --speaker shiwei --audio file1.ogg file2.ogg ...

  # 查看基线
  python3 speaker_baseline.py show --speaker shiwei

  # 校准新音频（输出校准后的特征）
  python3 speaker_baseline.py calibrate --speaker shiwei --audio new_file.ogg

  # 提取单条音频特征（不校准）
  python3 speaker_baseline.py extract --audio file.ogg
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile
import numpy as np
from pathlib import Path

PROFILES_DIR = os.path.expanduser("~/.hermes/data/speaker_profiles")


def convert_to_wav(input_path: str) -> str:
    """Convert audio to 16kHz mono wav for feature extraction."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path
    output_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "wav",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    return output_path


def extract_features(audio_path: str) -> dict:
    """
    从音频中提取 4 类声学特征：
    1. RMS Energy（音量）— mean, std, dynamic_range
    2. F0 基频（语调）— mean, std, range
    3. Speech Rate（语速）— 估算音节/秒
    4. Pause Pattern（停顿）— 停顿次数, 平均停顿时长, 停顿占比
    """
    import librosa

    wav_path = convert_to_wav(audio_path)
    cleanup = wav_path != audio_path

    try:
        y, sr = librosa.load(wav_path, sr=16000)
        duration = len(y) / sr

        if duration < 0.5:
            return {"error": "Audio too short (<0.5s)", "duration": duration}

        # --- 1. RMS Energy ---
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        rms_dynamic_range = float(np.max(rms) - np.min(rms)) if len(rms) > 1 else 0.0

        # --- 2. F0 (Pitch) ---
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if len(f0_valid) > 0:
            f0_mean = float(np.mean(f0_valid))
            f0_std = float(np.std(f0_valid))
            f0_range = float(np.max(f0_valid) - np.min(f0_valid))
            voiced_ratio = float(np.sum(~np.isnan(f0)) / len(f0))
        else:
            f0_mean = 0.0
            f0_std = 0.0
            f0_range = 0.0
            voiced_ratio = 0.0

        # --- 3. Speech Rate (approximate) ---
        # 用能量包络的过零率估算：能量高于阈值的连续段 = 音节
        rms_threshold = rms_mean * 0.3
        is_speech = rms > rms_threshold
        # 计算 speech segments 的数量（上升沿 = 新音节开始）
        speech_onsets = np.diff(is_speech.astype(int))
        onset_count = int(np.sum(speech_onsets == 1))
        speech_duration = float(np.sum(is_speech) * 512 / sr)  # hop_length=512
        syllables_per_sec = onset_count / speech_duration if speech_duration > 0 else 0

        # --- 4. Pause Pattern ---
        is_silent = ~is_speech
        silent_runs = []
        run_len = 0
        for s in is_silent:
            if s:
                run_len += 1
            else:
                if run_len > 0:
                    silent_runs.append(run_len * 512 / sr)  # 转秒
                run_len = 0
        if run_len > 0:
            silent_runs.append(run_len * 512 / sr)

        # 只统计 >0.3s 的停顿（短于 0.3s 的是正常辅音间隔）
        pauses = [p for p in silent_runs if p > 0.3]
        pause_count = len(pauses)
        pause_mean_dur = float(np.mean(pauses)) if pauses else 0.0
        pause_ratio = float(sum(pauses) / duration) if duration > 0 else 0.0

        return {
            "duration": round(duration, 2),
            "rms_mean": round(rms_mean, 6),
            "rms_std": round(rms_std, 6),
            "rms_dynamic_range": round(rms_dynamic_range, 6),
            "f0_mean": round(f0_mean, 2),
            "f0_std": round(f0_std, 2),
            "f0_range": round(f0_range, 2),
            "voiced_ratio": round(voiced_ratio, 3),
            "syllables_per_sec": round(syllables_per_sec, 2),
            "pause_count": pause_count,
            "pause_mean_dur": round(pause_mean_dur, 3),
            "pause_ratio": round(pause_ratio, 3),
        }

    finally:
        if cleanup:
            os.unlink(wav_path)


FEATURE_KEYS = [
    "rms_mean", "rms_std", "rms_dynamic_range",
    "f0_mean", "f0_std", "f0_range",
    "voiced_ratio",
    "syllables_per_sec",
    "pause_count", "pause_mean_dur", "pause_ratio",
]


def build_baseline(speaker: str, audio_paths: list) -> dict:
    """
    从多条"正常状态"音频建立说话人基线。
    
    输出：每个特征的 mean 和 std（用于后续 Z-score 校准）。
    至少需要 3 条音频，推荐 5-10 条。
    """
    if len(audio_paths) < 3:
        print(f"⚠️  建议至少 3 条音频（当前 {len(audio_paths)} 条），结果可能不够稳定")

    all_features = []
    for i, path in enumerate(audio_paths):
        print(f"  [{i+1}/{len(audio_paths)}] 提取: {os.path.basename(path)}", file=sys.stderr)
        feat = extract_features(path)
        if "error" in feat:
            print(f"    ⚠️  跳过: {feat['error']}", file=sys.stderr)
            continue
        all_features.append(feat)

    if len(all_features) < 2:
        return {"error": f"有效音频不足（{len(all_features)}条），至少需要2条"}

    # 计算每个特征的 mean 和 std
    baseline = {}
    for key in FEATURE_KEYS:
        values = [f[key] for f in all_features if key in f]
        if values:
            baseline[key] = {
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values)), 6),
                "min": round(float(np.min(values)), 6),
                "max": round(float(np.max(values)), 6),
                "n": len(values),
            }

    profile = {
        "speaker": speaker,
        "n_samples": len(all_features),
        "features": baseline,
        "raw_samples": all_features,  # 保留原始值，方便后续增量更新
    }

    # 保存
    os.makedirs(PROFILES_DIR, exist_ok=True)
    profile_path = os.path.join(PROFILES_DIR, f"{speaker}.json")
    with open(profile_path, "w") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    print(f"✅ 基线已保存: {profile_path}", file=sys.stderr)

    return profile


def calibrate(speaker: str, audio_path: str) -> dict:
    """
    对新音频做 Z-score 校准。
    
    返回：每个特征的原始值 + Z-score（偏离基线多少个标准差）。
    Z-score > 0 = 高于该人的正常值
    Z-score < 0 = 低于该人的正常值
    """
    profile_path = os.path.join(PROFILES_DIR, f"{speaker}.json")
    if not os.path.exists(profile_path):
        return {"error": f"找不到 {speaker} 的基线文件: {profile_path}"}

    with open(profile_path) as f:
        profile = json.load(f)

    features = extract_features(audio_path)
    if "error" in features:
        return features

    baseline = profile["features"]
    calibrated = {}

    for key in FEATURE_KEYS:
        raw = features.get(key, 0)
        if key in baseline:
            b_mean = baseline[key]["mean"]
            b_std = baseline[key]["std"]
            if b_std > 0:
                z_score = (raw - b_mean) / b_std
            else:
                z_score = 0.0  # std=0 表示该特征在基线中无变化
            calibrated[key] = {
                "raw": raw,
                "baseline_mean": b_mean,
                "baseline_std": b_std,
                "z_score": round(z_score, 2),
            }
        else:
            calibrated[key] = {"raw": raw, "z_score": None}

    # 生成摘要：哪些特征偏离最大
    deviations = []
    for key, val in calibrated.items():
        if val.get("z_score") is not None:
            deviations.append((key, abs(val["z_score"])))
    deviations.sort(key=lambda x: -x[1])

    summary_parts = []
    for key, abs_z in deviations[:3]:
        if abs_z > 1.5:
            z = calibrated[key]["z_score"]
            direction = "偏高" if z > 0 else "偏低"
            summary_parts.append(f"{key} {direction}({z:+.1f}σ)")

    return {
        "speaker": speaker,
        "audio": audio_path,
        "features": calibrated,
        "top_deviations": summary_parts,
        "profile_n_samples": profile["n_samples"],
    }


def show_baseline(speaker: str) -> dict:
    """查看已建立的基线 profile。"""
    profile_path = os.path.join(PROFILES_DIR, f"{speaker}.json")
    if not os.path.exists(profile_path):
        return {"error": f"找不到 {speaker} 的基线文件: {profile_path}"}
    with open(profile_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Speaker Baseline System v1.0")
    sub = parser.add_subparsers(dest="command")

    # build
    p_build = sub.add_parser("build", help="Build baseline from normal-state audio files")
    p_build.add_argument("--speaker", required=True, help="Speaker ID (e.g. shiwei)")
    p_build.add_argument("--audio", nargs="+", required=True, help="Audio file paths")

    # show
    p_show = sub.add_parser("show", help="Show existing baseline")
    p_show.add_argument("--speaker", required=True)

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Calibrate new audio against baseline")
    p_cal.add_argument("--speaker", required=True)
    p_cal.add_argument("--audio", required=True)

    # extract
    p_ext = sub.add_parser("extract", help="Extract features from audio (no calibration)")
    p_ext.add_argument("--audio", required=True)

    args = parser.parse_args()

    if args.command == "build":
        result = build_baseline(args.speaker, args.audio)
    elif args.command == "show":
        result = show_baseline(args.speaker)
    elif args.command == "calibrate":
        result = calibrate(args.speaker, args.audio)
    elif args.command == "extract":
        result = extract_features(args.audio)
    else:
        parser.print_help()
        return

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()