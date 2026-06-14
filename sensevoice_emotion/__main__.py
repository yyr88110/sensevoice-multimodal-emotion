#!/usr/bin/env python3
"""
多模态情绪融合分析 v3.0 — Whisper + SenseVoice 双引擎

架构：
  Whisper (ASR)              SenseVoice (情绪)
       ↓                         ↓
  精准转写文本                CTC logits → 5类概率
       ↓                         ↓
       └─────── 融合层 ──────────┘
                 ↓
         最终情绪判断 + 融合理由

用法：
  python3 sensevoice_analyze.py <audio_file> [--json]

输出：
  声学概率分布 + 文本语义分析 + 融合决策 + agent hint
"""

import sys
import os
import json
import re
import subprocess
import tempfile
import time
import logging
from typing import Optional

# Redirect funasr logging to stderr to keep stdout clean for JSON output
logging.getLogger().setLevel(logging.WARNING)
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        handler.stream = sys.stderr

# ============================================================
# SenseVoice 情绪 token 映射
# ============================================================
EMO_TOKEN_MAP=***
    25001: "happy",
    25002: "sad",
    25003: "angry",
    25004: "neutral",
    25009: "unk",
}
EMO_TOKEN_IDS=list(E...s())
EMO_LABELS_ZH = {
    "happy": "开心",
    "sad": "难过",
    "angry": "生气",
    "neutral": "平静",
    "unk": "未知",
}

# ============================================================
# 声学通道：从 SenseVoice CTC logits 提取情绪概率
# ============================================================
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            from funasr import AutoModel
            _model = AutoModel(
                model="iic/SenseVoiceSmall",
                device="cpu",
                disable_update=True,
            )
        except Exception as e:
            print(f"Error loading SenseVoice model: {e}", file=sys.stderr)
            raise
    return _model




# ============================================================
# Whisper ASR 引擎
# ============================================================
_whisper_model = None

def get_whisper_model():
    """Lazy load Whisper model (singleton)"""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            _whisper_model = whisper.load_model("small")
        except Exception as e:
            print(f"Error loading Whisper model: {e}", file=sys.stderr)
            raise
    return _whisper_model


def whisper_transcribe(audio_path: str) -> str:
    """用 Whisper 做精准 ASR 转写"""
    try:
        model = get_whisper_model()
        result = model.transcribe(audio_path, language="zh")
        return result["text"].strip()
    except Exception as e:
        print(f"Whisper transcription failed: {e}", file=sys.stderr)
        return ""

def convert_audio(input_path: str) -> str:
    """Convert to 16kHz mono wav"""
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


def extract_acoustic_emotion(audio_path: str) -> dict:
    """
    从 SenseVoice CTC logits 提取情绪概率分布。
    
    SenseVoice 的 CTC 输出序列前4个位置是特殊token：
      位置 0: language query
      位置 1: event query
      位置 2: emotion query  ← 我们要这个
      位置 3: textnorm query
    
    情绪 token ID:
      25001=happy, 25002=sad, 25003=angry, 25004=neutral, 25009=unk
    """
    import torch
    import torch.nn.functional as F

    model = get_model()
    inner = model.model

    # 1. 提取 fbank 特征
    from funasr.utils.load_utils import load_audio_text_image_video
    from funasr.utils.load_utils import extract_fbank

    frontend = model.kwargs.get("frontend", None)
    if frontend is None:
        # 通过 model 对象获取 frontend
        frontend = getattr(model, "frontend", None)
    if frontend is None:
        return {"error": "Cannot access frontend", "probs": {}, "raw_emotion": "NEUTRAL"}

    audio_sample_list = load_audio_text_image_video(
        audio_path, fs=frontend.fs, audio_fs=16000, data_type="sound"
    )
    speech, speech_lengths = extract_fbank(
        audio_sample_list, data_type="sound", frontend=frontend
    )

    # 2. 构建输入（复制 inference 的逻辑）
    device = next(inner.parameters()).device
    speech = speech.to(device)
    speech_lengths = speech_lengths.to(device)

    # language query
    lid = inner.lid_dict.get("auto", 0)
    language_query = inner.embed(
        torch.LongTensor([[lid]]).to(device)
    ).repeat(speech.size(0), 1, 1)

    # event + emotion query
    event_emo_query = inner.embed(
        torch.LongTensor([[1, 2]]).to(device)
    ).repeat(speech.size(0), 1, 1)

    # textnorm query
    textnorm_query = inner.embed(
        torch.LongTensor([[inner.textnorm_dict["withitn"]]]).to(device)
    ).repeat(speech.size(0), 1, 1)

    speech = torch.cat((textnorm_query, speech), dim=1)
    speech_lengths += 1
    input_query = torch.cat((language_query, event_emo_query), dim=1)
    speech = torch.cat((input_query, speech), dim=1)
    speech_lengths += 3

    # 3. Encoder forward
    with torch.no_grad():
        encoder_out, encoder_out_lens = inner.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        ctc_logits = inner.ctc.log_softmax(encoder_out)

    # 4. 提取位置 2（emotion query）的 logits
    emo_logits = ctc_logits[0, 2, :]  # shape: [vocab_size]

    # 5. 提取5类情绪token的概率
    emo_token_logits = torch.tensor([emo_logits[tid].item() for tid in EMO_TOKEN_IDS])
    emo_probs = F.softmax(emo_token_logits, dim=0).numpy()

    # 6. 整理结果
    prob_dict = {}
    for idx, tid in enumerate(EMO_TOKEN_IDS):
        label = EMO_TOKEN_MAP[tid]
        prob_dict[label] = {
            "prob": float(emo_probs[idx]),
            "prob_pct": f"{float(emo_probs[idx]) * 100:.1f}%",
            "label_zh": EMO_LABELS_ZH[label],
        }

    # 最高概率的情绪
    best_idx = int(emo_probs.argmax())
    best_emotion = EMO_TOKEN_MAP[EMO_TOKEN_IDS[best_idx]]
    best_prob = float(emo_probs[best_idx])

    # 同时获取文本（用标准 inference）
    results = model.generate(
        input=audio_path, language="auto", use_itn=True, batch_size_s=60
    )
    raw_sv_text = results[0].get("text", "") if results else ""
    sv_text = re.sub(r"<\|[^|]+\|>", "", raw_sv_text).strip()

    # 从raw_tags提取原始声学标签
    raw_emotion_match = re.search(r"<\|(HAPPY|SAD|ANGRY|NEUTRAL)\|>", raw_sv_text)
    raw_emotion = raw_emotion_match.group(1) if raw_emotion_match else "NEUTRAL"

    return {
        "sv_text": sv_text,
        "raw_emotion": raw_emotion,
        "best_emotion": best_emotion,
        "best_prob": best_prob,
        "probs": prob_dict,
        "raw_tags": raw_sv_text,
    }


# ============================================================
# 文本通道：LLM 语义情绪分析
# ============================================================

def analyze_text_emotion(text: str) -> dict:
    """
    用规则+启发式分析文本语义情绪。
    
    比之前的硬编码映射更细：
    - 分析句式结构（反问、感叹、陈述）
    - 分析语用意图（试探、反驳、确认、抱怨）
    - 分析情感词（显性情绪词 + 隐性情绪标记）
    """
    if not text:
        return {
            "semantic_emotion": "neutral",
            "confidence": 0.0,
            "intent": "unknown",
            "signals": [],
        }

    signals = []
    scores = {
        "happy": 0.0,
        "sad": 0.0,
        "angry": 0.0,
        "neutral": 0.0,
    }

    # --- 1. 显性情绪词 ---
    happy_words = ["开心", "高兴", "太好了", "哈哈", "笑死", "厉害", "牛", "绝了", "终于", "居然", "不错", "挺好"]
    sad_words = ["难过", "伤心", "可惜", "遗憾", "哎", "算了", "没办法", "无所谓"]
    angry_words = ["生气", "烦", "讨厌", "受不了", "凭什么", "不公平", "气死", "滚"]

    for w in happy_words:
        if w in text:
            scores["happy"] += 0.3
            signals.append(f"情绪词'{w}'→开心+0.3")
    for w in sad_words:
        if w in text:
            scores["sad"] += 0.3
            signals.append(f"情绪词'{w}'→难过+0.3")
    for w in angry_words:
        if w in text:
            scores["angry"] += 0.3
            signals.append(f"情绪词'{w}'→生气+0.3")

    # --- 2. 语用意图分析 ---
    # 反问句（"不是...吗？"、"还要...？"）
    rhetorical_patterns = [
        (r"不是.*吗", "反问/反驳", 0.2),
        (r"还要.*[？?]", "无奈质问", 0.15),
        (r"你觉得.*[？?]", "试探", 0.1),
        (r"你说呢[？?]", "试探", 0.1),
        (r"凭什么.*[？?]", "委屈质问", 0.25),
        (r"怎么又.*", "不耐烦", 0.2),
        (r"算了", "放弃/无奈", 0.15),
        (r"随便", "敷衍/放弃", 0.15),
        (r"都行", "敷衍/放弃", 0.1),
    ]

    for pattern, intent, weight in rhetorical_patterns:
        if re.search(pattern, text):
            signals.append(f"语用'{intent}'(匹配'{pattern}')+{weight}")
            # 根据意图类型分配分数
            if intent in ("反问/反驳", "无奈质问", "委屈质问"):
                scores["angry"] += weight * 0.5
                scores["sad"] += weight * 0.5
            elif intent in ("试探",):
                scores["neutral"] += weight  # 试探通常是中性的
            elif intent in ("放弃/无奈", "敷衍/放弃"):
                scores["sad"] += weight
            elif intent in ("不耐烦",):
                scores["angry"] += weight

    # --- 3. 句式结构 ---
    excl_count = text.count("！") + text.count("!")
    quest_count = text.count("？") + text.count("?")

    if excl_count > 0:
        scores["angry"] += 0.1 * excl_count
        signals.append(f"感叹号×{excl_count}→情绪强度+{0.1 * excl_count}")
    if quest_count > 0:
        # 问号本身不偏向某情绪，但增加"不确定性"
        signals.append(f"问号×{quest_count}→增加不确定性")

    # --- 4. 否定词叠加（"不是...还要...?"模式）---
    negation_count = len(re.findall(r"不是|不要|不行|不是吧|还要|还要我", text))
    if negation_count >= 2:
        scores["angry"] += 0.15
        scores["sad"] += 0.1
        signals.append(f"多重否定(×{negation_count})→无奈+0.25")

    # --- 5. 语气词 ---
    particles = {
        "呢": 0.05,  # 中性偏疑问
        "吧": 0.05,  # 中性偏确认
        "啊": 0.1,   # 感叹
        "嘛": 0.05,  # 撒娇/不耐烦
        "嗯": 0.1,   # 敷衍
        "哦": 0.1,   # 敷衍
    }
    for p, w in particles.items():
        if p in text:
            scores["neutral"] += w
            signals.append(f"语气词'{p}'→中性+{w}")

    # --- 6. "假装"类元认知词 ---
    meta_words = ["假装", "装", "演", "故意"]
    for w in meta_words:
        if w in text:
            scores["happy"] += 0.15  # 假装笑 = 实际不开心但表面开心
            signals.append(f"元认知词'{w}'→表面开心+0.15（可能掩盖真实情绪）")

    # --- 计算最终语义情绪 ---
    total = sum(scores.values())
    if total > 0:
        probs = {k: v / total for k, v in scores.items()}
    else:
        probs = {"happy": 0.1, "sad": 0.1, "angry": 0.1, "neutral": 0.7}

    best_semantic = max(probs, key=probs.get)
    confidence = probs[best_semantic]

    # 确定语用意图
    intent = "陈述"
    if quest_count > 0 and any(s.startswith("语用'试探'") for s in signals):
        intent = "试探"
    elif quest_count > 0 and any("反问" in s or "质问" in s for s in signals):
        intent = "反问"
    elif quest_count > 0:
        intent = "疑问"
    elif excl_count > 0:
        intent = "感叹"

    return {
        "semantic_emotion": best_semantic,
        "semantic_emotion_zh": EMO_LABELS_ZH.get(best_semantic, best_semantic),
        "confidence": round(confidence, 3),
        "probs": {k: round(v, 3) for k, v in probs.items()},
        "intent": intent,
        "signals": signals,
    }


def calibrate_baseline(audio_path: str, speaker: str) -> dict:
    """
    用说话人基线校准当前音频，返回 Z-score 偏差。
    如果基线不存在或校准失败，返回 None。
    """
    try:
        # 动态导入 speaker_baseline 模块
        import importlib.util
        baseline_path = os.path.join(os.path.dirname(__file__), "speaker_baseline.py")
        if not os.path.exists(baseline_path):
            # 尝试 scripts 目录
            baseline_path = os.path.expanduser("~/.hermes/scripts/speaker_baseline.py")
        if not os.path.exists(baseline_path):
            return None

        spec = importlib.util.spec_from_file_location("speaker_baseline", baseline_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        profile_path = os.path.expanduser(f"~/.hermes/data/speaker_profiles/{speaker}.json")
        if not os.path.exists(profile_path):
            return None

        result = mod.calibrate(speaker, audio_path)
        if "error" in result:
            return None
        return result
    except Exception as e:
        print(f"Baseline calibration failed: {e}", file=sys.stderr)
        return None


def interpret_baseline_emotion(baseline: dict) -> dict:
    """
    将 Z-score 偏差映射为情绪方向信号。

    逻辑：
    - 音量↑ + 语速↑ + 语调起伏↑ → happy/angry（激动型）
    - 音量↓ + 语速↓ + 停顿↑ → sad（低落型）
    - 各维度偏离不大 → neutral
    - 单维度极端偏离（>2σ）也给信号
    """
    features = baseline.get("features", {})

    def z(key):
        val = features.get(key, {})
        return val.get("z_score", 0) or 0

    # 提取关键 Z-scores
    z_vol = z("rms_mean")          # 音量
    z_vol_range = z("rms_dynamic_range")  # 音量变化幅度
    z_f0 = z("f0_mean")            # 基频（音调高低）
    z_f0_var = z("f0_std")         # 语调起伏
    z_speed = z("syllables_per_sec")  # 语速
    z_pause_dur = z("pause_mean_dur")  # 停顿时长（负=停顿短=语速快）
    z_pause_ratio = z("pause_ratio")   # 停顿占比

    # --- 情绪维度打分 ---
    scores = {"happy": 0.0, "sad": 0.0, "angry": 0.0, "neutral": 0.0}
    signals = []

    # 激动维度（音量↑ + 语速↑ + 语调起伏↑）
    arousal = (z_vol * 0.35 + z_speed * 0.25 + z_f0_var * 0.2 + z_vol_range * 0.2)
    if arousal > 1.0:
        scores["happy"] += arousal * 0.4
        scores["angry"] += arousal * 0.3
        signals.append(f"高唤醒({arousal:+.1f}σ)→激动倾向")
    elif arousal < -1.0:
        scores["sad"] += abs(arousal) * 0.5
        scores["neutral"] += abs(arousal) * 0.2
        signals.append(f"低唤醒({arousal:+.1f}σ)→低落倾向")

    # 愤怒专项（音量↑ + 基频↑ + 语调起伏大）
    anger_signal = (z_vol * 0.4 + z_f0 * 0.3 + z_f0_var * 0.3)
    if anger_signal > 1.5:
        scores["angry"] += anger_signal * 0.3
        signals.append(f"愤怒信号({anger_signal:+.1f}σ)：音量/音调/语调均偏高")

    # 难过专项（音量↓ + 语速↓ + 停顿长↑）
    sad_signal = (-z_vol * 0.35 + (-z_speed) * 0.25 + z_pause_dur * 0.2 + z_pause_ratio * 0.2)
    if sad_signal > 1.0:
        scores["sad"] += sad_signal * 0.4
        signals.append(f"难过信号({sad_signal:+.1f}σ)：音量/语速偏低，停顿偏长")

    # 兴奋专项（语速↑ + 基频↑ + 音量↑，但语调起伏不一定大）
    excite_signal = (z_speed * 0.35 + z_f0 * 0.3 + z_vol * 0.35)
    if excite_signal > 1.5:
        scores["happy"] += excite_signal * 0.35
        signals.append(f"兴奋信号({excite_signal:+.1f}σ)：语速/音调/音量均偏高")

    # 沉默/敷衍（语速极低 + 停顿占比极高）
    if z_speed < -1.5 and z_pause_ratio > 1.5:
        scores["sad"] += 0.3
        scores["neutral"] += 0.2
        signals.append("低语速+高停顿→沉默/敷衍倾向")

    # 无明显偏离 → neutral
    if not signals:
        scores["neutral"] += 0.5
        signals.append("各维度偏离<1σ，情绪平稳")

    # 归一化
    total = sum(scores.values())
    if total > 0:
        probs = {k: round(v / total, 3) for k, v in scores.items()}
    else:
        probs = {"happy": 0.1, "sad": 0.1, "angry": 0.1, "neutral": 0.7}

    best = max(probs, key=probs.get)
    conf = probs[best]

    return {
        "best_emotion": best,
        "best_prob": conf,
        "probs": probs,
        "signals": signals,
        "top_deviations": baseline.get("top_deviations", []),
        "speaker": baseline.get("speaker", ""),
    }


# ============================================================
# 融合层：声学 + 文本 + 基线偏差 → 最终判断
# ============================================================

def fuse_emotion(acoustic: dict, textual: dict, baseline: dict = None) -> dict:
    """
    多模态融合决策（支持三路融合）：

    1. 声学提供"情绪基调"（大方向：开心/难过/生气/平静）
    2. 文本提供"语义细节"（具体意图和细微差别）
    3. 基线偏差提供"个人校准"（偏离该人正常状态的程度）
    """
    acoustic_best = acoustic.get("best_emotion", "neutral")
    acoustic_prob = acoustic.get("best_prob", 0.0)
    textual_best = textual.get("semantic_emotion", "neutral")
    textual_conf = textual.get("confidence", 0.0)
    intent = textual.get("intent", "陈述")

    fusion_reason = []
    conflict = False

    # --- 基线偏差通道（如果有）---
    baseline_best = None
    baseline_conf = 0.0
    baseline_signals = []
    if baseline and baseline.get("features"):
        baseline_interp = interpret_baseline_emotion(baseline)
        baseline_best = baseline_interp["best_emotion"]
        baseline_conf = baseline_interp["best_prob"]
        baseline_signals = baseline_interp["signals"]
        fusion_reason.append(f"基线偏差通道：{baseline_best}({baseline_conf:.0%}) | {'; '.join(baseline_signals[:2])}")

    # --- 情况1：声学和文本一致 ---
    if acoustic_best == textual_best:
        final = acoustic_best
        final_conf = min(acoustic_prob * 0.6 + textual_conf * 0.4, 0.99)
        fusion_reason.append(f"声学({acoustic_best})与文本({textual_best})一致，置信度叠加")

    # --- 情况2：声学高置信度 + 文本低置信度 ---
    elif acoustic_prob > 0.6 and textual_conf < 0.35:
        final = acoustic_best
        final_conf = acoustic_prob
        fusion_reason.append(f"声学高置信({acoustic_prob:.0%})且文本信号弱({textual_conf:.0%})，采信声学")

    # --- 情况3：文本高置信度 + 声学低置信度 ---
    elif textual_conf > 0.5 and acoustic_prob < 0.4:
        final = textual_best
        final_conf = textual_conf
        fusion_reason.append(f"文本高置信({textual_conf:.0%})且声学分散({acoustic_prob:.0%})，采信文本")

    # --- 情况4：声学是neutral但文本有明确情绪 ---
    elif acoustic_best == "neutral" and textual_best != "neutral" and textual_conf > 0.4:
        final = textual_best
        final_conf = textual_conf * 0.8  # 降低一点，因为声学不支持
        fusion_reason.append(f"声学=平静但文本明确={textual_best}({textual_conf:.0%})，倾向文本")

    # --- 情况5：文本是neutral但声学有明确情绪 ---
    elif textual_best == "neutral" and acoustic_best != "neutral" and acoustic_prob > 0.5:
        final = acoustic_best
        final_conf = acoustic_prob * 0.8
        fusion_reason.append(f"文本=平静但声学明确={acoustic_best}({acoustic_prob:.0%})，倾向声学")

    # --- 情况6：真正的冲突 ---
    else:
        conflict = True
        # 语用意图可以帮助解决冲突
        if intent == "试探":
            final = "neutral"  # 试探通常是表面中性的
            final_conf = 0.5
            fusion_reason.append(f"语用意图=试探，表面中性掩盖真实情绪")
        elif intent == "反问":
            final = textual_best  # 反问的情绪通常在文本里
            final_conf = textual_conf * 0.7
            fusion_reason.append(f"反问语气，文本情绪({textual_best})优先")
        else:
            # 无法确定，取概率更高的
            if acoustic_prob > textual_conf:
                final = acoustic_best
                final_conf = acoustic_prob * 0.6
                fusion_reason.append(f"冲突无法解决，采信置信度更高的声学({acoustic_prob:.0%})")
            else:
                final = textual_best
                final_conf = textual_conf * 0.6
                fusion_reason.append(f"冲突无法解决，采信置信度更高的文本({textual_conf:.0%})")

    # --- 基线偏差修正（三路融合的核心）---
    # 基线偏差通道可以在以下情况修正声学/文本的判断：
    # 1. 基线和声学一致 → 提升置信度
    # 2. 基线强烈偏离(>2σ)但声学说neutral → 基线有话语权
    # 3. 基线和声学冲突 → 看偏离幅度决定是否修正
    if baseline_best and baseline_conf > 0.3:
        # 基线强烈偏离且与当前判断矛盾
        if baseline_best != final and baseline_conf > 0.5:
            # 检查是否有极端偏离（>2σ）
            extreme_devs = [d for d in baseline.get("top_deviations", []) if "2." in d or "3." in d or "4." in d or "5." in d or "6." in d or "7." in d or "8." in d]
            if extreme_devs:
                old_final = final
                final = baseline_best
                final_conf = baseline_conf * 0.6  # 基线修正但给较低置信度
                fusion_reason.append(f"⚠️ 基线极端偏离修正：{old_final}→{final} | {', '.join(extreme_devs[:2])}")
                conflict = True
            # 非极端偏离但基线置信度高 → 提示但不覆盖
            elif baseline_conf > 0.6:
                fusion_reason.append(f"基线倾向{baseline_best}({baseline_conf:.0%})但偏离不极端，维持{final}")

        # 基线和当前判断一致 → 提升置信度
        elif baseline_best == final:
            final_conf = min(final_conf + 0.1, 0.99)
            fusion_reason.append(f"基线({baseline_best})与融合({final})一致，置信度+10%")

    # --- 细粒度标注 ---
    # 在最终情绪基础上，结合语用意图给出更细的描述
    refined_map = {
        ("happy", "试探"): "调侃",
        ("happy", "感叹"): "兴奋",
        ("happy", "陈述"): "满意",
        ("sad", "反问"): "无奈",
        ("sad", "感叹"): "委屈",
        ("sad", "陈述"): "失落",
        ("angry", "反问"): "不耐烦",
        ("angry", "感叹"): "愤怒",
        ("angry", "疑问"): "困惑",
        ("neutral", "试探"): "试探",
        ("neutral", "疑问"): "好奇",
        ("neutral", "陈述"): "平静",
    }
    refined = refined_map.get((final, intent), EMO_LABELS_ZH.get(final, final))

    result = {
        "final_emotion": final,
        "final_emotion_zh": refined,
        "confidence": round(final_conf, 3),
        "conflict": conflict,
        "fusion_reason": fusion_reason,
        "acoustic_vote": acoustic_best,
        "textual_vote": textual_best,
        "intent": intent,
    }
    if baseline_best:
        result["baseline_vote"] = baseline_best
        result["baseline_confidence"] = round(baseline_conf, 3)
        result["baseline_signals"] = baseline_signals
        result["baseline_deviations"] = baseline.get("top_deviations", [])
    return result


# ============================================================
# Agent Hint 生成
# ============================================================

def build_agent_hint(acoustic: dict, textual: dict, fusion: dict) -> str:
    """
    生成给 agent 的情绪上下文提示。
    
    格式：结构化、简洁、可直接注入 prompt。
    """
    text = acoustic.get("text", "")
    final = fusion.get("final_emotion_zh", "未知")
    conf = fusion.get("confidence", 0)
    intent = fusion.get("intent", "陈述")
    conflict = fusion.get("conflict", False)

    # 声学概率分布（简写）
    probs = acoustic.get("probs", {})
    prob_str = " | ".join(
        f"{v['label_zh']}:{v['prob_pct']}" for v in probs.values()
    )

    hint_parts = [
        f"[语音情绪融合分析]",
        f"最终判断={final}(置信{conf:.0%})",
        f"语用意图={intent}",
        f"声学分布=[{prob_str}]",
        f"文本语义={textual.get('semantic_emotion_zh', '?')}({textual.get('confidence', 0):.0%})",
    ]

    # 基线偏差信息
    if fusion.get("baseline_vote"):
        hint_parts.append(f"基线偏差={fusion['baseline_vote']}({fusion.get('baseline_confidence', 0):.0%})")
        if fusion.get("baseline_deviations"):
            hint_parts.append(f"偏离={', '.join(fusion['baseline_deviations'][:2])}")

    if conflict:
        hint_parts.append(f"⚠️ 声学/文本冲突：{'; '.join(fusion.get('fusion_reason', []))}")

    if textual.get("signals"):
        hint_parts.append(f"文本信号={', '.join(textual['signals'][:3])}")

    hint_parts.append(f"转写=\"{text}\"")

    return " | ".join(hint_parts)


# ============================================================
# 主函数
# ============================================================

def analyze(audio_path: str, use_llm: bool = False, speaker: str = None) -> dict:
    """
    完整的多模态情绪分析流程。
    
    双引擎架构：
    - Whisper: 精准 ASR 转写
    - SenseVoice: 声学情绪概率提取

    三路融合（v4.0）：
    - 基线偏差通道：如有 speaker 参数且基线存在，自动加入个人校准
    """
    wav_path = None
    try:
        # 1. 转换音频格式
        wav_path = convert_audio(audio_path)

        # 2. 声学通道：SenseVoice 提取情绪概率分布
        acoustic = extract_acoustic_emotion(wav_path)

        # 3. ASR 转写：Whisper 精准转写
        whisper_text = whisper_transcribe(audio_path)
        acoustic["text"] = whisper_text  # 用 Whisper 的转写替代 SenseVoice 的

        # 4. 文本通道：用 Whisper 转写做语义情绪分析
        textual = analyze_text_emotion(whisper_text)

        # 4.5 基线偏差通道（v4.0）
        baseline = None
        if speaker:
            baseline = calibrate_baseline(audio_path, speaker)

        # 5. 融合（三路）
        fusion = fuse_emotion(acoustic, textual, baseline)

        # 6. 生成 agent hint
        hint = build_agent_hint(acoustic, textual, fusion)

        result = {
            "acoustic": {
                "raw_emotion": acoustic["raw_emotion"],
                "best_emotion": acoustic["best_emotion"],
                "best_prob": acoustic["best_prob"],
                "probs": acoustic["probs"],
            },
            "asr": {
                "whisper": whisper_text,
                "sensevoice": acoustic.get("sv_text", ""),
            },
            "textual": textual,
            "fusion": fusion,
            "text": whisper_text,
            "agent_hint": hint,
        }

        if baseline:
            result["baseline"] = {
                "speaker": baseline.get("speaker", ""),
                "top_deviations": baseline.get("top_deviations", []),
                "features": {
                    k: {"raw": v["raw"], "z_score": v["z_score"]}
                    for k, v in baseline.get("features", {}).items()
                    if v.get("z_score") is not None
                },
            }

        return result

    finally:
        if wav_path and wav_path != audio_path and os.path.exists(wav_path):
            os.unlink(wav_path)


def main():
    if len(sys.argv) < 2:
        print("用法: python3 sensevoice_analyze.py <audio_file> [--json] [--speaker <id>]")
        sys.exit(1)

    audio_path = sys.argv[1]
    json_mode = "--json" in sys.argv
    speaker = None
    if "--speaker" in sys.argv:
        idx = sys.argv.index("--speaker")
        if idx + 1 < len(sys.argv):
            speaker = sys.argv[idx + 1]

    if not os.path.exists(audio_path):
        print(json.dumps({"error": f"File not found: {audio_path}"}))
        sys.exit(1)

    result = analyze(audio_path, speaker=speaker)

    if json_mode:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # 人类友好输出
        print("=" * 60)
        print("🔊 声学通道 (SenseVoice CTC)")
        print("=" * 60)
        ac = result["acoustic"]
        print(f"  原始标签: {ac['raw_emotion']}")
        print(f"  概率最优: {ac['best_emotion']} ({ac['best_prob']:.1%})")
        print(f"  概率分布:")
        for label, info in ac["probs"].items():
            bar = "█" * int(info["prob"] * 40)
            print(f"    {info['label_zh']:4s} {info['prob_pct']:>6s} {bar}")

        print()
        print("=" * 60)
        print("📝 文本通道 (语义分析)")
        print("=" * 60)
        tx = result["textual"]
        print(f"  语义情绪: {tx['semantic_emotion_zh']} ({tx['confidence']:.0%})")
        print(f"  语用意图: {tx['intent']}")
        print(f"  概率分布: {tx['probs']}")
        if tx["signals"]:
            print(f"  信号:")
            for s in tx["signals"]:
                print(f"    • {s}")

        print()
        print("=" * 60)
        print("🔗 融合层")
        print("=" * 60)
        fu = result["fusion"]
        print(f"  最终情绪: {fu['final_emotion_zh']} ({fu['confidence']:.0%})")
        print(f"  声学投票: {fu['acoustic_vote']}")
        print(f"  文本投票: {fu['textual_vote']}")
        if fu.get("baseline_vote"):
            print(f"  基线投票: {fu['baseline_vote']} ({fu.get('baseline_confidence', 0):.0%})")
        print(f"  冲突: {'⚠️ 是' if fu['conflict'] else '否'}")
        print(f"  融合理由:")
        for r in fu["fusion_reason"]:
            print(f"    → {r}")

        if result.get("baseline"):
            print()
            print("=" * 60)
            print("📏 基线偏差通道")
            print("=" * 60)
            bl = result["baseline"]
            print(f"  说话人: {bl['speaker']}")
            if bl.get("top_deviations"):
                print(f"  主要偏离:")
                for d in bl["top_deviations"]:
                    print(f"    • {d}")
            for feat, val in bl.get("features", {}).items():
                print(f"    {feat}: raw={val['raw']:.4f} z={val['z_score']:+.2f}σ")

        print()
        print("=" * 60)
        print("💬 Agent Hint")
        print("=" * 60)
        print(f"  {result['agent_hint']}")

        print()
        print("=" * 60)
        print("📄 转写文本")
        print("=" * 60)
        print(f"  \"{result['text']}\"")


if __name__ == "__main__":
    main()