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
from typing import Optional

# ============================================================
# SenseVoice 情绪 token 映射
# ============================================================
EMO_TOKEN_MAP = {
    25001: "happy",
    25002: "sad",
    25003: "angry",
    25004: "neutral",
    25009: "unk",
}
EMO_TOKEN_IDS = list(EMO_TOKEN_MAP.keys())
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
        from funasr import AutoModel
        _model = AutoModel(
            model="iic/SenseVoiceSmall",
            device="cpu",
            disable_update=True,
        )
    return _model




# ============================================================
# Whisper ASR 引擎
# ============================================================
_whisper_model = None

def get_whisper_model():
    """Lazy load Whisper model (singleton)"""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("small")
    return _whisper_model


def whisper_transcribe(audio_path: str) -> str:
    """用 Whisper 做精准 ASR 转写"""
    model = get_whisper_model()
    result = model.transcribe(audio_path, language="zh")
    return result["text"].strip()

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


# ============================================================
# 融合层：声学 + 文本 → 最终判断
# ============================================================

def fuse_emotion(acoustic: dict, textual: dict) -> dict:
    """
    多模态融合决策：
    
    1. 声学提供"情绪基调"（大方向：开心/难过/生气/平静）
    2. 文本提供"语义细节"（具体意图和细微差别）
    3. 融合策略：
       - 一致 → 高置信度，直接采信
       - 冲突 → 分析冲突原因，给出融合判断
       - 文本信号弱 → 主要依赖声学
       - 声学信号弱（概率分散）→ 主要依赖文本
    """
    acoustic_best = acoustic.get("best_emotion", "neutral")
    acoustic_prob = acoustic.get("best_prob", 0.0)
    textual_best = textual.get("semantic_emotion", "neutral")
    textual_conf = textual.get("confidence", 0.0)
    intent = textual.get("intent", "陈述")

    fusion_reason = []
    conflict = False

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

    return {
        "final_emotion": final,
        "final_emotion_zh": refined,
        "confidence": round(final_conf, 3),
        "conflict": conflict,
        "fusion_reason": fusion_reason,
        "acoustic_vote": acoustic_best,
        "textual_vote": textual_best,
        "intent": intent,
    }


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

    if conflict:
        hint_parts.append(f"⚠️ 声学/文本冲突：{'; '.join(fusion.get('fusion_reason', []))}")

    if textual.get("signals"):
        hint_parts.append(f"文本信号={', '.join(textual['signals'][:3])}")

    hint_parts.append(f"转写=\"{text}\"")

    return " | ".join(hint_parts)


# ============================================================
# 主函数
# ============================================================

def analyze(audio_path: str, use_llm: bool = False) -> dict:
    """
    完整的多模态情绪分析流程。
    
    双引擎架构：
    - Whisper: 精准 ASR 转写
    - SenseVoice: 声学情绪概率提取
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

        # 5. 融合
        fusion = fuse_emotion(acoustic, textual)

        # 6. 生成 agent hint
        hint = build_agent_hint(acoustic, textual, fusion)

        return {
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

    finally:
        if wav_path and wav_path != audio_path and os.path.exists(wav_path):
            os.unlink(wav_path)


def main():
    if len(sys.argv) < 2:
        print("用法: python3 multimodal_emotion.py <audio_file> [--json]")
        sys.exit(1)

    audio_path = sys.argv[1]
    json_mode = "--json" in sys.argv

    if not os.path.exists(audio_path):
        print(json.dumps({"error": f"File not found: {audio_path}"}))
        sys.exit(1)

    result = analyze(audio_path)

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
        print(f"  冲突: {'⚠️ 是' if fu['conflict'] else '否'}")
        print(f"  融合理由:")
        for r in fu["fusion_reason"]:
            print(f"    → {r}")

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
