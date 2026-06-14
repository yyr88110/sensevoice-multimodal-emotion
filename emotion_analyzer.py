#!/usr/bin/env python3
"""
情绪离线分析 v2.0 — 文本+语音双通道融合

定时运行，分析 partner agent 的聊天记录，检测情绪趋势。

双通道：
  1. 文本通道：关键词情绪打分（原有逻辑）
  2. 声学通道：SenseVoice + 说话人基线校准（v2.0 新增）

融合策略：
  - 有语音消息时：声学通道权重 0.6，文本通道 0.4
  - 只有文本消息时：纯文本打分
  - 声学通道依赖 sensevoice_analyze.py + speaker_baseline.py
"""

import json
import sqlite3
import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# 配置
# ============================================================
PARTNER_STATE_DB = os.path.expanduser("~/.hermes/profiles/partner/state.db")
EMOTION_LOG = os.path.expanduser("~/.hermes/profiles/partner/emotion_log.jsonl")
AUDIO_CACHE = os.path.expanduser("~/.hermes/audio_cache")
ALERT_THRESHOLD = 3       # 连续低分次数阈值
LOW_SCORE_THRESHOLD = 5   # 低分阈值
COOLDOWN_HOURS = 6        # 冷却期（小时）
SPEAKER_ID = "shiwei"     # 石维的说话人 ID
SENSEVOICE_SCRIPT = os.path.expanduser("~/.hermes/scripts/sensevoice_analyze.py")

# ============================================================
# 情绪词库
# ============================================================
NEGATIVE_KEYWORDS = {
    1: ["想哭", "崩溃", "心塞", "受不了", "太难了", "活不下去了", "救命", "不想活了", "撑不住", "绝望", "没希望"],
    2: ["烦死了", "好累啊", "不想动", "emo", "无语", "服了", "累了", "心烦", "头疼", "不想说话", "烦透了", "累死了", "精疲力尽"],
    3: ["好难啊", "郁闷", "委屈", "难受", "不舒服", "好烦", "有点累", "心情不好", "丧丧的", "不开心", "有点难过", "心里堵得慌"],
    4: ["还好吧", "一般般", "凑合", "就那样", "无所谓", "随便吧", "行吧", "也就那样吧", "没什么感觉", "马马虎虎"]
}

POSITIVE_KEYWORDS = {
    7: ["开心", "不错", "厉害", "好喜欢", "好可爱", "好甜", "好暖", "舒服", "满足", "挺开心的", "心情不错"],
    8: ["太棒了", "好开心", "感动", "幸福", "爽", "绝了", "爱了", "好赞", "超喜欢", "太好了", "太开心了"],
    9: ["哈哈哈哈", "笑死", "太厉害了吧", "天哪", "好惊喜", "不敢相信", "绝绝子", "太绝了", "我天"],
    10: ["太幸福了", "人生巅峰", "完美", "圆满", "值了", "死而无憾", "这辈子值了", "无与伦比"]
}

NEUTRAL_KEYWORDS = {
    5: ["好的", "嗯", "知道了", "继续", "行", "可以", "都行", "你定", "随便", "嗯嗯", "好呀"],
    6: ["还不错", "挺好的", "可以啊", "也行", "好的呀", "还挺有意思的", "有点意思"]
}

SPECIAL_KEYWORDS = {
    "身体不适": ["好饿", "好困", "好冷", "好热", "肚子疼", "头疼", "想睡觉", "身体不舒服", "有点晕", "恶心"],
    "工作压力": ["加班", "deadline", "做不完", "好忙", "焦头烂额", "工作好多", "压力好大", "忙死了"],
    "家庭相关": ["想孩子", "担心", "操心", "累死了", "孩子又闹了", "带娃好累", "宝宝不听话"]
}

# SenseVoice 情绪 → 1-10 分映射
ACOUSTIC_EMOTION_SCORE = {
    "happy": 8,
    "neutral": 5,
    "sad": 3,
    "angry": 2,
    "unk": 5,
}

# 情绪置信度 → 分数微调幅度
CONFIDENCE_ADJUSTMENT = 2  # 高置信度时 ±2 分


# ============================================================
# 文本通道：关键词情绪打分
# ============================================================

def score_text_message(content):
    """对文本消息进行关键词情绪打分"""
    if not content:
        return None

    for score, keywords in NEGATIVE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in content:
                return score

    for score, keywords in POSITIVE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in content:
                return score

    for score, keywords in NEUTRAL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in content:
                return score

    return 5  # 默认中性


# ============================================================
# 声学通道：SenseVoice + 基线校准
# ============================================================

def extract_audio_path(content):
    """从消息中提取音频文件路径"""
    import re
    # 匹配 [User sent audio: path] 或 [The user sent an audio file attachment: path]
    patterns = [
        r'\[User sent audio:\s*([^\]]+)\]',
        r"saved at:\s*(/[^\s\)]+\.ogg)",
        r'(\/Users\/[^\s\)]+\.(?:ogg|wav|mp3))',
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            path = match.group(1).strip()
            if os.path.exists(path):
                return path
    return None


def analyze_voice_emotion(audio_path):
    """用 SenseVoice + 石维基线分析语音情绪"""
    try:
        cmd = [
            "python3", SENSEVOICE_SCRIPT,
            audio_path,
            "--speaker", SPEAKER_ID,
            "--json"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"  SenseVoice error: {result.stderr[:200]}", file=sys.stderr)
            return None

        output = result.stdout.strip()
        # 跳过非 JSON 输出（funasr 日志）
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                data = json.loads(line)
                break
        else:
            # 尝试找 JSON 块
            import re
            json_match = re.search(r'\{[\s\S]*\}', output)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return None

        fusion = data.get("fusion", {})
        emotion = fusion.get("final_emotion", "neutral")
        confidence = fusion.get("confidence", 0.5)
        baseline_vote = fusion.get("baseline_vote")
        baseline_deviations = fusion.get("baseline_deviations", [])

        # 转换为 1-10 分
        base_score = ACOUSTIC_EMOTION_SCORE.get(emotion, 5)
        # 高置信度时调整分数
        if confidence > 0.7:
            if base_score <= 4:
                base_score -= int(CONFIDENCE_ADJUSTMENT * (confidence - 0.5))
            elif base_score >= 6:
                base_score += int(CONFIDENCE_ADJUSTMENT * (confidence - 0.5))
        base_score = max(1, min(10, base_score))

        return {
            "score": base_score,
            "emotion": emotion,
            "emotion_zh": fusion.get("final_emotion_zh", emotion),
            "confidence": confidence,
            "baseline_vote": baseline_vote,
            "baseline_deviations": baseline_deviations,
            "text": data.get("text", ""),
            "acoustic_prob": data.get("acoustic", {}).get("best_prob", 0),
        }
    except subprocess.TimeoutExpired:
        print(f"  SenseVoice timeout for {audio_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Voice analysis error: {e}", file=sys.stderr)
        return None


# ============================================================
# 消息获取
# ============================================================

def get_recent_messages(hours=24):
    """获取最近 N 小时的消息"""
    try:
        conn = sqlite3.connect(PARTNER_STATE_DB)
        cursor = conn.cursor()

        cutoff = datetime.now() - timedelta(hours=hours)
        cutoff_ts = int(cutoff.timestamp())

        cursor.execute("""
            SELECT timestamp, role, content
            FROM messages
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (cutoff_ts,))

        messages = []
        for row in cursor.fetchall():
            messages.append({
                "timestamp": row[0],
                "role": row[1],
                "content": row[2]
            })

        conn.close()
        return messages
    except Exception as e:
        print(f"Error reading state.db: {e}")
        return []


# ============================================================
# 双通道融合分析
# ============================================================

def analyze_emotion_trend(messages):
    """分析情绪趋势（双通道融合）"""
    user_messages = [m for m in messages if m.get("role") == "user"]

    if not user_messages:
        return {
            "total_messages": 0,
            "avg_score": None,
            "low_score_count": 0,
            "consecutive_low": 0,
            "should_alert": False,
            "trend": "unknown",
            "voice_count": 0,
            "text_count": 0,
        }

    scores = []
    voice_analyses = []

    for msg in user_messages:
        content = msg.get("content", "")
        if not content:
            continue

        # 检查是否是语音消息
        audio_path = extract_audio_path(content)

        if audio_path and os.path.exists(audio_path):
            # 声学通道
            print(f"  🎤 语音消息: {os.path.basename(audio_path)}")
            voice_result = analyze_voice_emotion(audio_path)
            if voice_result:
                # 有语音分析结果：声学 0.6 + 文本 0.4
                text_score = score_text_message(content)
                if text_score is not None:
                    fused_score = round(voice_result["score"] * 0.6 + text_score * 0.4)
                else:
                    fused_score = voice_result["score"]

                scores.append(max(1, min(10, fused_score)))
                voice_analyses.append(voice_result)
                print(f"    声学={voice_result['emotion_zh']}({voice_result['confidence']:.0%}) "
                      f"基线={voice_result.get('baseline_vote', '?')} "
                      f"→ 分数={fused_score}")
            else:
                # 声学分析失败，降级为纯文本
                text_score = score_text_message(content)
                if text_score is not None:
                    scores.append(text_score)
        else:
            # 纯文本消息
            text_score = score_text_message(content)
            if text_score is not None:
                scores.append(text_score)

    if not scores:
        return {
            "total_messages": len(user_messages),
            "avg_score": None,
            "low_score_count": 0,
            "consecutive_low": 0,
            "should_alert": False,
            "trend": "unknown",
            "voice_count": len(voice_analyses),
            "text_count": 0,
        }

    # 统计
    avg_score = sum(scores) / len(scores)
    low_score_count = sum(1 for s in scores if s <= LOW_SCORE_THRESHOLD)

    consecutive_low = 0
    for score in reversed(scores):
        if score <= LOW_SCORE_THRESHOLD:
            consecutive_low += 1
        else:
            break

    should_alert = consecutive_low >= ALERT_THRESHOLD

    if len(scores) >= 3:
        recent_3 = scores[-3:]
        if all(s <= LOW_SCORE_THRESHOLD for s in recent_3):
            trend = "持续低落"
        elif recent_3[-1] > recent_3[0]:
            trend = "回升"
        elif recent_3[-1] < recent_3[0]:
            trend = "下降"
        else:
            trend = "平稳"
    else:
        trend = "数据不足"

    result = {
        "total_messages": len(user_messages),
        "avg_score": round(avg_score, 2),
        "low_score_count": low_score_count,
        "consecutive_low": consecutive_low,
        "should_alert": should_alert,
        "trend": trend,
        "scores": scores,
        "voice_count": len(voice_analyses),
        "text_count": len(user_messages) - len(voice_analyses),
    }

    # 如果有语音分析结果，附加摘要
    if voice_analyses:
        voice_emotions = [v["emotion"] for v in voice_analyses]
        from collections import Counter
        emotion_dist = Counter(voice_emotions)
        result["voice_emotion_distribution"] = dict(emotion_dist)
        # 基线偏离摘要
        all_deviations = []
        for v in voice_analyses:
            all_deviations.extend(v.get("baseline_deviations", []))
        if all_deviations:
            result["baseline_deviations"] = list(set(all_deviations))[:3]

    return result


# ============================================================
# 冷却期 + 日志 + 告警
# ============================================================

def check_cooldown():
    """检查冷却期"""
    if not os.path.exists(EMOTION_LOG):
        return False

    try:
        with open(EMOTION_LOG, "r") as f:
            lines = f.readlines()

        if not lines:
            return False

        last_line = lines[-1].strip()
        if not last_line:
            return False

        last_record = json.loads(last_line)
        last_alert_time = datetime.fromisoformat(last_record.get("timestamp", "2000-01-01T00:00:00"))

        cooldown_until = last_alert_time + timedelta(hours=COOLDOWN_HOURS)
        return datetime.now() < cooldown_until
    except Exception as e:
        print(f"Error checking cooldown: {e}")
        return False


def log_emotion(analysis, triggered=False):
    """记录情绪分析结果"""
    record = {
        "timestamp": datetime.now().isoformat(),
        "avg_score": analysis.get("avg_score"),
        "low_score_count": analysis.get("low_score_count", 0),
        "consecutive_low": analysis.get("consecutive_low", 0),
        "trend": analysis.get("trend", "unknown"),
        "voice_count": analysis.get("voice_count", 0),
        "text_count": analysis.get("text_count", 0),
        "triggered_care": triggered,
    }

    # 附加声学通道数据
    if analysis.get("voice_emotion_distribution"):
        record["voice_emotions"] = analysis["voice_emotion_distribution"]
    if analysis.get("baseline_deviations"):
        record["baseline_deviations"] = analysis["baseline_deviations"]

    try:
        with open(EMOTION_LOG, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error writing emotion log: {e}")


def generate_alert_message(analysis):
    """生成告警消息"""
    avg_score = analysis.get("avg_score", 5)
    trend = analysis.get("trend", "未知")
    consecutive_low = analysis.get("consecutive_low", 0)
    voice_count = analysis.get("voice_count", 0)

    parts = [
        f"最近24小时情绪分析：",
        f"- 平均分数：{avg_score}/10",
        f"- 情绪趋势：{trend}",
        f"- 连续低分次数：{consecutive_low}",
    ]

    if voice_count > 0:
        parts.append(f"- 语音分析：{voice_count} 条")
        if analysis.get("voice_emotion_distribution"):
            dist = analysis["voice_emotion_distribution"]
            parts.append(f"  情绪分布: {dist}")
        if analysis.get("baseline_deviations"):
            parts.append(f"  基线偏离: {', '.join(analysis['baseline_deviations'])}")

    parts.append(f"建议：石维最近情绪可能不太好，可以适当关心一下。")
    return "\n".join(parts)


# ============================================================
# 主函数
# ============================================================

def main():
    print(f"[{datetime.now().isoformat()}] 开始情绪离线分析 v2.0...")

    messages = get_recent_messages(hours=24)
    print(f"获取到 {len(messages)} 条消息")

    if not messages:
        print("没有消息，跳过分析")
        return

    analysis = analyze_emotion_trend(messages)
    print(f"\n分析结果：{json.dumps(analysis, ensure_ascii=False, indent=2)}")

    if analysis.get("should_alert"):
        if check_cooldown():
            print("在冷却期内，跳过告警")
            log_emotion(analysis, triggered=False)
        else:
            print("触发情绪告警！")
            log_emotion(analysis, triggered=True)
            alert_message = generate_alert_message(analysis)
            print(f"ALERT:{alert_message}")
    else:
        print("情绪正常，无需告警")
        log_emotion(analysis, triggered=False)


if __name__ == "__main__":
    main()