# SenseVoice 多模态情绪融合分析系统

基于阿里 SenseVoice 模型的声学+文本双通道情绪分析系统，用于语音消息的实时情绪识别。

## 架构

```
声学通道 (SenseVoice CTC)     文本通道 (语义分析)
       ↓                           ↓
  CTC logits → 5类概率        转写文本 → 细粒度情绪
       ↓                           ↓
       └──────── 融合层 ───────────┘
                 ↓
         最终情绪判断 + 融合理由
```

### 声学通道
从 SenseVoice CTC logits 的位置2提取情绪 token 概率分布，输出5类情绪（happy/sad/angry/neutral/unk）的真实概率，而非单一标签。

### 文本通道
- 语义情绪词匹配（开心/难过/生气词库）
- 语用意图分析（反问/试探/疑问/感叹/陈述）
- 句式结构分析（感叹号、问号、否定叠加）
- 语气词检测（呢/吧/啊/嘛/嗯/哦）
- 元认知词识别（假装/装/演/故意）

### 融合层（6种策略）
1. **声学+文本一致** → 高置信度，直接采信
2. **声学高置信+文本弱** → 采信声学（语气比文字诚实）
3. **文本高置信+声学弱** → 采信文本（语义比声学精确）
4. **声学neutral+文本有情绪** → 倾向文本（表面平静但话里有话）
5. **文本neutral+声学有情绪** → 倾向声学（文字平淡但语气明显）
6. **真正冲突** → 看语用意图（试探→中性，反问→文本优先）

## 安装

```bash
pip install funasr torch torchaudio
git clone https://github.com/yyr88110/sensevoice-multimodal-emotion.git
cd sensevoice-multimodal-emotion
```

## 使用

```bash
# 人类友好输出
python sensevoice_analyze.py test.ogg

# JSON 输出（适合程序调用）
python sensevoice_analyze.py test.ogg --json
```

## 输出结构

| 字段 | 说明 |
|---|---|
| `acoustic` | 声学通道：5类情绪概率分布 |
| `textual` | 文本通道：语义情绪 + 语用意图 + 信号列表 |
| `fusion` | 融合层：最终情绪 + 置信度 + 融合理由 |
| `text` | ASR 转写文本 |
| `agent_hint` | 给 AI agent 的上下文提示 |

## 测试结果

| 语音 | 声学概率 | 文本意图 | 融合结果 |
|---|---|---|---|
| "你听一下我现在什么态度呢？" | sad 44.8% vs neutral 43.4% | 疑问 | 好奇(60%) |
| "哎，不是我这还要这么假装笑吗？" | sad 70.8% | 反问 | 无奈(62%) |
| "那你觉得我现在说话是什么情绪呢？" | neutral 58.9% | 试探 | 试探(75%) |
| "不错不错，这个逻辑还是很合理的" | neutral 99.3% | 陈述 | 满意(72%) |

## 已知限制

- ASR 转写精度有限（SenseVoice 主要优化情绪识别，ASR 是附带能力）
- 5类情绪粒度较粗（后续可通过微调扩展到20-30类）
- 文本分析基于规则（后续可用 LLM 替代）

## License

MIT
