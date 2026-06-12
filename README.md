# SenseVoice 多模态情绪融合分析系统

基于 Whisper + SenseVoice 双引擎的声学+文本情绪分析系统，用于语音消息的实时情绪识别。

## 架构 (v3.0)

```
Whisper (ASR)              SenseVoice (情绪)
       ↓                           ↓
  精准转写文本                CTC logits → 5类概率
       ↓                           ↓
  文本通道 (语义分析)         声学通道
       ↓                           ↓
       └──────── 融合层 ───────────┘
                 ↓
         最终情绪判断 + 融合理由
```

### 双引擎设计
- **Whisper (small)**: 负责 ASR 转写，中文准确率高
- **SenseVoice**: 负责声学情绪识别，从 CTC logits 提取概率分布

### 声学通道 (SenseVoice)
从 CTC logits 的位置2提取情绪 token 概率分布，输出5类情绪（happy/sad/angry/neutral/unk）的真实概率，而非单一标签。

### 文本通道 (Whisper + 语义分析)
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

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 克隆仓库
git clone https://github.com/yyr88110/sensevoice-multimodal-emotion.git
cd sensevoice-multimodal-emotion

# 运行分析
python sensevoice_analyze.py your_audio.ogg
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
| `asr` | ASR 对比：Whisper vs SenseVoice 转写 |
| `textual` | 文本通道：语义情绪 + 语用意图 + 信号列表 |
| `fusion` | 融合层：最终情绪 + 置信度 + 融合理由 |
| `text` | 最终使用的转写文本（Whisper） |
| `agent_hint` | 给 AI agent 的上下文提示 |

## ASR 对比示例

```
语音: "不错不错，这个逻辑还是很合理的那你就把这个上线吧"

Whisper:     不错不错,这个逻辑还是很合理的那就把这个摄像吧然后给partner的agent也附上对应的技能
SenseVoice:  不错不错，这个逻辑还是很合理的那你就把这个上线吧，然后给partent也附上对应的金能

分析: Whisper 更准 (partner vs partent, 能力 vs 金能)
      SenseVoice 更准 (上线 vs 摄像)
```

## 示例输出

查看 `examples/` 目录下的完整示例：

| 文件 | 场景 | 融合结果 |
|---|---|---|
| `probing_tone.json` | "那你觉得我现在说话是什么情绪呢？" | 试探(75%) |
| `frustrated_tone.json` | "哎，不是我这还要这么假装笑吗？" | 无奈(62%) |
| `satisfied_tone.json` | "不错不错，这个逻辑还是很合理的" | 满意(72%) |

## 测试结果

| 语音 | 声学概率 | 文本意图 | 融合结果 |
|---|---|---|---|
| "你听一下我现在什么态度呢？" | sad 44.8% vs neutral 43.4% | 疑问 | 好奇(60%) |
| "哎，不是我这还要这么假装笑吗？" | sad 70.8% | 反问 | 无奈(62%) |
| "那你觉得我现在说话是什么情绪呢？" | neutral 58.9% | 试探 | 试探(75%) |
| "不错不错，这个逻辑还是很合理的" | neutral 99.3% | 陈述 | 满意(72%) |

## 技术细节

### 情绪 Token ID
SenseVoice 使用 CTC 解码，情绪 token 位于序列位置2：
- 25001: happy
- 25002: sad
- 25003: angry
- 25004: neutral
- 25009: unk (unknown)

### 概率提取
从 CTC logits 的位置2提取上述5个 token 的 log 概率，再通过 softmax 得到归一化概率分布。

### 为什么用双引擎？
- **SenseVoice ASR** 不够准（"partent" vs "partner"，"金能" vs "能力"）
- **Whisper ASR** 更准但没有情绪识别
- **双引擎各取所长**：Whisper 管转写，SenseVoice 管情绪

## 已知限制

- Whisper small 模型在 CPU 上较慢（~30秒/条语音）
- 5类情绪粒度较粗（后续可通过微调扩展到20-30类）
- 文本分析基于规则（后续可用 LLM 替代）

## 后续规划

1. **Whisper 量化**：用 whisper.cpp 或 faster-whisper 加速
2. **LLM 文本分析**：用 LLM 替代规则做更精确的语义分析
3. **情绪类别扩展**：通过微调扩展到20-30类
4. **流式处理**：支持实时语音流输入
5. **多语言支持**：扩展到英语、粤语等

## License

MIT
