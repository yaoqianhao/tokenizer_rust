---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_92f20f88c9d211f18e0e52540064ee0f
    ReservedCode1: Oniy1H5PcyoxzC1iMEUphaYNGigKGPFZlXusXDBDs4mZAh9hfSP1tvDNaz/MwQmw33xsGALU5+FJeBdvqF8PCEfwKVE2V5xaDCB55QqQWW8+e+Fny0one3OhqjiqfmYEuwbJu0NBE32DZ6fsRWRLA0UTCLymRI0cKv7a7iSK+iF3i0vDYqG1ahJQ9rY=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_92f20f88c9d211f18e0e52540064ee0f
    ReservedCode2: Oniy1H5PcyoxzC1iMEUphaYNGigKGPFZlXusXDBDs4mZAh9hfSP1tvDNaz/MwQmw33xsGALU5+FJeBdvqF8PCEfwKVE2V5xaDCB55QqQWW8+e+Fny0one3OhqjiqfmYEuwbJu0NBE32DZ6fsRWRLA0UTCLymRI0cKv7a7iSK+iF3i0vDYqG1ahJQ9rY=
---

# Tokenizer 完全指南：从零到选型

> 文档类型：技术学习手册
> 整理日期：2026-10-17
> 适用对象：从零开始学习 LLM 分词器，需要建立完整体系并做技术选型的读者

---

## 目录

- [0. 一句话定位](#0-一句话定位)
- [1. 为什么必须存在（取舍三角）](#1-为什么必须存在取舍三角)
- [2. 统一流水线：六层骨架与模块拆解](#2-统一流水线六层骨架与模块拆解)
- [3. 第 3 层算法库：子词模型逐个精讲](#3-第-3-层算法库子词模型逐个精讲)
- [4. 横向对比总表](#4-横向对比总表)
- [5. 选型与工程实践](#5-选型与工程实践)
- [6. 现代趋势](#6-现代趋势)
- [7. 从零学习路径（建议顺序）](#7-从零学习路径建议顺序)
- [参考来源](#参考来源)

---

## 0. 一句话定位

Tokenizer（分词器）是 LLM 与人类文本之间的**编解码层**：把原始字符串转换成模型能吃的整数 ID 序列（encode），再把 ID 序列还原成字符串（decode）。它**不参与训练、不属于模型参数**，但决定了模型"看到世界的粒度"，其影响贯穿训练成本、推理成本、多语言公平性、甚至算术能力。

---

## 1. 为什么必须存在（取舍三角）

神经网络只接受定长数值输入，所以需要一个"词典 + 切分规则"。可选粒度只有三种，构成一个不可兼得三角：

| 粒度 | 词表规模 | 序列长度 | 致命问题 |
|---|---|---|---|
| 词级 Word | 百万级（爆炸） | 最短 | 未登录词 OOV 无解；形态变化（run/running/ran）各自独立，参数浪费 |
| 字符级 Char | ~256 或数千 | 极长（4~6 倍） | 语义单元太碎，注意力被稀释，训练慢 |
| **子词级 Subword** | 3 万 ~ 26 万 | 折中 | 主流方案，也是整篇文档的主体 |

子词的核心假设：**高频词保持完整，低频词拆成有意义的片段**（如 `unhappiness` → `un` + `happi` + `ness`）。所有子词算法都是这一假设的不同实现。

> 现代补充：大词表趋势下，子词粒度还在往"更大单元"漂移（见 §6），理论上限是彻底取消 tokenizer。

---

## 2. 统一流水线：六层骨架与模块拆解

**本文档只有一条流水线。** 无论 BPE、WordPiece、Unigram、SentencePiece 还是 tiktoken，都只是"同一条流水线 + 一份不同的配置"。所谓"不同算法"，分歧只出现在**第 3 层**；其余各层都是填空式的开关。因此本指南不把各方案列为彼此独立的体系，而是统一用下面这条骨架描述。

一次完整的 `tokenizer("Hello, 世界!")` 调用要依次穿过下面 6 个核心层（HuggingFace `tokenizers` 的架构正是按此切分的），另外 4 项是外围能力。每层都有多种实现可选：

```
input 字符串
 → 1 规范化 Normalization
 → 2 预分词 Pre-tokenization
 → 3 子词切分 Model        ← 唯一真正的"算法分歧层"
 → 4 词表与 ID 空间
 → 5 后处理 Post-processing
 → 6 解码 Decoding
 ┈ 外围：7 对齐/偏移 · 8 截断/填充 · 9 实现与性能 · 10 多模态扩展
```

各主流方案在这条流水线上的逐层取值，见 §2.11 的统一视图。

### 2.1 规范化 Normalization

把"表面形式各异但语义相同"的文本收敛到统一形式。

| 选项 | 作用 | 例子 |
|---|---|---|
| NFC | 组合字符合成 | `e` + `´` → `é` |
| NFD | 分解 | `é` → `e` + `´` |
| **NFKC** | 兼容分解再合成（最常见） | `Ａ`→`A`、`ﬁ`→`fi`、`²`→`2`、`①`→`1` |
| NFKD | 兼容分解 | |
| Lowercase | 统一小写 | `The` → `the` |
| Casefold | 更激进的大小写折叠 | 德语 `ß` → `ss` |
| StripAccents | 去音标 | `café` → `cafe` |
| Strip / Replace | 去首尾空白 / 正则替换 | 多空格合一 |
| **Identity** | 不做任何处理 | 现代趋势，保留一切原始信息 |

**关键权衡**：规范化越多，词表越紧凑、压缩率越高，但**不可逆**（`ｆｉ`→`fi` 后无法还原），且会抹掉对模型有用的信息（大小写、全角、真实空格）。GPT-2 之后的主流做法是**尽量不做有损规范化**，改用字节级兜底。

### 2.2 预分词 Pre-tokenization

在子词切分之前，先把文本切成"词级"的粗块。

| 算法 | 规则 | 代表 |
|---|---|---|
| Whitespace / WhitespaceSplit | 按空白切 | 简单场景 |
| Punctuation | 标点独立成块 | |
| **BertPreTokenizer** | 空白切 + 标点隔离 | BERT |
| **ByteLevel（GPT-2 regex）** | 用正则把缩写、字母、数字、标点分别捕获 | GPT-2/3/4、LLaMA 3 |
| **Metaspace** | 空格替换为 `▁` 并切分 | SentencePiece / T5 |
| Digits | 每个数字单独切 | Gemma/Gemini（利于算术） |
| Split(regex) | 自定义正则 | 领域定制 |
| 无预分词 | 直接喂原始 UTF-8 流 | 字节级方案 |

GPT-2 的正则模式是关键细节，它把 `'s`、`don`+`'t` 这类缩写单独成块：

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

### 2.3 子词切分（核心 Model）

**整条流水线中唯一真正分歧的层。** 见 §3 精讲。选项：`BPE` / `ByteLevel-BPE` / `WordPiece` / `Unigram` / `WordLevel` / `CharLevel`。

必须强调：切换这一层的实现，会连带改变第 2 层的预分词方式、第 4 层的词表形式与第 6 层的解码规则（见 §2.11 的配置档案表）。工程上不存在"只换子词算法"的孤立替换。

### 2.4 词表与 ID 空间

- **词表构成**：基础字符或字节（256 个）+ 合并出的子词 + 特殊 token。
- **特殊 token 算法选择**：
  - 传统固定集：`[UNK]` `[CLS]` `[SEP]` `[PAD]` `[MASK]` `[BOS]` `[EOS]`
  - 现代角色化：`<|im_start|>` `<|im_end|>`、工具调用 token、`<|endoftext|>`
  - 字节兜底：`<0x00>`~`<0xFF>`
- **词表大小**：直接决定嵌入参数量 `vocab_size × d_model`（通常输入嵌入与输出头共享权重，即 tying）。是"参数预算 vs 序列长度"的分配问题。

### 2.5 后处理 Post-processing

把序列包装成模型需要的样子。

| 算法 | 效果 |
|---|---|
| TemplateProcessing | 通用模板，现代首选 |
| BertProcessing | 自动加 `[CLS]` / `[SEP]` |
| RobertaProcessing | 加 `<s>` / `</s>` |
| Sequence | 把多个序列拼接 |

### 2.6 解码 Decoding

把子词还原成文本，不同算法各有"还原规则"。

| 解码器 | 规则 |
|---|---|
| ByteLevel | 字节序列按 UTF-8 拼回 |
| WordPiece | 去掉词内前缀 `##` 并拼接 |
| Metaspace | `▁` 还原为空格 |
| BPEDecoder | 去掉词尾标记 `</w>` |
| Fuse | 简单拼接 |
| CTC | 语音场景去重复帧 |
| Sequence | 串联多个解码器 |

配套参数：`skip_special_tokens`、`clean_up_tokenization_spaces`。

### 2.7 对齐与偏移 Offsets / Alignment

输出每个 token 对应的**原文字符区间**，用于 NER 高亮、幻觉溯源、attention 可视化。

- **算法选择**：启发式字符计数（快但遇规范化会错位） vs **Needleman–Wunsch 序列对齐**（HuggingFace fast tokenizer 采用，跨规范化仍然准确）。

### 2.8 批处理控制 Truncation / Padding

| 能力 | 选项 |
|---|---|
| 截断 | `longest_first` / `only_first` / `only_second` / 头尾保留 / 从右/从左 |
| 填充 | `longest`（批次内对齐）/ 固定 `max_length` / `do_not_pad` |
| 附加张量 | `attention_mask`、`token_type_ids`、`position_ids` |
| 训练期随机化 | BPE-dropout、Unigram 采样（≡ 子词正则化） |

### 2.9 实现与性能

- 纯 Python（HF `tokenizers` 旧版慢）→ **Rust 实现**（HF `tokenizers`，多线程 + 正则编译）→ **tiktoken**（OpenAI，Rust + 高度优化的 BPE）。
- 训练分词器约几分钟到几小时（取决于语料）；推理期分词可能占端到端延迟的 5~20%。

### 2.10 多模态扩展

"Tokenizer"一词已泛化到非文本：

- **音频**：Whisper 用多语言 BPE；EnCodec / SoundStream 用 RVQ 残差向量量化码本。
- **视觉**：VQ-VAE / VQ-GAN 码本、ViT patch 切分。
- **统一 token 空间**：图文音共用一个 ID 词表（Chameleon 等），本质是"把不同模态都离散化"。

### 2.11 统一视图：一条流水线，五份配置档案

把 §2.1–§2.6 各层的选项组合起来，就得到一份"配置档案"。在同一条流水线上，主流方案的真实差异如下：

| 流水线层 | BPE（RoBERTa 型） | WordPiece（BERT 型） | Unigram（T5/Gemma 型） | SentencePiece | tiktoken（GPT 型） | BLT（无分词器） |
|---|---|---|---|---|---|---|
| 1 规范化 | 恒等 | 恒等 | `nmt_nfkc`（可配） | 由 `normalization_rule_name` 决定 | **恒等**（无此层） | 恒等 |
| 2 预分词 | GPT-2 regex | BertPreTokenizer | 无（SP 直接切流） | 无（`▁` 转义空格） | GPT-2 regex（= `pat_str`） | 无（按复杂度动态切 patch） |
| 3 子词模型 | BPE（频次合并） | BPE + PMI 准则 | Unigram（EM + 剪枝） | 可插 BPE 或 Unigram | **字节级 BPE（rank 即合并优先级）** | 无词表，字节分组 |
| 4 词表/ID | 合并序 + special | 合并序 + `##` + special | 概率表 + special | 由内核决定 + `byte_fallback` | `mergeable_ranks` + `special` | 动态，无固定 ID 空间 |
| 5 后处理 | Template / Roberta | BertProcessing | TemplateProcessing | TemplateProcessing | **无实现**（调用方自行拼接） | 无 |
| 6 解码 | 按合并逆还原 | 去 `##` 拼接 | Viterbi 路径逆还原 | 按内核逆还原 | 字节按 UTF-8 拼回 | 字节直接拼回 |
| 7 对齐/偏移 | fast tokenizer 支持 | 支持 | 支持 | 支持 | **不提供** offsets | 由 patch 边界给出 |
| 8 截断/填充 | 调用方控制 | 调用方控制 | 调用方控制 | 调用方控制 | **不提供** | 调用方控制 |
| 代表 | RoBERTa | BERT、DistilBERT | T5、Gemma | LLaMA 1/2、T5 | GPT-2/3.5/4/4o | Meta BLT |

**一眼可见的结论**：第 3 层是唯一"算法级"分歧点，第 4 层的词表形式随之改变，其余各层只是"填 / 不填"。所谓"选择 tokenizer 算法"，实质是在这同一张表上选一列。

#### tiktoken 的逐层落位（重点）

OpenAI 的实现只暴露 3 个对象，恰好对应本流水线的第 2、3、4 层：

| tiktoken 对象 | 类型 | 对应层 | 作用 |
|---|---|---|---|
| `pat_str` | 正则字符串 | 第 2 层 | 预分词模式，等价于 GPT-2 regex |
| `mergeable_ranks` | `dict[bytes, int]` | 第 3 + 4 层 | 既是"哪些字节对可合并"（切分规则），又是 rank→token id 映射；**rank 同时兼任合并优先级与 token id** |
| `special_tokens` | `dict[str, int]` | 第 4 层 | 特殊 token 直连 ID，切分时优先匹配 |

它**省略**的层：第 1 层规范化（恒等）、第 5 层后处理、第 7 层 offsets、第 8 层截断/填充；第 6 层解码隐含在"字节序列本身"中。

因此结论明确：**tiktoken 不是并列的第四种算法，而是这条流水线上一份省略外围层的极简配置**——它复用第 3 层的 BPE 内核，并把第 2、4 层直接暴露成可读数据。这也解释了它为什么既能与 HF 流水线对齐，又比后者轻。

#### 与 HF 流水线的对接通道

- **自动通道**：`transformers` 的 `convert_slow_tokenizer` / `convert_tiktoken_to_fast`——加载带 `tokenizer.model` 的模型（gpt2、llama3 等）时，HF 会自动把 tiktoken 文件转成 fast tokenizer。
- **手工通道**：用 HF `tokenizers` 复刻同一份配置——`ByteLevel` 预分词 + `BPE` 模型（灌入 `mergeable_ranks`）+ `ByteLevel` 解码器 + `TemplateProcessing` 补上第 5 层。
- **对齐三检查点**（三者一致则逐 token 输出必然相同）：① 同一份 `pat_str`；② 同一套 byte↔unicode 映射；③ special token 的切分顺序与 ID 一致。
- **已知缺项**：`tokenizer.model` 文件**不含** special tokens 与 pattern string，从 tiktoken 转 HF 时须额外补齐这两项。

---

## 3. 第 3 层算法库：子词模型逐个精讲

> **层次归属**：本章全部内容都属于 §2 流水线的**第 3 层（子词切分 / Model）**。只有同层选项之间才构成"算法之争"。
>
> 注意一个例外：§3.5 的 tiktoken **不是本层内部的并列选项**，而是整条流水线的一份完整配置档案（本层取字节级 BPE，外围层省略），详见 §2.11。

### 3.1 BPE（Byte-Pair Encoding）

**出身**：Gage 1994 数据压缩；Sennrich et al. 2016 引入机器翻译；GPT-2 起成为事实标准。

**训练（自底向上合并）**：

```
1. 词表初始化 = 所有单字符（或 256 个字节）
2. 把语料按预分词切成词，每个词表示为字符序列
3. 循环：
   统计所有相邻符号对 (a,b) 的出现频次
   合并频次最高的那一对 → 新符号 ab 加入词表
   直到达到目标词表大小 / 合并次数
```

**编码（贪心按优先级合并）**：

按训练学到的**合并顺序（rank）**，每次优先合并 rank 最小（最早学到）的那一对，反复直到无法合并。等价于"从高优先级到低优先级"的确定性贪心。

**变体**

- **Byte-level BPE**：基础字母表换成 256 个字节（再做可打印化映射，避免不可见字符）。好处是**永不 OOV**，任何 Unicode 文本、emoji、乱码都能编码；代价是非英语常见词消耗更多 token。
- **BPE-dropout**：训练/推理时按概率随机跳过部分合并，同一文本产生多种切分，起到**正则化与数据增强**作用。

**优点**：实现极简、词表大小完全可控、字节级版本零 OOV、合并顺序天然体现"高频优先"。

**缺点**：贪心合并**不保证全局最优**（只局部最优）；合并不可撤销，早期错误合并永久污染词表；对形态复杂语言（土耳其语、芬兰语）效率一般；同一词不同上下文切分相同（缺乏上下文感知）。

### 3.2 WordPiece

**出身**：Schuster & Nakajima 2012（日韩语音识别），BERT 采用并闻名。

**训练**：结构与 BPE 相同（也是自底向上合并），但**合并准则不同**——BPE 用原始频次，WordPiece 用**似然增益**：

```
score(a, b) = freq(ab) / (freq(a) * freq(b))
```

选 score 最高的对合并。这本质是 PMI（点互信息）近似：只有"两个符号的共现显著超出独立假设"时才值得合并，从而更偏向统计上紧实的搭配，而非单纯高频组合。

**编码**：**最长匹配优先**（longest-match-first）。从左向右，在当前位置贪心地匹配词表中**最长**的可用子串；匹配不到则输出 `[UNK]`。词内部的续接片段加前缀 `##`（如 `playing` → `play` + `##ing`），解码时去掉 `##` 再拼接。

**特点**：需要先做标点隔离（`BertPreTokenizer`）；只做**词内**切分，不跨空格合并；相比 BPE 在小词表下往往能给出更"可解释"的子词块。

**缺点**：最长匹配同样是贪心，不保证最优；`[UNK]` 机制在字符集外会直接丢信息（BERT 对中文/emoji 不友好）。

### 3.3 Unigram LM（一元语言模型）

**出身**：Kudo 2018，与 SentencePiece 同时发布。

**训练（自顶向下剪枝）**——与 BPE 方向完全相反：

```
1. 初始化一个超大的候选子词池（语料中所有高频子串/后缀）
2. EM 迭代：用当前概率分布对所有句子做最优切分，
   再统计子词频次，重新估计每个子词的概率 p(x)
3. 计算删掉某个子词后总似然的损失（loss 增量）
4. 剪掉损失最小的若干子词（保留 top-k）
5. 重复 2-4，直到词表缩到目标大小
```

**编码（Viterbi 全局最优）**：

把文本构造成有向图（每个位置可匹配多个子词），用动态规划求使 `Σ log p(x_i)` **最大化**的切分路径。这不是贪心，是**全局最优**。

**独门能力：子词正则化（Subword Regularization）**

- 训练时按概率分布**采样**切分（而非总取最优），同句每轮切分不同 → 显著提升模型鲁棒性与多语言表现。
- 可输出 **N-best 切分列表**，下游做 ensemble。

**优点**：概率框架严谨、切分质量高、天然支持正则化与 N-best、多语言友好（尤其配合 SentencePiece）。

**缺点**：训练比 BPE 慢（EM + 剪枝多次迭代）、实现复杂；Viterbi 推理比贪心稍慢。

**代表模型**：T5、ALBERT、XLNet、mBART、Gemma/Gemini（SentencePiece-Unigram，词表 262K）。

### 3.4 SentencePiece

**注意：它不是"算法"，而是框架/工具包**——把 BPE 和 Unigram 统一封装，并解决它们的共同前提问题。

**核心突破：语言无关**

传统 BPE/WordPiece 都假设"空格可以切词"，这对中文、日文、泰文（无空格）失效。SentencePiece 直接**在原始文本流上训练**，把空格转义为可见符号 `▁`（U+2581），因此：

- 不需要预分词，不需要任何语言特定规则；
- 空格本身也成为可学习的 token（如 `▁the`）；
- 解码时可逆（`▁` → 空格）。

**关键参数**

| 参数 | 含义 |
|---|---|
| `model_type` | `bpe` / `unigram` / `char` / `word` |
| `vocab_size` | 目标词表大小 |
| `character_coverage` | 覆盖多少比例的字符（日语等大字符集语言常设 0.9995） |
| `byte_fallback` | OOV 时回退到 `<0xXX>` 字节 token，永不丢失信息 |
| `normalization_rule_name` | `nmt_nfkc` / `identity` 等 |
| `split_digits` | 数字单独切分（利于算术） |
| `user_defined_symbols` | 强制不被切碎的自定义符号 |

**代表模型**：T5（Unigram）、ALBERT、XLNet、LLaMA 1/2（BPE，32K）、Gemma/Gemini（262K）。

### 3.5 tiktoken：字节级 BPE 的完整配置档案

> **归位说明**：本节不是第 3 层内部的"第四种算法"，而是**整条流水线的一份具体配置**——第 3 层取字节级 BPE，外围层大量省略。逐层落位表见 §2.11。

GPT 家族的路线选择：**放弃"字符"概念，只用 256 个字节作为初始词表**，再做 BPE。所有文本先 UTF-8 编码成字节序列，再做可打印化映射（把不可见字节映射到可见 Unicode 区间，避免控制字符污染词表）。

| 模型 | 词表 | 特点 |
|---|---|---|
| GPT-2 | 50,257 | 首个字节级 BPE |
| GPT-3 / GPT-3.5 / GPT-4 | `cl100k_base` ≈ 100K | |
| GPT-4o | `o200k_base` ≈ 200K | 非英语与代码压缩率显著提升（2024-05） |

**优点**：对语言、书写系统、emoji 完全不可知，零 OOV，鲁棒性最强。

**缺点**：同样是贪心、对非英语常见词 token 消耗偏高（中文尤甚）；**跨语言 token 效率不公**是学界关注点——同样的语义内容，英文用户付的钱更少。

### 3.6 非主流 / 其他路线

| 路线 | 说明 |
|---|---|
| Word-level / Char-level | 历史基线，已基本不用 |
| **无 tokenizer**：ByT5、CANINE | 直接吃字节/字符序列，配合特殊架构处理长序列 |
| **BLT（Byte Latent Transformer）** | Meta 的"patch"方案：直接从原始字节学习，按数据复杂度动态决定 patch 边界，跳过词表，在规模上能与 token 方案持平且有效率优势 |
| VQ 码本（音频/图像） | 连续信号离散化，概念上的 tokenizer |
| 形态学分割 | Morfessor 等，用 MDL 准则无监督发现词素，语言学导向 |
| FastText 字符 n-gram | 严格说不是子词切分，是哈希桶 + n-gram 特征 |

---

## 4. 横向对比总表

| 维度 | BPE | Byte-level BPE | WordPiece | Unigram LM | SentencePiece(+BPE/Unigram) |
|---|---|---|---|---|---|
| 合并方向 | 自底向上 | 自底向上 | 自底向上 | **自顶向下剪枝** | 取决于内核 |
| 合并准则 | 频次最高 | 频次最高 | 似然/PMI 增益 | EM 概率 + loss 剪枝 | 继承内核 |
| 切分算法 | 贪心按 rank | 贪心按 rank | 最长匹配 | **Viterbi 全局最优** | 继承内核 |
| 确定性 | 是 | 是 | 是 | 可采样（非确定） | 可配置 |
| OOV | 字符兜底 | **无 OOV** | `[UNK]` | 字符/字节兜底 | `byte_fallback` |
| 空格处理 | 需预分词 | 视为普通字符 | 需预分词 | 依赖 SentencePiece | `▁` 转义，语言无关 |
| 训练成本 | 低 | 低 | 低 | 高（EM 迭代） | 中 |
| 可解释性 | 中 | 低（字节碎片） | 较高（`##` 清晰） | 高（概率可查） | 中高 |
| 正则化能力 | BPE-dropout | BPE-dropout | 无 | **subword sampling + N-best** | 继承内核 |
| 代表 | RoBERTa、GPT-2 | GPT 全系、LLaMA 3 | BERT、DistilBERT | T5、Gemma | LLaMA 1/2、T5 |

> **表的读法**：上表的列是"第 3 层 + 第 4 层"的组合快照，不是彼此独立的算法家族。tiktoken 没有单列，因为它正是"Byte-level BPE"列的一份工业实现——差别在于外围层的省略与暴露方式（见 §2.11）。

---

## 5. 选型与工程实践

### 5.1 决策清单

1. **语言覆盖面**：单语言（英语）→ 小词表 BPE 足够；多语言/含 CJK/代码 → SentencePiece + byte_fallback 或字节级 BPE，词表 ≥128K。
2. **是否有空格分词假设**：中文/日文必须用 SentencePiece 类方案，否则用字符级。
3. **是否需要保证无损**：需要 → byte_fallback 或字节级；不需要 → 可 NFC/NFKC 提升压缩率。
4. **是否训练期需要正则化**：需要 → Unigram 采样或 BPE-dropout。
5. **算力预算**：嵌入参数量 ≈ `vocab_size × d_model`，词表翻倍 = 嵌入层翻倍（若 tying 则与输出头共享）。

### 5.2 关键指标

- **压缩率**：`bytes/token` 或 `tokens/word`。Llama 3 英文 3.94 字符/token（Llama 2 为 3.17），同文本 token 数减少最多 15%——直接等价于训练与推理成本下降。
- **多语言均衡度**：同一语义在不同语言下的 token 数差距，即"token 税"。
- **词表利用率**：实际被激活的 token 占比，过低说明词表冗余。

### 5.3 常见坑

| 坑 | 说明 |
|---|---|
| tokenizer 与权重不匹配 | 换 tokenizer 但沿用旧权重 = ID 语义错乱，必须成对使用 |
| 特殊 token ID 硬编码 | 必须从 `config.json` 读，不同版本 ID 不同（如 LLaMA 各版本 BOS/EOS 差异） |
| 擅自加 token | 加新 token 后必须 `resize_token_embeddings` 并**重新训练 embedding**，否则新 token 是随机向量 |
| 规范化有损 | NFKC 会把全角/连字改写，做原文对齐（offset）时会错位；需用 fast tokenizer 的 `return_offsets_mapping` |
| 中文/代码效率预设 | 用英文分词效率推断中文成本是大忌，必须实测 |
| 忽略 padding 侧 | 生成模型（LLaMA 系）惯例左侧 padding，BERT 系右侧，搞错会导致 batch 推理结果错乱 |
| 训练环境依赖 | 加载 HF tokenizer 需要 `transformers` 真实可用，纯 `tokenizers` 无法兜底 `AutoTokenizer` |
| tiktoken 转 HF 缺项 | `tokenizer.model` **不含** special tokens 与 pattern string，转换后必须补齐这两项，否则特殊 token 会被切碎 |
| 跨实现结果对不齐 | tiktoken 与 HF 输出不一致时，按三检查点排查：`pat_str` 相同、byte↔unicode 映射相同、special token 切分顺序与 ID 相同 |

---

## 6. 现代趋势

1. **词表持续变大**：32K（Llama 2）→ 128K（Llama 3）→ 200K（GPT-4o `o200k_base`）→ 262K（Gemma/Gemini），目的都是提高多语言与代码压缩率。
2. **不做有损规范化**：保留大小写、空白、全角，改用字节兜底换取信息无损。
3. **数字单独切分**：Gemma/Gemini 显式拆数字，改善算术与数值推理。
4. **Anthropic 是例外**：从未公开 Claude 的词表与合并规则，官方要求通过 API 计数，不接受第三方工具推算——意味着**跨模型 token 计数不可互换**。
5. **无 tokenizer 路线**：BLT 等方案主张"patch 比 token 更好扩展"，用动态字节分组替代固定词表，在效率与鲁棒性上都呈现优势。
6. **多模态统一 token 空间**：文本、图像、音频共用 ID 空间，tokenizer 从"文本预处理器"升级为"世界离散化器"。

---

## 7. 从零学习路径（建议顺序）

| 阶段 | 做什么 | 产出 |
|---|---|---|
| 1. 建立直觉 | 用现成 tokenizer 切几段中英混排文本，观察 token 边界与 ID | 亲手看到 `你好` 被切成几个 token |
| 2. 理解 BPE | 手写 30 行 Python 实现 BPE 训练 + 编码 | 体会"合并频次最高对" |
| 3. 对比 WordPiece / Unigram | 用 HF `tokenizers` 库各训一个，比较同一语料的词表与切分 | 对比表 |
| 4. 掌握 SentencePiece | 训一个 `unigram` + `byte_fallback` + `split_digits` 的多语言分词器 | 能处理中文与 emoji |
| 5. 打通端到端 | 用自训 tokenizer 训一个 10M 参数的小 GPT，看 loss 曲线 | 理解 embedding 与词表大小的关系 |
| 6. 进阶 | BPE-dropout、subword sampling、offset 对齐、词表裁剪实验 | 能回答"这个模型该用什么 tokenizer" |
| 7. 打通实现层 | 用同一份配置在 HF `tokenizers` 与 tiktoken 之间互转（§2.11），验证同一段文本两边 token id 完全一致 | 建立"只有一条流水线，实现只是配置"的认知 |

---

## 参考来源

以下内容来自联网检索：

- [分词器：LLM 背后的隐形操作者](https://www.jxxy.net/ai/articles/theahmadosman-1938330949722509652/)
- [Advanced tokenization — beyond BPE](https://labo-llm.fr/en/fondamentaux/tokenisation-avancee)
- [What is tokenization? — AlphaCorp AI](https://alphacorp.ai/blog/what-is-tokenization)
- [Patches Over Tokens: How Byte Latent Transformers Could Change the Future of LLMs](https://engineering.zooz.com/@wiz-wizdomgr/patches-over-tokens-how-byte-latent-transformers-could-change-the-future-of-llms-de352929b977)

> 说明：BPE / WordPiece / Unigram / SentencePiece 的算法原理属稳定领域知识；词表规模、版本特性等时效性数据以上述来源为据。
*（内容由AI生成，仅供参考）*
