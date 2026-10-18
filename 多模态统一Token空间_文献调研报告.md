---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_3707d479c9d311f18533525400638852
    ReservedCode1: /uqvB9R3N7aAl2Ka8qPQy0ryxK9RtLj5SOB8Me0TzXl6H+qUSli8TTtbP+FQtvDUoQRsaS4eNN3NGr1Kfgp2TZxXGkMDgf7IIDuCxEj+4Q2M6FStHil9Ek5+tF0TRToJ/MxNBoNTgsZ9e9bYp5lILGcaF4qEnbb7Dg4GJQ9yc3iIsLRyeRNrXMEwljI=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_3707d479c9d311f18533525400638852
    ReservedCode2: /uqvB9R3N7aAl2Ka8qPQy0ryxK9RtLj5SOB8Me0TzXl6H+qUSli8TTtbP+FQtvDUoQRsaS4eNN3NGr1Kfgp2TZxXGkMDgf7IIDuCxEj+4Q2M6FStHil9Ek5+tF0TRToJ/MxNBoNTgsZ9e9bYp5lILGcaF4qEnbb7Dg4GJQ9yc3iIsLRyeRNrXMEwljI=
---

# 多模态统一 Token 空间：深度文献调研

> 调研日期：2026-10-17
> 调研对象：上一份文档中"tokenizer 从文本预处理器升级为世界离散化器"这一判断的完整展开
> 覆盖范围：2017（VQ-VAE）— 2026（CVPR 2026），含奠基论文、系统论文、综述与工程实现
> 引用规范：标注为"论文/综述"的为原始文献；标注为"技术分析"的为二手整理，可信度弱于原文

---

## 目录

- [0. 摘要：一句结论](#0-摘要一句结论)
- [1. 这句话的三个必要条件](#1-这句话的三个必要条件)
- [2. 三种设计范式](#2-三种设计范式)
- [3. 各模态的离散化算法栈](#3-各模态的离散化算法栈)
- [4. 统一词表的工程细节](#4-统一词表的工程细节)
- [5. 统一空间的四大挑战](#5-统一空间的四大挑战)
- [6. 评测体系与关键数据点](#6-评测体系与关键数据点)
- [7. 代表系统对照表](#7-代表系统对照表)
- [8. 趋势判断](#8-趋势判断)
- [9. 参考文献](#9-参考文献)

---

## 0. 摘要：一句结论

"多模态统一 token 空间"不是"把图片也塞进文本词表"这么简单，而是一整套**架构范式的替换**：把过去"各模态各用一套编码器、最后用适配器（adapter）投影拼接"的**晚期融合（late fusion）**，换成"所有模态先各自离散化为整数序列，拼成一条序列，交给同一个 Transformer 用同一个 next-token prediction 目标训练"的**早期融合（early fusion）**。

这条技术线上真正被解决的问题是**训练目标的统一**；真正没被解决的问题是**离散化的信息损失**与**模态间的参数/梯度竞争**。而 2024 年 Transfusion 的对照实验给出了一个反直觉结论：**在视觉生成上，保留图像为连续表示反而比强行离散化更省算力**——这直接动摇了"必须统一到离散 token"的前提。

---

## 1. 这句话的三个必要条件

要称得上"统一 token 空间"，三个条件必须同时满足，缺一不可：

| 条件 | 内容 | 只满足前 N 项时的称呼 |
|---|---|---|
| ① 各模态都有离散化器 | 图像→visual tokenizer，音频→neural codec，文本→BPE | 多 tokenizer 系统 |
| ② 离散单元进入同一 ID 空间 | 共享一张 embedding 表、一个输出头（lm_head） | 共享 embedding / 混合词表 |
| ③ 单一序列 + 单一目标 | 拼成一条序列，统一用 next-token prediction 训练 | **统一 token 空间** |

对比一下晚期融合（2020–2023 主流，如 Flamingo、IDEFICS，以及 LLaVA 系列）：

| 维度 | 晚期融合 Late Fusion | 早期融合 Early Fusion（token-based） |
|---|---|---|
| 视觉表示 | 连续特征（ViT/CLIP 输出） | 离散 token（来自码本） |
| 跨模态汇合点 | 网络中部或末端用适配器投影 | 序列最开头就拼接 |
| 训练目标 | 各有损失（对比损失 + 语言建模） | 全部统一为 next-token prediction |
| 图像生成 | 通常不具备，或另挂一个 diffusion 头 | 原生支持，只需预测下一个图像 token |
| 信息损失 | 适配器投影有损 | 离散量化有损（横向量化误差） |
| 工程复杂度 | 高（多模块 + 对齐训练） | 低（一个 Transformer 走天下） |

Meta 的 Chameleon 论文把这条路线追溯到 CM3（2022）与 CM3Leon（2023），前者把自回归 token 建模用于混合模态文档，后者用于文生图（技术分析来源）。

---

## 2. 三种设计范式

同样是"共享"，共享的方式差异很大，实际工程里可归为三类。

### 范式 A：子集式共享（图像 token 是文本词表的子集）

**代表：Chameleon**

- 训练一个统一 BPE tokenizer，词表 **65,536**；
- 图像侧用一个 **8,192** 大小的 codebook，512×512 图像切成 **1024 个 patch**，每个 patch 从码本取一个 token → 整图变 1024 个 token；
- 这 8192 个图像 token **直接占用文本词表内的 8192 个位置**（而不是追加在末尾）；
- 于是文本 token 与图像 token 在 ID 空间里不分区，模型只能靠上下文判断当前 token 的模态。

**含义**：词表预算被图像"吃掉"了一部分，文本词表实际可用容量变小；换来的好处是词表总量可控、无需额外的模态路由逻辑。

### 范式 B：拼接式分区（文本区 + 模态区 + 控制符区）

**代表：Liquid、Emu3、MMaDA**

- Liquid：基座 Gemma-7B 文本词表 256,000，图像用 VQVAE 编成 32×32=1024 个 token，码本 8192；做法是把整个词表扩到 **256,000 + 8,192**，前段归文本、后段归图像，并把 lm_head 同步扩 8192 维（论文来源）；
- MMaDA-Parallel 的实际布局（来自代码文档）：

| 区间 | 内容 |
|---|---|
| `0 – 126355` | 文本子词（126,356 个） |
| `126336` | `MASK`（迭代解码时的占位符） |
| `126084` | 换行符（2D 图像 token 网格的行分隔） |
| `126349` | `BOI`（Begin-of-Image） |
| `126350` | `EOI`（End-of-Image） |
| `126351` | `UNCONDITION`（无分类器引导的无条件路径） |
| 其余 | 图像 VQ 码、其他控制符 |

- Emu3：把图像、文本、视频都离散化进一个空间，词表规模 65,536（论文摘要 + 技术分析），自回归 transformer 单目标训练，并能生成视频。

**含义**：这是目前工程上最主流的形态——**显式分区 + 显式控制符**。分区避免 ID 冲突，控制符负责告诉模型"模态从这里切换"以及"这里要用哪种解码方式"。

### 范式 C：不共享离散空间（各模态各保留最合适的表示）

**代表：Transfusion**

- 目标函数**按模态分开**：文本用 next-token prediction（离散），图像用 diffusion（连续）；
- 实现方式：文本 token 走标准 embedding，图像走 **patchification**（每块图像 patch 一个向量）；
- 注意力也是分模态的：**文本用因果注意力，图像 patch 用双向注意力**；
- 每个训练步同时暴露两种模态与两个损失。

**对照实验结果（论文 2408.11039）**：
- 文生图：Transfusion 用 **不到 1/3 的算力**就超过 Chameleon 的离散方案（FID 与 CLIP 双指标）；
- 控制 FLOPs 后，Transfusion 的 FID 约为 Chameleon 的 **1/2**；
- 图生文：Transfusion 用 **21.8% 的 FLOPs** 追平 Chameleon。

**含义**：这是整条技术线上最重要的一记反证——**"统一"的价值主要在训练目标与架构的简洁性，而不必然要求"统一到离散"**。强行把图像离散化，会用掉大量词表容量与建模能力来换取一个"任意模态都能自回归生成"的形式统一。

---

## 3. 各模态的离散化算法栈

### 3.1 视觉：从 VQ-VAE 到语义优先

**第一代：VQ-VAE（2017，van den Oord 等）**
结构 = 编码器 + 码本 + 解码器：
1. 编码器把图像映射为连续 latent 网格；
2. 每个 latent 向量在码本中做**最近邻查表** `k = argmin_j ||z_e - e_j||₂`，替换为码本向量；
3. 解码器重建图像。

三个关键工程件：
- **直通估计器**（straight-through estimator）：`argmin` 不可导，训练时把解码器输入的梯度**直接复制**给编码器输出；
- **承诺损失**（commitment loss）：约束编码器输出不要离码本向量太远；
- **EMA 码本更新**：用指数滑动平均更新码本，比梯度更新稳定。

**第二代：VQ-GAN（2021，Esser 等）**
在 VQ-VAE 上加三样东西：更强的编解码器（残差块 + 注意力）、**PatchGAN 判别器**（对抗训练）、**感知损失**。结果是重建更锐利、码本表达力更强。此后"VQGAN 系"成为统一多模态模型的标准视觉分词器（Unified-IO 系列、CM3 等均采用）。

**码本效率的三条改良路线**

| 路线 | 做法 | 解决的问题 |
|---|---|---|
| **残差量化（RVQ / RQ-VAE）** | 同一位置用多层码本由粗到细逐层量化，token 是"层 × 索引" | 单个大码本难以覆盖细节；分层量化在相同码率下质量更高 |
| **FSQ（Finite Scalar Quantization, 2023, Google）** | 不做向量量化，直接投影到 **<10 维**，每维量化到少量固定值，**隐式码本 = 各维取值集合的笛卡尔积** | 彻底规避**码本坍缩**；不再需要承诺损失、码本重播种、码本分裂、熵惩罚等一整套复杂机制 |
| **LFQ（Look-free Quantization，MAGVIT-v2）** | 去掉查表式量化，用可扩展的方式把码本做大 | 把码本容量从早期视觉 tokenizer 的约 8K 提升到 **2¹⁸（262,144）** |

MAGVIT-v2 的论文标题本身就是结论：**"Language Model Beats Diffusion — Tokenizer is key to visual generation"**（论文 2310.05737）——在 ImageNet 与 Kinetics 上，配了好 tokenizer 的语言模型超过 diffusion；同时该 tokenizer 在视频压缩上可与下一代视频编解码器相媲美。

**语义 vs 像素的核心矛盾**

视觉 tokenizer 面临一个"不可能三角"：**理解**任务偏好连续语义嵌入，**生成**任务需要离散 token 供自回归，**重建**任务要求保留量化通常会破坏的细节。三条主流解法：

| 解法 | 代表 | 思路 |
|---|---|---|
| 对比学习隐式对齐语义 | VILA-U（2409.04429） | 视觉塔同时用文本-图像对比损失 + VQ 重建损失训练，让离散 token 与文本语义对齐 |
| 双分支显式解耦 | TokenFlow、SemHiTok | 语义信息与像素信息走两条分支 |
| 量化语义特征而非像素特征 | VQRAE | 用连续语义特征做理解、把同一批特征量化后做生成与重建（两阶段 + 自蒸馏） |
| 层次化解耦冲突代理任务 | Rosetta Stone（CVPR 2026） | 论文明确验证"对比学习式模型在统一生成与理解上存在**内在冲突**"，故把冲突的代理任务分层解耦，并引入 attention-prioritized mapping |

Rosetta Stone 的量化结果：256×256 下 **rFID 0.33**、ImageNet 零样本 **80.9%**，比 VILA-U 高 7.6%，并超过 SigLIP 的连续嵌入；落到 7B 离散统一 MLLM 上，理解超 TokenFlow-13B 3.1%（论文）。

### 3.2 音频：神经 codec + 语义 token

音频离散化的独门结构是**残差向量量化（RVQ）**：同一时间步配多层码本，由粗到细逐层量化，还原时把各层向量相加。没有它，"每秒几十到上百个 token"会直接爆炸。

| 系统 | 配置 | 设计要点 |
|---|---|---|
| SoundStream | 常用 4 / 8 / 16 层 RVQ，论文甚至做到 80 层 | 证明层数可极大扩展；同码率下"少层大码本"与"多层小码本"可互换 |
| EnCodec（Meta） | 24kHz 最多 32 码本，48kHz 最多 16 个，每码本 1024 条目 | 训练时随机选层参与量化 → 一套权重覆盖 1.5/3/6/12/24 kbps 多档带宽 |
| Jukebox（OpenAI） | 三层不同时间尺度的 VQ-VAE | 顶层最稀承载段落结构，底层最密承载音色细节（**横向按时间分层**，不同于 RVQ 的同一时间步纵向堆叠） |

**语义 token 与声学 token 的分工**
- **语义 token**：用自监督模型（HuBERT / wav2vec 2.0 / WavLM / w2v-BERT）抽帧级嵌入，再做 k-means 聚类取中心；偏向"内容/音素"；
- **声学 token**：纯声学监督的 RVQ 码；偏向"音色/韵律/房间混响"。

**最精彩的一体化设计：Mimi codec（Moshi）**
- 每 **12.5 ms** 一帧输出 **8 个 RVQ 索引**；
- **Split RVQ**：第 1 层码本用"预测 WavLM token"做语义监督，充当**流式语义 token**；其余 7 层是标准声学残差量化；
- 瓶颈层用**因果注意力的 Transformer**，保证可流式；
- 8 层索引的自回归交给 **RQ Transformer**：在时间轴上自回归建模第 1 层，在**深度轴**上以第 1 层为条件顺序预测其余 7 层。

这套设计让 Moshi 成为端到端全双工语音对话模型：不需要 ASR→LLM→TTS 级联，天然支持打断、重叠说话与副语言信息，也是目前最接近 GPT-4o 语音模式的开源形态（技术分析来源）。

**统一到 LLM 的做法**（综述整理）：
- Unified-IO 2：用 ViT-VQGAN 架构训练音频分词器；
- AnyGPT、MIO：直接采用 SpeechTokenizer；
- C3LLM：用 EnCodec + 两阶段解码（粗粒度语言模型预测 + 细粒度非自回归精炼）。

### 3.3 文本与视频
- **文本**：复用 BPE，但要重新训练——Chameleon 用新训练的 BPE 词表（65536，含图像 token）；Liquid 直接基于 Gemma 的 256K 词表做扩展。
- **视频**：多数模型沿时间轴逐帧离散化（VILA-U、LWM）；Emu3 在 MoVQGAN 中引入**时间残差 3D 卷积层**，实现时空联合压缩以支持长视频；有分析指出处理超 30 秒长视频需采用分层自回归（先关键帧再补中间帧），否则出现时序漂移。

---

## 4. 统一词表的工程细节

### 4.1 词表布局的两种写法
- **内嵌（subset）**：如 Chameleon，图像 token 占据文本词表内的位置；
- **外挂（append）**：如 Liquid / MMaDA，文本词表之后追加模态区段与控制符。综述明确总结：**"离散词元通常作为扩展词汇表附加到文本词汇之后，并引入模态分隔符与任务控制符"**。

### 4.2 必需的控制符清单
| 控制符 | 作用 |
|---|---|
| `BOI` / `EOI` | 标记图像 token 序列的起止 |
| `MASK` | 掩码式并行解码的未揭示占位符 |
| 换行符 / 行分隔 | 2D 图像 token 网格的结构化 |
| `UNCONDITION` | 无分类器引导（CFG）的无条件路径 |
| 模态分隔符、任务控制符 | 指示模态切换与任务类型 |

### 4.3 参数代价
词表从 `V` 扩到 `V + k` 意味着：
- 输入 embedding 与输出 lm_head **同步扩 k 行**（Liquid 明确扩了 8192 维）；
- 若启用权重共享（tying），扩一次即可；若不共享，则参数翻倍计算；
- 训练时大词表会显著抬高输出层的 softmax 成本——这也是"词表预算 vs 序列长度"权衡在多模态下的重演。

### 4.4 训练不稳定及其解药（Chameleon 的公开经验）
"在两种模态上训练同一个模型，规模化时训练不稳定"，团队做了三处改动（技术分析来源）：
1. **QK-Norm**：在注意力点积之前归一化 query 与 key 向量；
2. **层归一化重排**：34B 模型把 LN 移到注意力块与前馈块**之后**，参考 Swin Transformer 的做法；
3. **z-loss**：加一个 `1e-5 × log²(softmax 配分函数)` 的小惩罚项，压住输出 logits 的漂移。

这三条对任何"要在同一张表里混两种模态分布"的训练都是现成的救火清单。

### 4.5 序列长度与 token 速率（量级参考）
| 模态 | 量级 |
|---|---|
| 图像 | Chameleon：512×512 → 1024 token；Liquid：512×512 → 32×32 = 1024 token；自适应分辨率方案常给 256–4096 token |
| 音频 | 神经 codec 通常 **50–100 token/秒**；Mimi 更密（8 RVQ/12.5ms，即每帧 8 个索引） |
| 视频 | 常见 8 fps 时序采样 + 每帧若干 token |

---

## 5. 统一空间的四大挑战

### 5.1 码本利用率坍缩（Codebook Collapse）
**问题**：只有少数码本条目被反复使用，大量条目成为死码，离散表示的表达力被浪费。
**现有手段**：重参数化技巧、多样性正则。
**综述指出的两难**：这些手段"往往以牺牲稳定性为代价"，未来方向是**课程式的码激活调度**与**混合码本设计**（引入语义/球面等结构先验）。
**制度性解法**：FSQ 通过"笛卡尔积隐式码本"从架构上取消了这个问题的发生条件。

### 5.2 离散化的信息损失
**问题**：多个不同的连续嵌入被映射到同一个码，语义上不同的实体变得不可区分；在码本过小或坍缩时尤其严重。
**性质**：这是离散化的**固有代价**，不能根除。
**综述建议的方向**：任务与模态感知的差异化策略——图像生成可容忍低显著区域的损失，分类/检索则需要更高码精度；按下游目标自适应分配码本容量。

### 5.3 梯度传播
**问题**：量化操作破坏可导性。
**手段**：直通估计器（主流）、Gumbel-Softmax、承诺损失、EMA 码本更新、FSQ 的隐式码本（无需查表即无梯度断裂点）。

### 5.4 模态竞争与代理任务冲突
**表现**：
- 词表容量与参数预算在模态间零和分配；
- 不同模态的梯度分布差异大，出现训练不稳定（见 §4.4）；
- **理解 vs 生成对视觉表示的要求内在冲突**——CVPR 2026 的 Rosetta Stone 论文专门验证了对比学习式模型在统一两者时存在固有冲突，并通过层次化解耦来缓解；
- 自回归建模不符合图像的**内在空间结构**（逐 token 生成效率低，通常超过 1k token）——这也是离散去噪（掩码预测、并行解码）路线兴起的动机。

---

## 6. 评测体系与关键数据点

### 6.1 分层指标体系（综述整理）
| 层次 | 指标 |
|---|---|
| tokenizer 重建质量 | **rFID / FID**、PSNR、SSIM |
| 语义保真 | 对比学习对齐（CLIP score）、ImageNet 零样本准确率 |
| 图像/视频生成 | FID、CLIP score、FVD |
| 理解能力 | MMBench、VQA |
| 语音 | WER、WIL、MOS |
| 词表本身 | 码本利用率、token 效率（压缩率 / BPT）、切分稳定性 |
| 跨模态 | BLEU / METEOR / ROUGE / CIDEr / SPICE（captioning） |

### 6.2 值得记住的量化结论
| 结论 | 出处 |
|---|---|
| Transfusion 文生图以 **<1/3 算力**超过 Chameleon 的离散方案；控 FLOPs 时 FID 低约 **2×**；图生文用 **21.8% FLOPs** 追平 | Transfusion 论文 2408.11039 |
| Rosetta Stone 达成 **rFID 0.33** + ImageNet 零样本 **80.9%**，超 VILA-U 7.6%，并超 SigLIP 连续嵌入 | CVPR 2026 论文 |
| 视觉 token 数在保持语义保真的前提下可压到原始量的 **10–25%** | 综述整理（2025） |
| 有效 tokenizer 的统一模型（QLIP、ILLUME+、Slot-MLLM 等）一致优于碎片化 / 纯 patch 基线，且参数与资源成本仅为零头 | 综述整理 |
| "一刀切"的 tokenization 很少够用，最优策略高度依赖任务、模态与数据 | 综述整理 |

---

## 7. 代表系统对照表

| 系统 | 年份 | 统一方式 | 视觉 tokenizer | 词表 / 码本 | 训练目标 | 一句话定位 |
|---|---|---|---|---|---|---|
| **VQ-VAE** | 2017 | —（基础组件） | 编码器+码本+解码器 | 可配 | 重建 + VQ | 视觉离散化奠基 |
| **VQ-GAN** | 2021 | —（基础组件） | VQ-VAE + 对抗 + 感知损失 | 常 1K–16K | 重建 + 对抗 | 视觉 tokenizer 事实标准 |
| **FSQ** | 2023 | —（量化方案） | 投影到 <10 维、每维固定值 | 隐式笛卡尔积 | 同 VQ 下游任务 | 取消复杂机制且不坍缩 |
| **MAGVIT-v2** | 2023/24 | —（视觉 tokenizer） | LFQ，码本扩至 2¹⁸ | 262,144 | 自回归 LM 做视觉生成 | "语言模型超过 diffusion" |
| **Chameleon** | 2024-05 | 范式 A：子集式共享 | VQ 系，512² → 1024 patch | 8,192 图像 / 65,536 总 | 纯 next-token prediction | 早期融合 token 方案的标杆 |
| **Emu3** | 2024-09 | 范式 B：拼接式 | MoVQGAN + 时间残差 3D 卷积 | 65,536 | 纯 next-token prediction | 图/文/视频单目标统一，胜过 SDXL 与 LLaVA-1.6 |
| **Transfusion** | 2024-08 | 范式 C：不共享离散空间 | 无（图像保持连续 patch） | — | 文本 NTP + 图像 diffusion | 用对照实验否定强制离散化 |
| **Liquid** | 2024-12 | 范式 B：拼接式 | VQVAE，512² → 32² | Gemma 256K + 8,192，lm_head 扩 8192 | next-token prediction | 可挂在任意现成 LLM 上的通用改造 |
| **VILA-U** | 2024-09 | 范式 B | 残差量化 + 对比损失对齐 CLIP 语义 | — | next-token prediction | 让离散 token 带语义 |
| **Show-o** | 2024 | 范式 B + 去噪 | MAGVIT-v2 风格因果 CNN 分词器 | — | 掩码 token 预测（并行去噪） | 离散去噪路线的代表 |
| **MMaDA** | 2025 | 范式 B | VQ 图像码 + 显式分区 | 126K 文本 + 控制符区 | 全模态离散扩散 | 把去噪范式扩展到文本 |
| **Moshi** | 2024 | 音频+文本联合 | Mimi codec（split RVQ） | 8 RVQ / 12.5ms | 层次化自回归（RQ Transformer） | 端到端全双工语音对话 |
| **VQRAE** | 2025/26 | 单 tokenizer 多用途 | 量化语义特征，连续→理解、离散→生成/重建 | — | 两阶段 + 自蒸馏 | 一个 tokenizer 同时服务理解/生成/重建 |
| **Rosetta Stone** | CVPR 2026 | 单 tokenizer 多用途 | 层次解耦冲突代理任务 + attention-prioritized mapping | — | 统一 MLLM | 正面处理"理解 vs 生成"的内在冲突 |
| **Mind the Gap No More** | 2026-08 | One Tokenizer | — | — | — | 宣称用单一 tokenizer 达到"零差距"多模态集成（之江实验室） |
| **BLT** | 2025 | 无 tokenizer | 动态 patch（非固定词表） | — | 字节级建模 | 反方向：取消 tokenizer 本身 |

---

## 8. 趋势判断

1. **"统一"的重心从形式转向效果**：2024 年 Chameleon 确立了"纯离散 + 单目标"的范式；同年 Transfusion 用更少算力证明强制离散化有代价。此后研究重心不再是"能不能统一"，而是"统一到什么程度最优"。
2. **连续与离散的混合成为默认答案**：文本必然离散（语言本就是符号），图像/音频保留连续通道或用连续-离散双输出，正在成为新共识（VQRAE、Transfusion 均属此类）。
3. **tokenizer 从"压缩器"升级为"语义对齐器"**：VILA-U 用对比学习、Rosetta Stone 用层次解耦、VQRAE 用量化语义特征——目标都是让离散 token 携带语义而非仅像素。
4. **一个 tokenizer 服务多种任务**：理解、生成、重建三合一是 2025–2026 的明确焦点（VQRAE、Rosetta Stone、Mind the Gap）。
5. **量化机制本身在"去复杂化"**：FSQ / LFQ 的思路是删掉 VQ 的一整套工程补丁（承诺损失、码本重播种、码本分裂、熵惩罚），用结构性设计换稳定性。
6. **工业界路线已明确转向原生统一**：有技术分析指出 GPT-4o、Gemini 1.5 Pro、DeepSeek-VL2 这一代已全面转向 early fusion 与 unified tokenization，视觉 token 自适应 256–4096 个、音频 codec 每秒 50–100 token、统一用 2D/3D RoPE 绑定时空跨度。
7. **并行解码取代逐 token 自回归**：离散去噪（掩码预测）带来数量级加速，但引入"训练时掩码上下文 vs 推理时完整上下文"的不匹配与域差距，是尚在解决的新问题。
8. **反方向也在跑**：BLT 主张"patch 比 token 更好扩展"，用动态字节分组彻底绕开词表——如果这条线成立，"世界离散化器"的最终形态可能根本没有固定词表。

---

## 9. 参考文献

### 奠基算法
1. Van den Oord et al., *Neural Discrete Representation Learning*（VQ-VAE, 2017）
2. Esser et al., *Taming Transformers for High-Resolution Image Synthesis*（VQ-GAN, 2021）
3. Mentzer et al., *Finite Scalar Quantization: VQ-VAE Made Simple* — https://arxiv.org/html/2309.15505v2
4. Yu et al., *Language Model Beats Diffusion — Tokenizer is key to visual generation*（MAGVIT-v2 / LFQ）— https://oplclaw.com/paper/107068
5. Zeghidour et al., *SoundStream*；Défossez et al., *High Fidelity Neural Audio Compression*（EnCodec）
6. 李维，*说说神经 codec，大模型时代的音频技术要点* — https://blog.sciencenet.cn/blog-362400-1501649.html

### 统一多模态系统
7. Chameleon Team, *Chameleon: Mixed-Modal Early-Fusion Foundation Models*（2024-05）— 解读：https://blog.ailab.sh/2024/06/introducing-chameleon-transforming.html ；https://aiwiki.ai/wiki/chameleon
8. Emu3 Team (BAAI), *Emu3: Next-Token Prediction is All You Need* — https://arxiv.org/abs/2409.18869
9. Meta, *Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model* — https://arxiv.org/html/2408.11039
10. *Liquid: Language Models are Scalable and Unified Multi-modal Generators* — https://arxiv.org/html/2412.04332v4/
11. *VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation* — https://arxiv.org/abs/2409.04429
12. Défossez et al. (Kyutai), *Moshi: A Speech-Text Foundation Model for Real-Time Dialogue* — https://achuthrao.com/notes/papers/moshi-a-speech-text-foundation-model-for-real-time-dialogue-2024
13. Sun et al., *Rosetta Stone For Unified MLLMs: A Unified Tokenizer to Decipher Understanding and Generation*, CVPR 2026, pp. 22964–22974 — https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Rosetta_Stone_For_Unified_MLLMs_A_Unified_Tokenizer_to_Decipher_CVPR_2026_paper.html
14. Li et al., *Mind the Gap No More: Achieving Zero-Gap Multimodal Integration via One Tokenizer*（arXiv 2602.12286, 2026-08）
15. *VQRAE: One Tokenizer for Understanding, Generation, and Reconstruction* — https://www.emergentmind.com/videos/vqrae-quantized-autoencoder-for-multimodal-ai-e425c1d2

### 综述与工程整理
16. *Discrete Tokenization for Multimodal LLMs: A Comprehensive Survey* — https://arxiv.org/pdf/2507.22920v1
17. *Multimodal Tokenization Overview* — https://www.emergentmind.com/topics/multimodal-tokenization
18. *Discrete Speech Tokens* — https://www.emergentmind.com/topics/discrete-speech-tokens
19. 多模态理解与生成综述（83 页）— https://2048ai.net/693988272087ae0db7a0ca0d.html
20. MMaDA-Parallel Token Space Design — https://deepwiki.com/tyfeld/MMaDA-Parallel/3.2.3-token-space-design
21. *Patches Over Tokens: How Byte Latent Transformers Could Change the Future of LLMs* — https://engineering.zooz.com/@wiz-wizdomgr/patches-over-tokens-how-byte-latent-transformers-could-change-the-future-of-llms-de352929b977
22. 2026 年 AI 前沿技术全景深度解析（原生多模态章节）— https://opc.csdn.net/6a801876662f9a54cb9d042c.html

> 可信度说明：第 1–15 项为论文或论文官方解读页；第 16–22 项为综述或技术社区整理，其中具体数值（如 Emu3 的 256 token 序列长度、FID=5.7、UCF101 82.1%）来自二手转载，引用前建议回原文核对。
*（内容由AI生成，仅供参考）*
