---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_a798db87c9d611f18533525400638852
    ReservedCode1: /vB3jBnw5skpvhQFa/JHKi8dNUMYiSThMwc8Mhp5mDNffjPG2EwJDnCZg2f1roWeNwoJbxvb6JVOnU/TBDx+s168opvKElixNwIIfTwG8+yrIoBxUlHPbe5eS/940w/0GUGRXsiYK1NsI8SokZf7HEnlvhcDT8kSbbTBG28XTZy17h9ilHp7DlJi/1A=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_a798db87c9d611f18533525400638852
    ReservedCode2: /vB3jBnw5skpvhQFa/JHKi8dNUMYiSThMwc8Mhp5mDNffjPG2EwJDnCZg2f1roWeNwoJbxvb6JVOnU/TBDx+s168opvKElixNwIIfTwG8+yrIoBxUlHPbe5eS/940w/0GUGRXsiYK1NsI8SokZf7HEnlvhcDT8kSbbTBG28XTZy17h9ilHp7DlJi/1A=
---

# Tokenizer 最新研究成果综述（2026）

> 本文汇总本轮检索到的 tokenizer 相关最新研究成果，按四条主线分章组织。
> 所有论文信息（标题、arXiv 编号、提交日期、量化结论）均来自本轮联网检索的原始记录，未做推测性补充。
> 整理日期：2026-10-17

---

## 导读：四条主线的共同语境

过去一年，tokenizer 研究从"预处理细节"上升为**模型效率与能力的一等设计变量**。四条主线分别沿着不同的假设推进：

| 主线 | 核心假设 | 代表工作 |
|---|---|---|
| ① 超越贪心合并 | 问题不在"每个 token 算得更快"，而在"token 能不能更少" | ConvexTok、CNF-MCE、CoBPE、Online/Delta Tokenizer |
| ② 无分词器 / 可学切分 | 固定词表本身就是瓶颈，切分粒度应由模型自己学 | BLT、H-Net/H-Net++、ByteFlow Net、SSLM |
| ③ 词表规模与多语言公平性 | "词表越大越好"是伪命题，最优规模随部署形态与语言而变 | Lifecycle-Optimal、Modular Tokenizers、Over-Tokenized Transformer、Vocab Diet |
| ④ 多模态统一 token 空间 | tokenizer 应从"文本预处理器"升级为"世界离散化器" | LongCat-Next、UniWeTok、One Tokenizer、M2Tok、Wavelet as Tokenizer |

**评审状态总览**：本章所涉论文绝大多数为 **arXiv 预印本（Preprint）**，未标注同行评审结果；唯一明确标注已通过评审的是 Vocab Diet（ACL 2026 Findings）。其余各篇的预印本状态在正文中逐条标明，请勿将其结论视为已定稿共识。

---

## 第 ① 章 超越贪心合并的分词算法

传统 BPE / Unigram 都是**贪心算法**——每一步做局部最优合并，不考虑最终词表的整体质量；同时预分词边界限制还使 "the United States" 这类高频短语难以成为单个 token。本章的几项工作从"全局最优切分""词表构建与编码解耦""用结构替代序列长度""在线/增量计算"四个角度分别突破。

### 1.1 Tokenisation via Convex Relaxations（ConvexTok）

- **原文链接**：https://arxiv.org/abs/2605.22821
- **arXiv 编号与提交日期**：arXiv:2605.22821，v1 于 **2026-05-21** 提交（ETH Zurich + Kensho Technologies）
- **核心结论**：
  - 将 tokenizer 构建形式化为**线性规划（LP）**、用凸优化工具求解（把整数规划 IP 松弛为 LP），得到新算法 **ConvexTok**，取代 BPE/Unigram 的贪心过程。
  - 在内在分词指标与语言模型的 **bits-per-byte（BpB）** 上均取得一致改进；第三方解读给出的量化口径为 **BpB 相比 BPE 与 Unigram 降低 2–5%**；下游任务性能也有提升，但**一致性较低**。
  - 提供理论"标尺"：通过下界**证明**一个 tokenizer 距离特定目标的最优解有多远，经验上在常见词表规模下**距最优不足 1%**（256K 词表下 integrality gap 达 99.997%）。
  - 代价是**构建阶段计算开销更高**，第三方评测称构建耗时可达贪心方法的 **10–50 倍**（一次性成本，可在后续训练/推理中摊销）；另有解读提示词表规模极小（<4K）时收益递减。
- **评审状态**：arXiv 预印本（v1），**未见同行评审标注**。

### 1.2 Counting and Min-Cost Encoding for Tokenization（CNF-MCE）

- **原文链接**：https://arxiv.org/abs/2610.01127
- **arXiv 编号与提交日期**：arXiv:2610.01127，v1 于 **2026-10-01** 提交（最后更新 2026-10-02；作者含 Shuming Shi、Xiang Zhang 等）
- **核心结论**：
  - 提出 **CNF（Counting and Filtering）** 词表训练 + **MCE（Min-Cost Encoding）** 编码算法：MCE 定义文本片段代价函数，**全局最小化切分总代价**；CNF 直接统计合法子串构建原始词表，再用"以 MCE 切分语料时的实际 token 使用频率"过滤出最终词表——**把词表构建与文本编码解耦，摆脱合并历史约束**。
  - **量化结果**：250K 词表下，英文网页文本的字符级压缩率相比 `o200k_base` 提高 **26.7%**、相比 `qwen250k` 提高 **32.4%**（论文摘要口径为"提升 26% 与 30%"）。
  - 词表扩展到约百万条目时，相比同规模 BPE 的 token 效率提升 **超过 60%**（中文媒体解读口径为 **67.0%**），且**词表利用率从 52.9% 升至 96.9%**。
  - 在 1.8B 与 8B 规模从头训练的语言模型，在 **11 个 benchmark** 上平均表现与使用 BPE 的模型**相当**。
  - MCE **不依赖 merge list（如 BPE）或 token 概率（如 UnigramLM）**，因此可套用于 BPE、UnigramLM、CNF 等各类词表。
  - 切分质量直观例子：同一 `qwen250k` 词表下，`" pysimplesoap"` 被 BPE 切成 5 个 token（` pys｜imp｜les｜o｜ap`），MCE 得到 3 个（` py｜simple｜soap`）。
- **第三方解读（非论文本身，供参考）**：腾讯新闻《Token 压缩率飙升 60%：大模型分词告别贪心合并》，https://news.qq.com/rain/a/20261014A07OH200 （2026-10-14）
- **评审状态**：arXiv 预印本（v1），**未见同行评审标注**。

### 1.3 More Than Words: Compositional Tokenization（CoBPE）

- **原文链接**：https://arxiv.org/abs/2610.05597
- **arXiv 编号与提交日期**：arXiv:2610.05597，v1 于 **2026-10-04** 提交
- **核心结论**：
  - 指出标准分词下 `"On the table."` 需要 4 次独立预测（介词 / 冠词 / 名词 / 标点），每次都占用一个序列位置并增加推理成本。
  - 提出 **CoBPE**：把这类短语表示为"**词干基元 token（table）+ 一小组可复用的表层修饰符**"，在 **embedding 空间组合输入**、在**输出端联合预测**——即用结构化表示替代部分序列长度。
  - **量化结果**：在 780M 与 1.3B 规模、匹配训练算力的受控从头预训练中，CoBPE 相对标准 BPE **序列缩短 30%**，下游平均性能**提升 1.2 分**。
- **评审状态**：arXiv 预印本（v1），**未见同行评审标注**。

### 1.4 Online Tokenization with Optimal Delay in Linear Time

- **原文链接**：https://arxiv.org/abs/2610.19139
- **arXiv 编号与提交日期**：arXiv:2610.19139，v1 于 **2026-10-13** 提交（Mina Dalirrooyfard, Christopher Ye）
- **核心结论**：
  - 现有 BPE 分词流水线（特殊 token 分段 → 预分词 → BPE）是**顺序执行**的，每一阶段只能处理上一阶段的完整输出，导致首 token 延迟（TTFT）与内存占用偏高。
  - 本文**形式化并解决**了在线分词问题：引入 **delay（延迟）** 概念刻画算法输出落后于（可能计算上不可行的）最优解的程度，给出**近线性时间、最优延迟（至常数因子）**的算法。
  - 附带**条件下界**：除非 **P = NP**，不存在多项式时间算法能在延迟上超过该算法一个以上常数因子。
  - **量化结果**：在 OpenAI tiktoken 与 HuggingFace tokenizers 上实现实用变体，TTFT 相对 tiktoken（以及 gigatoken）取得 **7.9×–34.8×** 延迟优势；吞吐相比 tiktoken 提升 **2.4×–7.0×**、相比 HF tokenizers 提升 **15.8×–38.1×**；内存占用**近似恒定、与输入规模无关**。
- **评审状态**：arXiv 预印本（v1），**未见同行评审标注**。

### 1.5 Delta Tokenizer（会话增量分词）

- **原文链接**：arXiv:2610.19801（检索来源：https://arxiv-troller.com/paper/3344126 ）
- **arXiv 编号与提交日期**：arXiv:2610.19801，**2026-10-14** 提交（最后更新 2026-10-15；Zhenyu Zhang, Zhichao Cao）
- **核心结论**：
  - 指出现有推理引擎会复用共享前缀的 KV cache，但**分词没有对应捷径**——服务引擎 CPU 前端每轮都要从第一个字符重新编码整个请求。
  - 提出 **Delta Tokenizer**：只对同一会话请求与上一请求的**增量**做分词，保留既有 token 序列，仅重编码追加文本与可能受影响的短尾部，并在**分词切分规则本身无法移动的位置**拼接以保证结果精确一致。
  - 对无状态可复用的长请求，按"实用 tokenizer 已保证相邻片段独立"的边界切分，**多核并行**编码后拼成参考编码；共享 CPU 池把短增量更新调度在两个 chunk 之间执行。
  - **量化结果**：增量路径将会话轮次的分词耗时缩短约 **40×**（相对 HF 参考 tokenizer）；16 核下对 400 万字符请求比串行编码快 **8.2×**、比 LoPT 快 **1.6–2.1×**；在 SGLang 中，100K tokens 下聊天接口的**中位 TTFT 降低 59%**，负载下 agent 会话 **P99 TTFT 降 9.5×**、长请求占用前端时降 **23×**；已验证 **41.8 亿篇文档**的编码输出与参考完全一致。
- **评审状态**：arXiv 预印本，**未见同行评审标注**。

### 1.6 Beyond Atomic Tokens: Factorizing Syllables（音位分解分词器）

- **原文链接**：https://arxiv.org/abs/2609.21362 （v2）
- **arXiv 编号与提交日期**：arXiv:2609.21362，v1 **2026-09-18**，v2 **2026-09-25**
- **核心结论**：
  - 指出常规 tokenizer 把文本当作字符或统计子词，**忽略了音节的内部音位结构**，且往往需要大词表。
  - 提出 **Phonemic Tokenizer**：把越南语与中文的每个音节转为 IPA，并**分解为 onset（声母）、rime（韵母）、tone（声调）三个音位成分**；三成分**共占一个上下文位置**，保持音节级序列长度，同时让音位相关的音节**共享表示**；非音位与不支持的单元走字符级 fallback。
  - 该设计**确定性、无需依赖语料的词表学习**，词表规模仅 **中文 112 条、越南语 256 条**。
  - 内在评测：两种语言上的 **Rényi 效率显著更高**；对标准越南语音节词典中的**每一条目 Fertility 恰为 1**；越南语序列通常比现有预训练 tokenizer 更短。
  - 实例化为 **PhonemicBERT**（因子化成分 embedding + 三个预测头重建被掩码的完整音节）：中文版本在多样语言理解任务上与 character / subword / SubChar 方案**持平或更优**；越南语版本与既有越南语及多语言预训练模型相比**有竞争力或更优**。
- **评审状态**：arXiv 预印本（v1/v2），**未见同行评审标注**。

### 1.7 本章小结

- "贪心合并"的替代方案已从**启发式改良**走向**可证明的全局优化**（ConvexTok 的 LP 松弛与 <1% 最优性认证）和**词表—编码解耦**（CNF-MCE 将压缩率提升到 26%–67% 区间）。
- 与此同时，另一条更"工程"的路线在**不改变切分质量**的前提下大幅削减分词开销：在线分词（理论最优延迟）与增量分词（40× 提速）——这两项直接改善 TTFT，对 agent 类多轮长上下文场景价值明确。
- 结构感知路线（CoBPE 的"基元 + 修饰符"、音位分解的"声母/韵母/声调"）提示：**部分原本由序列长度承载的信息，可以改由结构化表示承载**，这是与"压缩率"不同维度的收益。

---

## 第 ② 章 无分词器 / 可学切分

本章共同主张：**固定词表本身是瓶颈**。要么让模型在字节流上学习自己的切分粒度，要么把分词与语言建模**联合优化**。

### 2.1 Byte Latent Transformer: Patches Scale Better Than Tokens（BLT）

- **原文链接**：https://arxiv.org/abs/2412.09871 （HTML 版：https://arxiv.org/html/2412.09871 ）
- **arXiv 编号与提交日期**：arXiv:2412.09871，**2024-12-13**（FAIR at Meta；Artidoro Pagnoni 等）
- **核心结论**：
  - 提出 **BLT（Byte Latent Transformer）**，一种纯字节级 LLM 架构，**首次在规模化下匹配基于 tokenization 的 LLM 性能**，同时显著改善推理效率与鲁棒性。
  - 把字节编码为**动态大小的 patch**，patch 作为主要计算单元；**基于下一字节的熵**进行分段——数据越复杂、越不可预测，分配越多计算与模型容量；数据可预测时选择长 patch，从而同时改善训练与推理效率。
  - 给出**首个受 FLOPs 控制的字节级模型缩放研究，规模至 8B 参数、4T 训练字节**；结果表明**可在无固定词表的情况下扩展纯字节训练模型**。
  - 在推理与长尾泛化上呈现定性改善。
- **后续进展（第三方来源）**：解读文章指出 BLT 在 2024 年 12 月即匹配 Llama 3 8B 基线且**零子词词表**，2026 年 5 月的后续工作进一步将 **decode 成本降低最多 92%**（来源：https://temperature2.com/p/2026-08-31-did-you-know-tokenizer-free-byte-level-llms ，2026-08-31）。
- **评审状态**：arXiv 预印本，**未见同行评审标注**。

### 2.2 Dynamic Chunking for End-to-End Hierarchical Sequence Modeling（H-Net）

- **原文链接**：https://arxiv.org/abs/2507.07955 （PDF：https://arxiv.org/pdf/2507.07955v1 ）
- **arXiv 编号与提交日期**：arXiv:2507.07955，v1 **2025-07-10**，v2 **2025-07-15**（Sukjun Hwang 等）
- **核心结论**：
  - 提出端到端层级序列建模的**动态分块（dynamic chunking）**，把切分学习与序列建模放在同一目标下优化。
  - 引入多项架构与训练技术以提升端到端优化的**稳定性与可扩展性**（涉及信号传播等）。
- **同族工作 H-Net++**：https://www.alphaxiv.org/es/overview/2508.05628v1 → arXiv:2508.05628，**2025-08**，标题为 **H-Net++: Hierarchical Dynamic Chunking for Tokenizer-Free Language Modelling in Morphologically-Rich Languages**，面向**形态丰富语言**的无 tokenizer 建模，并加入**两层潜在超先验机制**以捕捉文档级一致性模式。
- **会议信息（第三方来源）**：解读页标注为 **ICLR 2026**（来源：https://en.papernotes.org/ICLR2026/llm_pretraining/dynamic_chunking_for_end-to-end_hierarchical_sequence_modeling ）。该标注来自第三方站点，**未在检索到的 arXiv 页面上确认**，此处如实标注来源。
- **评审状态**：arXiv 预印本；第三方站点标注 ICLR 2026，**待以官方记录核实**。

### 2.3 ByteFlow: Language Modeling through Adaptive Byte Compression without a Tokenizer

- **原文链接**：https://arxiv.org/abs/2603.03583
- **arXiv 编号与提交日期**：arXiv:2603.03583，v1 **2026-03-03**（Chunyuan Deng 等）
- **核心结论**：
  - 指出现代 LM 仍依赖固定子词分词，一旦 tokenizer 训练完成，模型只能在该固定粒度上工作，即使在推理能力很强的模型上也会产生**脆弱且反直觉的行为**。
  - 提出 **ByteFlow Net**：一种层级架构，**完全移除 tokenizer**，让模型自行学习把原始字节流切分为语义有意义的单元。
  - 关键机制：基于**潜在表示的编码率（coding rate）** 做**压缩驱动的分段**，产生自适应边界，同时**通过 Top-K 选择保持静态计算图**（避免动态图开销）。
  - 与依赖**人为设计归纳偏置的脆弱启发式**的自分词方法不同，ByteFlow Net 让内部表示粒度**适配输入本身**。
  - **量化结果**：实验显示该压缩式分块策略带来**显著性能提升**，ByteFlow Net **优于基于 BPE 的 Transformer 以及此前的字节级架构**，表明端到端无 tokenizer 建模不仅可行而且更有效。
- **评审状态**：arXiv 预印本（v1），**未见同行评审标注**。

### 2.4 What Tokens are Learned when Tokenization is Optimized Jointly with Language Modeling?

- **原文链接**：https://arxiv.org/abs/2608.17325
- **arXiv 编号与提交日期**：arXiv:2608.17325，v1 **2026-08-18**
- **核心结论**：
  - 在**18 种类型学与书写系统各异**的语言上，比较无 tokenizer 方法（**SSLM、H-Net**）与固定 tokenizer。
  - **联合优化从根本上改变了 token 结构**：SSLM 学到的是**与形态对齐、上下文高效**的 token；H-Net 则优先**字节级效率**，产生**更长的 token**，与标准子词词表**重叠度极低**。
  - **分词行为随语言类型学变化**：黏着语在训练中表现出**更动态的分割模式**。
  - 下游评测（预训练后微调的 BERT）显示：**基于 SSLM 的预分词在语言建模困惑度上一致更低**，且尽管词表差异明显，**下游性能仍有竞争力**。
  - 总体结论：无 tokenizer 方法优化的是**上下文与计算效率**，而非严格的形态结构，因此得到的是**结构迥异但同样有效**的词表。
- **评审状态**：arXiv 预印本（v1），**未见同行评审标注**。

### 2.5 本章小结

- BLT 是本章的"基准点"：**8B / 4T 字节**的受控缩放证明"无固定词表"在规模化下不掉队，并且熵驱动分段带来的是**计算分配**上的收益（可预测处用长 patch）。
- 机制上出现了清晰的分野：**熵驱动（BLT）** 与 **编码率/压缩驱动（ByteFlow Net）**——后者强调"不依赖人设计启发式"和"用 Top-K 保持静态计算图"。
- 二章与三章存在**关键张力**：H-Net 类方法的 token 与标准子词词表**重叠极低**，说明"可学切分"产出的可能是一套**与现有生态不兼容**的表征（对复用现有 embedding、评测与工具链构成现实约束）。

---

## 第 ③ 章 词表规模与多语言公平性

"词表越大越好"曾是主流直觉（32K → 128K → 200K → 262K）。本章的工作把这句话拆开检验，得到的结论更细致：**收益的来源是特定的（集中在高频词），而且最优规模取决于部署形态与语言集合**。

### 3.1 Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training

- **原文链接**：https://arxiv.org/abs/2508.15390
- **arXiv 编号与提交日期**：arXiv:2508.15390，v1 **2025-08-21**，v2 **2025-10-27**，v3 **2025-11-28**（KAIST / NAVER Cloud）
- **核心结论**：
  - 在**数据、算力、优化完全不变**的前提下，做词表规模从 **24K 扩到 196K** 的受控实验。
  - 用 **Kolmogorov 复杂度**形式化 tokenized text 的复杂度，证明**更大词表确实降低该复杂度**。
  - 但关键发现是：**超过 24K 之后，每个常见词都已成为单个 token**，继续扩大词表**只会加深相对 token 频率的不平衡**。
  - **词级 loss 分解**显示：更大词表降低交叉熵**几乎全部来自对最高频 2,500 个词的不确定性下降**，而**稀有长尾上的 loss 反而上升**。
  - 由于**同一批高频词覆盖了下游 benchmark 中约 75% 的 token**，这一训练优势**原封不动地转移到了下游**。
  - 进一步表明：**在固定词表下扩大模型参数**能得到与扩大词表**相同的高频词收益**。
  - 结论重述：把"更大的词表有帮助"改写为"**降低 tokenization 的复杂度**"才是收益的实际归因。
- **评审状态**：arXiv 预印本（v1–v3），**未见同行评审标注**。

### 3.2 Lifecycle-Optimal Tokenization: Vocabulary Size as a Deployment-Regime-Dependent Infrastructure Parameter

- **原文链接**：https://arxiv.org/abs/2608.11361
- **arXiv 编号与提交日期**：arXiv:2608.11361，v1 **2026-08-11**（Rima Mittal, Ankit Gubrani, Satyanarayana Kakollu；6 页 / 3 图 / 6 表）
- **核心结论**：
  - 提出词表规模是**部署形态依赖的基础设施参数**，而非训练时按惯例固定的常量。形式化总部署成本：**C_lifecycle(V) = C_train(V) + λ · C_infer(V, B)**（λ 为推理量，B 为服务批量）。
  - 在两代 GPU（A10G，ridge ≈ 117 FLOP/byte；A100，ridge ≈ 183 FLOP/byte）上做受控实验，跨越**内存受限到计算受限**两个区间，结论：
    1. **推理最优词表随服务批量移动 16 倍**：B=1 时约 **32K**，B=64+ 时约 **524K**，驱动力是 **V×d unembedding 矩阵读取的摊销**。
    2. 在 **1.3–2.3B** 模型规模上，质量（bits per byte, BPB）最优出现在 **V = 65K**，确认**词表偏好与规模相关**。
    3. 对生产部署而言，**生命周期最优词表与训练最优词表最多相差 16 倍**。
  - **质量在最优区间内近似不变（BPB 波动 <2%）**，说明在该测量范围内词表是一个**纯粹的系统优化问题、不付质量代价**。
  - 可操作的容量规划建议：**端侧部署（B=1）用 V ≈ 32K；数据中心服务（B ≥ 64，λ ≥ 10）用 V ≈ 131–262K**。
- **评审状态**：arXiv 预印本（v1），**未见同行评审标注**。

### 3.3 Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling

- **原文链接**：arXiv:2501.16975（检索来源：https://www.alphaxiv.org/pt/abs/2501.16975 ；Google Scholar 收录页）
- **arXiv 编号与提交日期**：arXiv:2501.16975，**2025-01-28**（Hongzhi Huang, Defa Zhu, Banggu Wu, Yutao Zeng, Ya Wang, Qiyang Min, Xun Zhou / ByteDance Seed）
- **核心结论**：
  - 提出 **Over-Tokenized Transformers**，**把输入词表与输出词表解耦**，以改善语言建模性能；核心做法是**扩大输入词表**以利用 multi-gram（n-gram）token。
  - 关键发现：**输入词表规模与训练 loss 之间存在 log-linear 关系**，即**指数级增大输入词表，loss 线性下降**，且**与模型规模无关**；表明 **embedding 参数是 LLM scaling law 中一个新增的可扩展维度**。
  - **量化结果**：**OE-12.8M（1280 万输入词表）+ 400M 参数**的模型，训练 loss 与 **1B 参数**基线相当（等效于 **2.5 倍**大的模型），而训练 **FLOPs 增加不到 0.5%**，额外计算开销极小。
  - 背景观察：扩大输入词表开销极小、扩大输出词表训练开销显著；此前的结论互相矛盾（更大 tokenizer 有利于大模型却可能损害小模型），解耦输入/输出词表正是为了化解这一矛盾。
  - 消融结论：**n-gram 编码（输入侧）在所有模型规模上都提升性能**，而 n-gram 解码（输出侧）只对**更大模型**有帮助。
- **评审状态**：arXiv 预印本，**未见同行评审标注**。

### 3.4 Scaling LLM Pre-training with Vocabulary Curriculum

- **原文链接**：https://arxiv.org/abs/2502.17910
- **arXiv 编号与提交日期**：arXiv:2502.17910，v1 **2025-02-25**（Fangyuan Yu）
- **核心结论**：
  - 现代语言模型依赖**预训练前固定的静态词表**，这与人学习语言时**自适应获得词汇**的过程相反。
  - 提出 **vocabulary curriculum learning（词表课程学习）**：在**熵引导的词表扩展**与**模型优化**之间交替，从而获得**相对词表规模的 log-linear 缩放增益**，并让模型学到可跨不同 token 粒度迁移的表征。
  - 该方法自然产生一种**最优算力分配模式**：**更长的 token 承接可预测内容，更短的 token 专注更难预测的复杂上下文**。
  - 在小规模 GPT 模型上验证了缩放效率提升；论文开源代码，并计划扩展到更大模型与更多领域。
- **评审状态**：arXiv 预印本，ADS 记录标注 **"under review"**（投稿审阅中）。例如 https://ui.adsabs.harvard.edu/abs/2025arXiv250217910Y/abstract

### 3.5 To Each Language Its Tokenizer: Modular Tokenizers for Efficient Multilingual LLMs

- **原文链接**：https://www.arxiv.org/abs/2609.15528 （https://arxiv.org/abs/2609.15528 ）
- **arXiv 编号与提交日期**：arXiv:2609.15528，v1 **2026-09-14**（Franck Signe Talla, Hippolyte Pilchen, François Yvon, Édouard Grave）
- **核心结论**：
  - 问题定位：多语言 LLM 传统上让**所有语言共用一个词表**，导致各语言**压缩不均**；且巨大的 embedding / 输出矩阵增加显存占用、拖慢推理（对小模型尤甚）；而模型实际常只服务**部分语言**，共用大词表也是浪费。
  - 方案一：提出学习**大型模块化 BPE 与 Unigram tokenizer** 的方法，可从中**抽取面向任意语言子集的子 tokenizer**。这些子 tokenizer 的**压缩率与单语 tokenizer 相当**，并**改善跨语言公平性**。
  - 方案二：设计预训练策略，**按 batch 采样子 tokenizer**，把预测**限制在相关词表子集**内，从而在大词表下仍能高效训练；支持**任意语言词表组合的高效推理**。
  - 结果：**在不牺牲性能的前提下降低显存占用、加速推理**。
- **评审状态**：arXiv 预印本（v1），**未见同行评审标注**。

### 3.6 Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic

- **原文链接**：https://www.arxiv.org/abs/2510.17001 （https://arxiv.org/abs/2510.17001 ）
- **arXiv 编号与提交日期**：arXiv:2510.17001，v1 **2025-10-19**，v2 **2026-04-20**
- **核心结论**：通过**向量算术**重塑 LLM 的词表（词表裁剪 / 重构方向）。检索记录仅覆盖到标题、编号、提交与更新时间，**未获取到该文的量化结论**，故此处不做量化陈述。
- **评审状态**：**ACL 2026 Findings**（检索页面标注）——本综述范围内**唯一明确标注已通过同行评审**的论文。

### 3.7 本章小结

- **"更大词表"的收益被归因到了具体位置**：几乎全部来自**最高频 ~2,500 词**的不确定性下降，且这批词覆盖下游约 **75%** 的 token（Exploiting Vocabulary Frequency Imbalance）。长尾 loss 反而变差。
- **最优词表不是一个数**：它随 **服务批量**（32K ↔ 524K，16 倍）、**模型规模**（1.3–2.3B 时 65K 最优）、**部署位置**（端侧 vs 数据中心）而变（Lifecycle-Optimal）。同时质量在该区间内 **BPB 波动 <2%**，说明这是**系统优化而非质量权衡**。
- **输入与输出应分开考虑**（Over-Tokenized Transformer）：扩大**输入**词表是 **log-linear 廉价**的缩放维度（FLOPs +<0.5% 换等效 2.5 倍模型），扩大**输出**词表代价高且只对大模型有利。
- **多语言公平性的解法从"共享大词表"转向"可抽取的模块化子词表"**（Modular Tokenizers）——这与第 ③ 章整体判断一致：**词表不该是一张全局唯一的表**。

---

## 第 ④ 章 多模态统一 token 空间

本章核心命题：把非语言模态也**离散化为 token**，与文本共用同一 ID 空间与同一自回归目标，让 tokenizer 从"文本预处理器"变成**世界离散化器**。关键争议点是：**离散化是否必然牺牲视觉理解能力**。

### 4.1 LongCat-Next: Lexicalizing Modalities as Discrete Tokens

- **原文链接**：https://arxiv.org/html/2603.27538v1 （arXiv:2603.27538；代码：https://github.com/meituan-longcat/LongCat-Next ）
- **arXiv 编号与提交日期**：arXiv:2603.27538，v1 **2026-03**（Meituan LongCat Team；检索页另标注 2026-03-29）
- **核心结论**：
  - 提出 **DiNA（Discrete Native Autoregressive）** 统一范式：在**共享离散空间**内表示多模态信息，使跨模态自回归建模一致且有原则。
  - 核心创新 **dNaViT（Discrete Native Any-resolution Visual Transformer）**：在**任意分辨率**下做 tokenization 与 de-tokenization，把连续视觉信号转成**层级离散 token**；可达 **最高 28× 压缩率**。
  - 识别出离散视觉建模的**双重瓶颈**：(i) 视觉表示容量；(ii) 离散化信息损失。对应解法：用 **Semantic-and-Aligned Encoders（SAE）** 保证语义完整性，用 **RVQ（残差向量量化）** 对"残差的残差"建模以保留理解与生成所需信息；并发现编码器的**残差结构天然保留了低级信号传播的潜在通路**（即使无重建监督）。
  - 自回归建模上用**多层 token 的加性编码** + **DepthTransformer** 高效解码，为多级 token 解锁**指数级表示空间**，同时保持**单步自回归的计算效率**。
  - 音频侧：Whisper 编码器 + **8 层 RVQ**，把波形压缩为 **12.5 Hz** 的离散 token；配对解码器 + flow matching 精修网络实现高保真重建；并引入文本/音频 token 的**随机延迟对齐**统一训练范式。
  - **量化结果**：视觉理解上**匹配 Qwen3-VL-A3B**；**GenEval 84.44 vs. Flux-dev 66.00**，文本渲染亦优于 Flux-dev；音频任务**超过 Gemini 2.5 Flash-Lite 与 MiMo-Audio** 等 omni / 语音专用模型。
  - 意义：**首次**表明纯离散视觉 tokenizer 能在理解 benchmark 上**追平甚至超过连续视觉编码器**，打破"离散 token 天然不利于视觉理解"的长期信念；同时在同一自回归框架内**调和了理解与生成的冲突**。
  - 局限（论文/解读均指出）：**强依赖 tokenizer 的语义完整性**；**尚未扩展到 any-to-any 交织式多模态推理**，仍限于经典的理解与生成任务。
- **评审状态**：arXiv 预印本，**未见同行评审标注**。

### 4.2 UniWeTok: An Unified Binary Tokenizer with Codebook Size 2^128

- **原文链接**：https://www.arxiv.org/abs/2602.14178 （https://arxiv.org/abs/2602.14178 ）
- **arXiv 编号与提交日期**：arXiv:2602.14178，v1 **2026-02-15**，v2 **2026-03-01**，v3 **2026-03-11**
- **核心结论**：
  - 统一 MLLM 需要视觉表示**同时**支持**高保真重建、复杂语义抽取、生成适用性**，而现有视觉 tokenizer 难以在单一框架内同时满足这三项冲突目标。
  - 提出 **UniWeTok**：使用**超大二进制码本（2^128）** 的统一离散 tokenizer。
  - 训练框架：**Pre-Post Distillation** 与 **Generative-Aware Prior**，增强离散 token 的语义抽取与生成先验。
  - 架构：**卷积-注意力混合架构 + SigLu 激活函数**——SigLu 既约束编码器输出、稳定语义蒸馏，又有效处理 **token entropy loss 与 commitment loss 之间的优化冲突**。
  - 提出**三阶段训练框架**，提升跨分辨率与感知敏感场景（人脸、文本内容）的适应性。
  - **量化结果**：ImageNet 上达到 **SOTA 图像生成性能（FID：UniWeTok 1.38 vs. REPA 1.42）**，且训练算力**极低（训练 token：UniWeTok 33B vs. REPA 262B）**；通用域上，图像生成 **DPG Score 86.63 vs. FLUX.1 [Dev] 83.84**，编辑 **GEdit Overall 5.09 vs. OmniGen 5.06**。
- **评审状态**：arXiv 预印本（v1–v3），**未见同行评审标注**。

### 4.3 Studying Image Tokenizers as Visual Languages in Unified Multimodal Models

- **原文链接**：https://arxiv.org/abs/2609.09143 （代码：https://github.com/amazon-far/Tokenizer_UMM ）
- **arXiv 编号与提交日期**：arXiv:2609.09143，v1 **2026-09-08**（Siting Li, Zhengyang Wang, Simon Shaolei Du, Xi Chen, Yang Liu；27 页 / 23 图）
- **核心结论**：
  - 指出图像 tokenizer 定义了统一多模态模型的"**视觉语言**"，但常被以**孤立指标**或**仅生成 / 仅理解**的评测来研究，无法反映视觉 token 与文本**联合建模**时的行为。
  - 方法：构建**受控的纯自回归测试台**，在多模态持续预训练中追踪**任务特定验证 loss**（文本、图像、T2I、I2T 四类预测），考察 loss 的缩放行为及其与下游性能的关系。
  - 四条发现：
    1. **loss 必须按任务分开分析**——不同任务的缩放行为不同，且对 tokenizer 的**排序也不同**。
    2. **loss–性能关系取决于所预测的 token 空间**：固定 tokenizer 时 T2I/I2T loss 与生成质量相关；**跨 tokenizer** 时 T2I 的 loss–性能关系随**图像 token 空间**漂移，而 **I2T loss（在共享文本词表上计算）提供更一致的信号**，并在监督微调后**同时关联生成质量与视觉理解性能**。
    3. **更好的重建不必然带来更低的任务 loss 或更强的下游性能**。
    4. **图像 tokenizer 的选择会影响文本建模**（在联合优化下）。
  - 以**判别器、语义监督、词表规模**三个设计轴为案例，检视它们对联合建模与下游性能的影响。
- **评审状态**：arXiv 预印本（v1），**未见同行评审标注**；Hugging Face Papers 收录并获 28 upvotes（第三方站点记录）。

### 4.4 Mind the Gap No More: Achieving Zero-Gap Multimodal Integration via One Tokenizer

- **原文链接**：https://arxiv.org/abs/2602.12286 （HTML：https://arxiv.org/html/2602.12286 ）
- **arXiv 编号与提交日期**：arXiv:2602.12286，检索页标注发布时间 **2026-08-12**（Yanan Li、Yuan Jin、Manli Luo、Tie Xu、Shuai Jiao、Wei He、Qing Zhang / Zhejiang Lab）
- **核心结论**：
  - 形式化刻画多模态的**几何模态间隙（modality gap）**，并**理论证明**：**原生架构（尤其是使用统一词表的架构）在所有隐藏层上内在保持 zero-gap 状态**。
  - 论证链条：模块化双编码器在**独立表示空间**中运行，因高维几何原因**必然存在严格为正的初始间隙**；而原生架构下所有模态**共享统一符号词表与单一底层分布**，**期望初始间隙随词表规模增大趋近于零**。由于 LLM 层的 **Lipschitz 连续性**，初始几何距离**无法通过加深网络压缩**，因此模块化设计的初始间隙在**整个前向传播中严格保留**，迫使网络持续消耗算力做几何对齐。
  - 基于理论提出 **One Tokenizer**：把所有模态**直接映射到共享 token 空间**的原生架构。
  - 在 **DNA–文本** 多模态测试台上验证：One Tokenizer **持续优于基于编码器的模块化对应方案**，为深度生物推理提供更优框架。
- **评审状态**：arXiv 预印本，**未见同行评审标注**（第三方研究信号站给出 75/100 的"研究信号"评分，该评分**不代表正确性或重要性**）。

### 4.5 Wavelet as Tokenizer: Preliminary Results on a Shared Wavelet Token Schema for Natural Signals

- **原文链接**：https://fr.arxiv.org/abs/2606.02631 （https://arxiv.org/abs/2606.02631 ）
- **arXiv 编号与提交日期**：arXiv:2606.02631，v1 **2026-05-30**（Shenghao Ding；12 页 / 3 图；eess.AS）
- **核心结论**：
  - 研究问题：**音频、图像、视频能否共享同一套小波 token schema**，而非各自依赖模态专用的潜在网格。
  - 模型构成：单级 **Haar DWT/IDWT** 前端、**共享系数-token 布局**、可选结构元数据、轻量**模态值适配器**、**共享的 token-wise 编解码主干**。
  - **量化结果**：在 Speech Commands、EuroSAT RGB、DAVIS 2017 上，密集共享模型达到 **音频 39.92 dB、图像 29.37 dB、视频 23.93 dB PSNR**；匹配速率的连续潜在标量预算扫描表明**视觉收益并非仅由潜在容量解释**，同时显示**加性元数据 embedding 并非普适的改进来源**。
  - **固定速率能量选择**提供强非参数基线：在压缩保留比下，`energy_global` 相对均匀选择**平均 PSNR 提升 16.73 dB（音频）、16.90 dB（图像）、15.86 dB（视频）**；**掩码稀疏训练**在仅用 **50%** 密集 token 时达到 **34.45 dB 视频 PSNR**。
  - 结论边界（论文自陈）：结果**支持统一小波 token schema 与稀疏 token 接口**，但**尚不足以建立通用离散词表**。
- **评审状态**：arXiv 预印本（v1），**未见同行评审标注**。

### 4.6 M2Tok: Multi-head Multi-codebook Discrete Action Tokenization for VLA Models

- **原文链接**：https://arxiv.org/abs/2609.18259 （https://it.arxiv.org/abs/2609.18259 ）
- **arXiv 编号与提交日期**：arXiv:2609.18259，v1 **2026-09-16**，v2 **2026-09-17**
- **核心结论**：
  - 问题：原始动作信号是连续的，需有效 tokenization 才能自回归处理；现有离散动作 tokenizer **重建损失高**，无法保留精确控制所需的**细粒度动力学**，这一"**离散化瓶颈**"显著限制了下游 Vision-Language-Action（VLA）模型的性能上限。
  - 提出 **M2Tok**：两项结构创新——(1) 把潜在动作特征**分解为多个 head**，让特定 head 隐式对齐特定动作维度；(2) **为每个 head 分配独立码本**做量化。利用多码本的**组合特性**大幅扩展 tokenizer 的表示能力，显著降低重建损失。
  - 在 **RoboTwin、Simpler-Env 及 3 个零样本真实任务**上评测：**重建保真度更优**，且**显著提升 VLA 模型成功率**；消融实验确认多头与多码本机制的有效性。代码已公开。
- **评审状态**：arXiv 预印本（v1/v2），**未见同行评审标注**。

### 4.7 本章小结

- **"离散化必然损失理解能力"这一长期信念正在被推翻**：LongCat-Next 以纯离散视觉 tokenizer 在理解任务上追平 Qwen3-VL-A3B，并在 GenEval（84.44 vs 66.00）与音频任务上取得优势；UniWeTok 用 2^128 二进制码本在 ImageNet FID 上压过 REPA（1.38 vs 1.42），且训练 token 数减少约 8 倍。
- **理论侧给出统一词表的"第一性"理由**：One Tokenizer 证明统一词表架构在**所有隐藏层**上内在 zero-gap，而模块化双编码器的初始几何间隙因 Lipschitz 连续性**无法通过加深网络消除**。
- **但评测方法论提出了反直觉警告**：更好的重建**不必然**带来更好的下游表现；且必须**按任务分别分析 loss**——跨 tokenizer 比较时，只有**在共享文本词表上计算的 I2T loss** 才是较一致的信号（Studying Image Tokenizers）。
- **模态边界持续外扩**：除文本/图像/音频外，已覆盖**动作**（M2Tok）、**生物序列**（One Tokenizer 的 DNA–文本）、**自然信号共享 schema**（Wavelet as Tokenizer，但作者明确止步于"通用离散词表"）。

---

## 跨论文趋势判断

1. **"减少 token 数量"已成为与"加速单个 token"并列的一等效率路线。** CNF-MCE 用全局最小代价切分把 250K 词表下的压缩率提升 26%–32%、百万词表下提升 60%+；CoBPE 以"基元 + 修饰符"缩短序列 30% 并同时提升下游 1.2 分。二者与注意力/KV-Cache 优化方向正交，可叠加。

2. **分词器构建正在从"贪心启发式"转向"有最优性保证的全局优化"。** ConvexTok 用 LP 凸松弛把"距离最优有多远"变成**可认证的量**（常见词表规模下 <1%），并附上 P=NP 条件下无法突破的延迟下界（Online Tokenization）。这意味着分词器设计首次具备了**可比较的理论标尺**。

3. **"最优词表规模"这个提法正在被"分层解耦"取代。** 证据来自三个方向：输入词表与输出词表应解耦（Over-Tokenized Transformer，输入侧 log-linear 且 FLOPs 仅 +<0.5%）；最优词表随服务批量在 32K–524K 间移动 16 倍且质量波动 <2%（Lifecycle-Optimal）；多语言场景用**可抽取的模块化子词表**替代全局共享表（Modular Tokenizers）。工程含义是：**词表从训练期常量变成部署期参数**。

4. **词表扩大的收益高度集中，"越大越好"被证伪。** 24K→196K 的受控实验显示，收益**几乎全部来自最高频 2,500 个词**的不确定性下降，长尾 loss 反而上升；而同一批高频词覆盖下游约 75% token，因此训练收益能原样转移（Exploiting Vocabulary Frequency Imbalance）。这与 Lifecycle-Optimal 的 "<2% BPB 波动" 相互印证：**在最优区间外扩大词表基本是浪费容量**。

5. **多模态统一的争论已从"工程偏好"升级为"几何必然性 + 评测方法论"两级。** 理论上，统一词表架构在**所有隐藏层内在 zero-gap**，模块化编码器的初始间隙**无法靠加深网络消除**（One Tokenizer）；实证上，LongCat-Next 与 UniWeTok 分别以 28× 压缩比和 2^128 码本在理解/生成上取得 SOTA 或接近 SOTA。但 Studying Image Tokenizers 给出的反直觉结论（**重建更好 ≠ 下游更好**，loss 必须**按任务**分析，I2T loss 才是最一致信号）说明：**该领域的评测标准尚未收敛**。

6. **分词器的价值正在从"文本预处理"移到"服务端基础设施"。** Delta Tokenizer 在 SGLang 中把 100K tokens 的中位 TTFT 降低 59%、agent 会话 P99 TTFT 降低 9.5×；Online Tokenization 给出近线性时间的最优延迟算法与 7.9×–34.8× 的 TTFT 优势。对多轮长上下文的 agent 负载而言，**分词开销已从"可忽略的前置步骤"变成显性瓶颈**。

---

## 与《Tokenizer完全指南.md》的衔接建议

> 现有文档：[Tokenizer完全指南.md](<C:\Users\neroy\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_c3844db437f84410b72023faeb6c3c62\output\Tokenizer完全指南.md>)
> 其结构为：§1 取舍三角 → §2 六层流水线（2.3 子词切分 Model 为核心）→ §3 算法库（BPE / WordPiece / Unigram / SentencePiece / tiktoken）→ §4 横向对比总表 → §5 选型与工程实践 → §6 现代趋势 → §7 学习路径 → 参考来源。

现有指南是一份**稳定的基础框架**，本文是**前沿增量**。建议按以下方式衔接，而非改写指南主干：

| 衔接位置 | 建议动作 |
|---|---|
| **§2.3 子词切分（核心 Model）** | 指南把该层定位为"唯一算法分歧层"。建议在此加一段**旁注**：ConvexTok 表明该层的贪心性本身可被 LP 凸松弛替代，且**距最优性可认证到 <1%**——即"分歧层"内部还有一层"局部最优 vs 全局最优"的分歧。 |
| **§2.9 实现与性能** | 指南此处讲实现与批处理。建议补充**在线/增量分词**这一新维度：Online Tokenization 的 delay 概念与 Delta Tokenizer 的 40× 增量提速、SGLang 中 TTFT 降 59%，都属实现层的新议题，且**不改变任何切分语义**（Delta Tokenizer 输出的 41.8 亿篇文档与参考完全一致）。 |
| **§3.6 非主流 / 其他路线** | 指南该节是现有算法之外的收容区。建议把 **CNF-MCE（词表-编码解耦，不依赖 merge list 或 token 概率）**、**CoBPE（基元 + 修饰符的组合式 tokenization）**、**Phonemic Tokenizer（onset/rime/tone 音位因子化，中文词表仅 112）** 三条路线并入，并标注它们都**不属于 BPE/Unigram 家族**。 |
| **§4 横向对比总表** | 建议新增一张"**新一轮方法对比表**"（维度：优化目标 / 是否需训练词表 / 词表规模 / 压缩率或 BpB 增益 / 构建成本 / 是否改语义），把 ConvexTok、CNF-MCE、CoBPE、BLT、ByteFlow、Modular Tokenizers 纳入，与原有 BPE/WordPiece/Unigram 表并列而非合并——因为它们的**对比维度不同**（原表是算法族，新表是设计取向）。 |
| **§5.1 决策清单 / §5.2 关键指标** | 这是最大的更新点。指南当前的关键指标偏"压缩率 + 词表大小"；建议改为**分层决策**：①输入/输出词表**分别**定；②按**部署形态**定规模（端侧 B=1 → 32K；数据中心 B≥64 → 131–262K，依据 Lifecycle-Optimal）；③按**语言集合**定是否用模块化子词表；④对 agent/多轮负载，把 **TTFT 与分词 CPU 开销**纳入指标。 |
| **§6 现代趋势 第 5、6 条** | 指南现有第 5 条（无 tokenizer 路线，提到 BLT）与第 6 条（多模态统一 token 空间）方向正确，但需要**用 2026 年的新证据补强与修正**：第 5 条可补 H-Net++（形态丰富语言）、ByteFlow（编码率驱动、Top-K 静态图）、SSLM vs H-Net 的 18 语对比（**token 与标准子词词表重叠极低**这一生态兼容性问题必须写明）；第 6 条可补 LongCat-Next 的 GenEval 84.44 vs 66.00、UniWeTok 的 FID 1.38 vs 1.42，以及 One Tokenizer 的 zero-gap 理论——同时**必须**加上 Studying Image Tokenizers 的反直觉警告（**重建更好 ≠ 下游更好**）。 |
| **§6 建议新增第 7 条** | "**词表规模不再是训练期常量**"：Lifecycle-Optimal 给出的 16 倍跨度与 <2% BPB 波动，是与指南 §6 第 1 条（"词表持续变大"）**直接冲突**的新证据，建议并列呈现，让读者看到张力而非单边结论。 |
| **参考来源** | 指南当前参考来源只有 4 条、且以科普解读为主。本文第 ①–④ 章的 arXiv 原始链接可作为**论文级来源**补入，并在顶部注明"预印本，未经同行评审"。 |
| **§7 学习路径** | 建议在阶段 6（进阶）后增加"阶段 7'：读一篇全局优化分词论文 + 复现一个增量分词 TTFT 实验"，把前沿从"知道"推进到"能验证"。 |

**一句话总括**：指南负责**稳定的骨架**（六层流水线 + 算法族），本文负责**骨架上的新生长点**——建议以"旁注 + 新增对比表 + §6 补强"的方式增量合并，**不要用前沿结论覆盖指南中已成立的原理性内容**。

---

## 完整来源清单

### 论文（arXiv 原始链接）

| 编号 | 标题 | 提交日期 | 评审状态 |
|---|---|---|---|
| [arXiv:2605.22821](https://arxiv.org/abs/2605.22821) | Tokenisation via Convex Relaxations（ConvexTok） | 2026-05-21 | 预印本 |
| [arXiv:2610.01127](https://arxiv.org/abs/2610.01127) | Counting and Min-Cost Encoding for Tokenization in Large Language Models（CNF-MCE） | 2026-10-01 | 预印本 |
| [arXiv:2610.05597](https://arxiv.org/abs/2610.05597) | More Than Words: Compositional Tokenization for Efficient Language Models（CoBPE） | 2026-10-04 | 预印本 |
| [arXiv:2610.19139](https://arxiv.org/abs/2610.19139) | Online Tokenization with Optimal Delay in Linear Time | 2026-10-13 | 预印本 |
| [arXiv:2610.19801](https://arxiv.org/abs/2610.19801) | Delta Tokenizer: Reference-Identical Tokenization of Session Deltas and Segment-Parallelism for Long Prompts on Multicore CPUs | 2026-10-14 | 预印本 |
| [arXiv:2609.21362](https://arxiv.org/abs/2609.21362) | Beyond Atomic Tokens: Factorizing Syllables for Language Model Pretraining（Phonemic Tokenizer） | 2026-09-18（v2 09-25） | 预印本 |
| [arXiv:2412.09871](https://arxiv.org/abs/2412.09871) | Byte Latent Transformer: Patches Scale Better Than Tokens（BLT） | 2024-12-13 | 预印本 |
| [arXiv:2507.07955](https://arxiv.org/abs/2507.07955) | Dynamic Chunking for End-to-End Hierarchical Sequence Modeling（H-Net） | 2025-07-10 | 预印本（第三方标注 ICLR 2026，待核实） |
| [arXiv:2508.05628](https://arxiv.org/abs/2508.05628) | H-Net++: Hierarchical Dynamic Chunking for Tokenizer-Free Language Modelling in Morphologically-Rich Languages | 2025-08 | 预印本 |
| [arXiv:2603.03583](https://arxiv.org/abs/2603.03583) | ByteFlow: Language Modeling through Adaptive Byte Compression without a Tokenizer | 2026-03-03 | 预印本 |
| [arXiv:2608.17325](https://arxiv.org/abs/2608.17325) | What Tokens are Learned when Tokenization is Optimized Jointly with Language Modeling? | 2026-08-18 | 预印本 |
| [arXiv:2508.15390](https://arxiv.org/abs/2508.15390) | Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training | 2025-08-21（v3 2025-11-28） | 预印本 |
| [arXiv:2608.11361](https://arxiv.org/abs/2608.11361) | Lifecycle-Optimal Tokenization: Vocabulary Size as a Deployment-Regime-Dependent Infrastructure Parameter | 2026-08-11 | 预印本 |
| [arXiv:2501.16975](https://arxiv.org/abs/2501.16975) | Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling | 2025-01-28 | 预印本 |
| [arXiv:2502.17910](https://arxiv.org/abs/2502.17910) | Scaling LLM Pre-training with Vocabulary Curriculum | 2025-02-25 | 预印本（ADS 标注 under review） |
| [arXiv:2609.15528](https://arxiv.org/abs/2609.15528) | To Each Language Its Tokenizer: Modular Tokenizers for Efficient Multilingual LLMs | 2026-09-14 | 预印本 |
| [arXiv:2510.17001](https://arxiv.org/abs/2510.17001) | Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic | 2025-10-19（v2 2026-04-20） | **ACL 2026 Findings** |
| [arXiv:2603.27538](https://arxiv.org/abs/2603.27538) | LongCat-Next: Lexicalizing Modalities as Discrete Tokens | 2026-03 | 预印本 |
| [arXiv:2602.14178](https://arxiv.org/abs/2602.14178) | UniWeTok: An Unified Binary Tokenizer with Codebook Size 2^128 for Unified Multimodal Large Language Model | 2026-02-15（v3 2026-03-11） | 预印本 |
| [arXiv:2609.09143](https://arxiv.org/abs/2609.09143) | Studying Image Tokenizers as Visual Languages in Unified Multimodal Models | 2026-09-08 | 预印本 |
| [arXiv:2602.12286](https://arxiv.org/abs/2602.12286) | Mind the Gap No More: Achieving Zero-Gap Multimodal Integration via One Tokenizer | 2026-08-12（检索页标注） | 预印本 |
| [arXiv:2606.02631](https://arxiv.org/abs/2606.02631) | Wavelet as Tokenizer: Preliminary Results on a Shared Wavelet Token Schema for Natural Signals | 2026-05-30 | 预印本 |
| [arXiv:2609.18259](https://arxiv.org/abs/2609.18259) | M2Tok: Multi-head Multi-codebook Discrete Action Tokenization for Vision-Language-Action Models | 2026-09-16（v2 09-17） | 预印本 |

### 第三方解读与聚合来源（非论文原文，仅作交叉参考）

| 来源 | 关联内容 | 链接 |
|---|---|---|
| 腾讯新闻《Token 压缩率飙升 60%：大模型分词告别贪心合并》 | CNF-MCE 量化口径（250K 词表 +26.7% / +32.4%；百万词表 +67.0%）、BPE 三点局限 | https://news.qq.com/rain/a/20261014A07OH200 |
| SynapsFlow《ConvexTok kills greedy tokenisation: 5% BpB gain for NLP》 | ConvexTok BpB 2–5%、构建耗时 10–50×、方法对比表 | https://synapsflow.com/article/convextok-kills-greedy-tokenisation-5-bpb-gain |
| alphaXiv（ConvexTok 页） | ConvexTok 论文解读、DAG/IP-LP 形式化说明 | https://www.alphaxiv.org/ro/abs/2605.22821 |
| alphaXiv（Over-Tokenized Transformer 页） | OE/OD 机制与 12.8M 词表实验细节 | https://www.alphaxiv.org/pt/abs/2501.16975 |
| arXiv Vanity | Lifecycle-Optimal Tokenization 全文镜像 | https://arxiv-vanity.com/papers/2608.11361 |
| Emergent Mind | ByteFlow 摘要页 | https://www.emergentmind.com/papers/2603.03583 |
| alphaXiv | H-Net++ 概览页 | https://www.alphaxiv.org/es/overview/2508.05628v1 |
| papernotes.org | H-Net 动态分块（标注 ICLR 2026） | https://en.papernotes.org/ICLR2026/llm_pretraining/dynamic_chunking_for_end-to-end_hierarchical_sequence_modeling |
| temperature2.com《Why tokenizer-free LLMs decode byte by byte》 | BLT 匹配 Llama 3 8B、后续 decode 成本降 92% | https://temperature2.com/p/2026-08-31-did-you-know-tokenizer-free-byte-level-llms |
| EngineersOfAI | Studying Image Tokenizers 论文拆解与代码链接 | https://engineersofai.com/docs/research/paper-breakdowns/2026-09-08-studying-image-tokenizers-as-visual-languages-in-unified-multimodal-models |
| Hugging Face papers-content | Mind the Gap（One Tokenizer）全文 Markdown | https://huggingface.co/buckets/huggingchat/papers-content/tree/2602/2602.12286.md |
| arXiv Troller | CNF-MCE 与 Delta Tokenizer 的编号/日期/摘要页 | https://arxiv-troller.com/paper/3325468 ・ https://arxiv-troller.com/paper/3344126 |
| PulseAugur | SuTRA（arXiv:2608.18087）、TokEval（arXiv:2608.18062）、Autocompleting Tokenizers（arXiv:2608.15080）线索 | https://pulseaugur.com/cluster/206281-new-method-compresses-transformer-inputs-by-predicting-and-removing-redundant |
| NASA ADS | Scaling LLM Pre-training with Vocabulary Curriculum（标注 under review） | https://ui.adsabs.harvard.edu/abs/2025arXiv250217910Y/abstract |

### 与既有一致性说明

- 本文所有论文的标题、编号、提交日期与量化结论均**逐条对应本轮检索记录**（检索结果文件与逐页抓取记录），未引入记录之外的论文、数字或链接。
- 若某篇论文的量化数据在检索记录中**未被覆盖**（如 Vocab Diet 仅取到标题与日期），本文明确标注"未获取到量化结论"，**不做推测性填充**。
- 第三方站点的标注（如 H-Net 的 ICLR 2026、研究信号评分）与论文自身信息**分开标注来源**，不混同。
*（内容由AI生成，仅供参考）*
