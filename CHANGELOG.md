# Changelog

本文件记录所有对外可见的行为变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

> 0.x 阶段：`MINOR` 版本内也可能包含**行为破坏性修复**（与 HF 语义对齐所必需），
> 此类条目一律在 `### Changed` 下显式标注 **BREAKING**，并在正文说明受影响的输入与迁移方式。

## [Unreleased]

### Changed

- **BREAKING** `pre_tokenizer = Whitespace` 的切分语义修正为 HF 等价实现（正则 `\w+|[^\w\s]+`）：
  连续 word 字符成一段、连续「非空白且非 word」符号成一段，空白丢弃。
  旧实现与 `WhitespaceSplit` 完全相同（只按空白切、不切标点），导致 `"world!"` 产出 `["world!"]`
  而 HF 产出 `["world", "!"]`。
  - 受影响输入：任何含标点 / 符号的文本。
  - 迁移：词表按 `Whitespace` 训练并依赖旧（不切标点）行为的产物，需用当前版本**重新训练**；
    若本就希望"只按空白切"，配置应显式改为 `WhitespaceSplit`（其语义未变）。
- **BREAKING** 配置解析失败不再静默降级：`from_tokenizer_json` 中 `truncation` / `padding`
  反序列化错误按 `TokenizerError` 上抛（旧实现用 `.ok()` 丢弃，畸形配置会"加载成功"但静默不生效）；
  各配置结构体启用 `deny_unknown_fields`，未知字段直接报错。
  - 迁移：原先能加载的畸形 `tokenizer.json` 现在会返回 `Err`，需修正字段或移除多余键。
- **BREAKING** 标点判定由"ASCII + 部分类别"改为严格 Unicode `P*`（Pc/Pd/Pe/Pf/Pi/Po/Ps）：
  emoji（So）、数学符号（Sm）、货币符号（Sc）不再被当作标点拆开。
  - 受影响：`BertPreTokenizer`、`Punctuation` 预分词对非 ASCII 文本的切分结果。
  - WordPiece 侧随之与 HF 对齐：整词 OOV 退化为**单个** `[UNK]`，而非逐字符多个 `[UNK]`。
- `Unigram` 训练器的剪枝日志默认静默（库代码不再无条件写 stderr），设置环境变量
  `TOK_VERBOSE=1` 可恢复逐轮输出。

### Fixed

- **BREAKING** `WordPiece` 训练器词表预算分配：旧实现按「字母表 → 整词（灌到预算上限）→
  续接片段」顺序装配，长尾语料下整词会把名额吃光，产物里**一个 `##` 片段都没有**，
  加载后词内第二字符起必然失配，整词退化为 `[UNK]`（实测 400/400 行全 UNK）。
  现改为：字符层按频次降序累计覆盖 99.95% 且最多占预算一半，同时产出裸字符与 `##` 单字符；
  剩余预算由整词与续接片段**统一按频次竞争**（同频优先整词，其次更长片段）。
  - 受影响输入：任何用旧版本训练的 `WordPiece` 词表。
  - 迁移：用当前版本**重新训练**；旧产物无法通过改配置修复。
- `Unigram` 剪枝轮次日志中"剪枝前词表大小"计算错误（`n + n/shrink - n` 等同于错值），
  改为在剪枝前快照 `pieces.len()`。
- `NormalizedString::convert_offsets_from` 增加 `debug_assert`：debug 构建下与二分的
  `convert_offsets` 逐次对拍，游标复用导致的非单调 / 跨空洞偏移错误可被测试直接捕获。
- `encode` 长文本的 O(n²) 退化（段 × 分隔符双重遍历），长文本缩放恢复线性。
- 外部 `tokenizer.json` 装配时特殊 token 角色丢失：`unk/bos/eos/cls/sep/mask` 从 `added_tokens`
  推断并逐槽位补齐（不覆盖显式配置），`unk` 缺失时回退到模型节声明的 `unk_token`。

### Added

- `normalizer = ByteLevel`（GPT-2 可见字节映射，`"你" -> "ä½ł"`），落盘形态与 HF 一致为
  `{"type":"ByteLevel"}`；`tok-cli` 侧写作 `--normalizer bytelevel`。
- `Replace` / `Split` 的**字面量（String）模式**：`PatternSpec::Literal` 在匹配前
  `regex::escape`，因此 `"."`、`"[a-z]+"` 不被解释为元字符；落盘按 HF 语义区分
  `{"String": ...}` 与 `{"Regex": ...}`，CLI 侧为 `replace:literal=` / `split:literal=`。
- `tok train --post-processor auto`（现为默认值）：训练结束后仅依据**显式成对声明**的
  `--role cls` / `--role sep`（或 `--cls` / `--sep`）推断后处理——`WordPiece` / `WordLevel`
  推断为 `BertProcessing`，其余（BPE / Unigram）推断为 `RobertaProcessing`；只声明一半或
  都不声明时保持无后处理。显式传 `--post-processor none` 仍可关闭推断。
- `crates/tok-pipeline/tests/data/roles_custom/`：自定义角色（8 个角色全部改写为 `<|...|>`
  形态）＋自动推断后处理的完整 `save_pretrained` 三件套快照；配套 golden 测试锁定
  「目录重载恢复角色」「单句 id/tokens/offsets」「pair type_ids 分段」三组 HF 期望值。
- `tok-cli`：`tok compare` 子命令（与 HF 产物逐条比对，支持 `--strict`）。
- `tok-cli` 端到端集成测试 13 例，直接 spawn `CARGO_BIN_EXE_tok-cli`，覆盖
  `train / encode / decode / inspect / compare`，报错契约锁定为「`错误:` 开头 + 非零退出码」。
- HF 对齐基准 `crates/tok-pipeline/tests/hf_parity_golden.rs`，期望值来自 HF `tokenizers`
  离线录制结果（`tests/data/*.json`），CI 因此无需 Python 与网络。
- GitHub Actions CI（`.github/workflows/ci.yml`）：`lint` / `test`（linux + windows）/
  `hf-parity-golden` / `cli-e2e` 四道门禁。

### Removed

- `crates/tok-models/tests/scratch_debug.rs`（WordPiece 手工调试草稿，从未入库，`.gitignore` 的 `scratch_*` 覆盖；相关行为已由正式单测覆盖）。
