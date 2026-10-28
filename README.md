---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_3c1b3934d27c11f18460525400393706
    ReservedCode1: ezLUF4hRNuZABqErx+faTjNBIGKLfftvnrpUclwwJA50ZOVvWQYIbGBtd93z7mt4fIwDCm3hrrIH8x3ZF0ifkRcBS/g2hvD2aPaxmBIXp4C718eweToP6/FctiBu8nyesBoNipU5KQm/QC2sKysnq0O940Oy5jRCiVqte5GmhZT/S2BPIZUHQvPLLuE=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_3c1b3934d27c11f18460525400393706
    ReservedCode2: ezLUF4hRNuZABqErx+faTjNBIGKLfftvnrpUclwwJA50ZOVvWQYIbGBtd93z7mt4fIwDCm3hrrIH8x3ZF0ifkRcBS/g2hvD2aPaxmBIXp4C718eweToP6/FctiBu8nyesBoNipU5KQm/QC2sKysnq0O940Oy5jRCiVqte5GmhZT/S2BPIZUHQvPLLuE=
---

# tokenizer_rust

[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#许可证)

纯 Rust 实现的分词器（tokenizer）流水线。按 HuggingFace `tokenizers` 的组件语义逐层实现，
训练产物可直接被 `transformers` / `tokenizers` 的 fast tokenizer 加载。

- 四个模型：BPE、WordPiece、WordLevel、Unigram（Viterbi + EM 剪枝）
- 六层组件（normalizer / pre-tokenizer / model / vocabulary / post-processor / decoder）+ L0 门面
- 导出 HF 三件套：`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`
- 命令行工具 `tok`：`train` / `encode` / `decode` / `inspect`

## 快速开始

### 构建

```bash
cargo build --release
# 二进制：target/release/tok
```

要求 Rust 1.75 及以上（见 `Cargo.toml` 的 `rust-version`）。

### 训练

```bash
tok train \
  --corpus corpus/ \
  --out artifacts/tok \
  --model unigram \
  --vocab-size 32000 \
  --normalizer nfkc \
  --special "<unk>" "<s>" "</s>" \
  --recursive \
  --smoke
```

`--model` 可选 `bpe`（默认）/ `wordpiece` / `wordlevel` / `unigram`。
预分词器缺省按模型推断：`unigram` → `metaspace`，其余 → `bytelevel`。
训练结束会在 `--out` 目录写出 `tokenizer.json` / `tokenizer_config.json` /
`special_tokens_map.json`（必要时追加 `config.json`）。

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--vocab-size` | 目标词表规模，默认 30000 |
| `--min-frequency` | 进入合并统计的最小词频，默认 2 |
| `--byte-fallback` | 字节级 BPE 追加 `<0xXX>` 常驻区，8 位任意字节可编码 |
| `--pre-tokenizer` | `bytelevel` / `whitespace` / `whitespace-split` / `bert` / `metaspace` / `digits` / `punctuation` / `none` |
| `--normalizer` | 可重复，按顺序组成 Sequence：`lowercase` / `nfc` / `nfd` / `nfkc` / `nfkd` / `strip-accents` |
| `--max-piece-length` | Unigram 候选 piece 最大字符数，默认 16 |
| `--shrinking-factor` | Unigram 每轮剪枝保留比例，默认 0.75 |
| `--sub-iterations` | Unigram 每个剪枝阶段内的 EM 迭代次数，默认 2 |
| `--hf-model-type` | `config.json` 的 `model_type`：`auto`（默认）/ `none` / 自定义值 |

### 编码 / 解码 / 体检 / 比对

```bash
tok encode --tokenizer artifacts/tok --text "你好，世界" --json
tok decode --tokenizer artifacts/tok --ids "1,2,3"
tok inspect --tokenizer artifacts/tok --head 20
tok compare --left artifacts/tok --right artifacts/tok_new --only-diff --strict
```

`--json` 输出完整结构（含 `offsets` / `special_tokens_mask` / `type_ids`）；
`encode` 支持 `--pair` 启用 pair 模板，`--no-special` 关闭特殊 token 插入。

`compare` 用 `tok-compat::diff_ids` 逐条比对两个 tokenizer 的编码 id，用于
跨版本回归（重构后与旧产物对照）或自洽校验；`--text-file` 指定每行一条的样本文件
（缺省走内置多语言样本，覆盖中文 / emoji / 重音 / 多空白），`--only-diff` 只报不一致，
`--strict` 在存在差异时以非零码退出，便于挂进 CI。

### 对齐基准（HF parity）

`crates/tok-pipeline/tests/hf_parity_golden.rs` 以 **HuggingFace `tokenizers`**
的输出为外部基准，锁定三套典型配置（byte-level BPE / Metaspace Unigram /
WordPiece+Bert）在 `ids` / `tokens` / `offsets` 三个字段上的逐项结果。
WordPiece 一组专锁 **UNK 语义**：未登录词的粒度是"预分词片段"，一个片段整体
退化为单个 `[UNK]`（`"😀🎉🧠"`、`"a😀b"`、`"中文字"` 均须为 1 个 UNK）。
基准还守住三条性质：批量编码与逐条编码逐字段一致、偏移起点在长文本上单调不减、
多字节字符被拆成多片时共享同一字符区间。基准数据为 `tests/data/*.json`，均为最小可复现词表。

## 架构

流水线自下而上分层，实现 crate 之间互不依赖，且不得反向依赖 `tok-pipeline`：

```
tok-pipeline (L0 门面/编排)
   ├── tok-normalizers (L1)   Identity / Lowercase / NFC / NFD / NFKC / NFKD / StripAccents / Replace
   ├── tok-pretok     (L2)    Whitespace / WhitespaceSplit / Bert / ByteLevel / Metaspace / Digits / Punctuation / Split
   ├── tok-models     (L3)    BPE / WordPiece / WordLevel / Unigram
   ├── tok-core       (L4)    词表与公共契约（数据结构、六层 trait、错误类型）
   ├── tok-post       (L5)    TemplateProcessing / BertProcessing / RobertaProcessing / Sequence
   └── tok-decoders   (L6)    ByteLevel / Fuse / WordPiece / Metaspace / BPEDecoder / Replace
tok-trainers          训练器：BpeTrainer / WordPieceTrainer / WordLevelTrainer
tok-compat            HF 三件套的读写与校验
tok-cli               命令行入口 tok
```

| crate | 职责 |
| --- | --- |
| `tok-core` | 公共契约层：数据结构、六层 trait、错误类型、词表 |
| `tok-normalizers` | L1 规范化层 |
| `tok-pretok` | L2 预分词层 |
| `tok-models` | L3 子词切分：BPE / WordPiece / WordLevel / Unigram（Viterbi） |
| `tok-post` | L5 后处理层：模板 / BERT / RoBERTa / Sequence |
| `tok-decoders` | L6 解码层 |
| `tok-trainers` | 训练器与语料读取 |
| `tok-pipeline` | L0 门面：装配、端到端编解码、批处理、截断填充、HF 落盘 |
| `tok-compat` | HF `tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json` 互操作 |
| `tok-cli` | 命令行工具 `tok` |

## 库用法

```rust
use std::path::Path;
use tok_pipeline::load_tokenizer_json;

fn main() -> tok_pipeline::core::Result<()> {
    let tk = load_tokenizer_json(Path::new("artifacts/tok/tokenizer.json"))?;

    // 编码：ids / tokens / offsets / special_tokens_mask / type_ids
    let enc = tk.encode("你好，世界", true)?;
    println!("{:?}", enc.ids);

    // 解码
    let text = tk.decode(&enc.ids, true)?;
    println!("{text}");

    // 批量（内部按 rayon 并行）
    let batch = tk.encode_batch(&["第一条".to_string(), "第二条".to_string()], true)?;
    println!("{}", batch.len());
    Ok(())
}
```

`Tokenizer` 也提供 `from_config` / `from_tokenizer_json`、`encode_pair`、`set_truncation`、
`set_padding`、`save_pretrained` 等方法，详见 `crates/tok-pipeline/src/tokenizer.rs`。

## 与 HuggingFace 的兼容性

落盘文件对齐 `tokenizers` 0.20+ 的 `PreTrainedTokenizerFast` 加载要求：
`model.type`、`vocab`、`merges`（`[["a","b"], ...]`）、`normalizer`、`pre_tokenizer`、
`post_processor`、`decoder`、`added_tokens`；`tokenizer.json` 的 schema 版本为 `1.0`。

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("artifacts/tok")
```

## 开发

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets
cargo test --workspace
```

## 许可证

双许可，任选其一：

- MIT，见 [LICENSE-MIT](LICENSE-MIT)
- Apache License 2.0，见 [LICENSE-APACHE](LICENSE-APACHE)
*（内容由AI生成，仅供参考）*
