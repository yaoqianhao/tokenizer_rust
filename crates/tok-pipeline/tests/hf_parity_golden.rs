//! HF fast tokenizer 对齐基准（golden test）。
//!
//! 表中的 `ids` / `tokens` / `offsets` 全部由 **HuggingFace `tokenizers` 0.23.3**
//! 加载同目录 `tests/data/*.json` 后 `encode(text, add_special_tokens=False)` 得到，
//! 属于外部权威基准而非本实现的自证，用于锁定：
//!
//! * 编码 id 序列（含 byte-level 的字节映射、Metaspace 的前缀 `▁` 语义）；
//! * 偏移坐标（统一为**字符**索引，多字节 UTF-8 字符需正确收敛）；
//! * 覆盖语义（一个原始字符被拆成多片时，多个 token 返回同一字符区间）。
//!
//! 一旦内部实现在这些边界上漂移，本文件会立刻失败——这正是旧版 `convert_offsets`
//! 退化修复期最容易被无声破坏的部分。

use std::path::PathBuf;

use tok_pipeline::{load_tokenizer_json, Tokenizer};

fn data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

/// (文本, ids, tokens, offsets) —— 期望值来自 HF `tokenizers`
type Case = (
    &'static str,
    &'static [u32],
    &'static [&'static str],
    &'static [(usize, usize)],
);

const BYTE_LEVEL_BPE: &[Case] = &[
    (
        "hello world",
        &[259, 220, 264],
        &["hello", "Ġ", "world"],
        &[(0, 5), (5, 6), (6, 11)],
    ),
    (
        "Hello world!",
        &[39, 68, 75, 260, 220, 264, 0],
        &["H", "e", "l", "lo", "Ġ", "world", "!"],
        &[(0, 1), (1, 2), (2, 3), (3, 5), (5, 6), (6, 11), (11, 12)],
    ),
    (
        "你好世界",
        &[160, 121, 254, 161, 98, 121, 160, 116, 244, 163, 243, 234],
        &["ä", "½", "ł", "å", "¥", "½", "ä", "¸", "ĸ", "ç", "ķ", "Į"],
        &[
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 2),
            (1, 2),
            (1, 2),
            (2, 3),
            (2, 3),
            (2, 3),
            (3, 4),
            (3, 4),
            (3, 4),
        ],
    ),
    (
        "你好 world",
        &[160, 121, 254, 161, 98, 121, 220, 264],
        &["ä", "½", "ł", "å", "¥", "½", "Ġ", "world"],
        &[
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 2),
            (1, 2),
            (1, 2),
            (2, 3),
            (3, 8),
        ],
    ),
    (
        "  hello",
        &[220, 220, 259],
        &["Ġ", "Ġ", "hello"],
        &[(0, 1), (1, 2), (2, 7)],
    ),
    (
        "hello你好",
        &[259, 160, 121, 254, 161, 98, 121],
        &["hello", "ä", "½", "ł", "å", "¥", "½"],
        &[(0, 5), (5, 6), (5, 6), (5, 6), (6, 7), (6, 7), (6, 7)],
    ),
    (
        "a b",
        &[64, 220, 65],
        &["a", "Ġ", "b"],
        &[(0, 1), (1, 2), (2, 3)],
    ),
];

const METASPACE_UNIGRAM: &[Case] = &[
    (
        "hello world",
        &[2, 6],
        &["▁hello", "▁world"],
        &[(0, 5), (5, 11)],
    ),
    (
        "Hello world!",
        &[1, 0, 4, 6, 0],
        &["▁", "He", "llo", "▁world", "!"],
        &[(0, 1), (0, 2), (2, 5), (5, 11), (11, 12)],
    ),
    (
        "你好世界",
        &[1, 9, 12],
        &["▁", "你好", "世界"],
        &[(0, 1), (0, 2), (2, 4)],
    ),
    (
        "你好 world",
        &[1, 9, 6],
        &["▁", "你好", "▁world"],
        &[(0, 1), (0, 2), (2, 8)],
    ),
    ("  hello", &[1, 2], &["▁", "▁hello"], &[(0, 1), (1, 7)]),
    ("hello你好", &[2, 9], &["▁hello", "你好"], &[(0, 5), (5, 7)]),
    (
        "a b",
        &[1, 0, 1, 0],
        &["▁", "a", "▁", "b"],
        &[(0, 1), (0, 1), (1, 2), (2, 3)],
    ),
];

fn check(file: &str, cases: &[Case]) {
    let tk: Tokenizer = load_tokenizer_json(&data_path(file)).expect("加载 tokenizer.json");
    for (text, ids, tokens, offsets) in cases {
        let enc = tk.encode(text, false).expect("encode");
        assert_eq!(&enc.ids[..], *ids, "ids 与 HF 不一致: {text:?}");
        assert_eq!(&enc.tokens[..], *tokens, "tokens 与 HF 不一致: {text:?}");
        assert_eq!(&enc.offsets[..], *offsets, "offsets 与 HF 不一致: {text:?}");
        assert!(enc.validate().is_ok(), "编码结果应通过内部一致性校验");
    }
}

#[test]
fn byte_level_bpe_matches_hf_golden() {
    check("byte_level_bpe.json", BYTE_LEVEL_BPE);
}

#[test]
fn metaspace_unigram_matches_hf_golden() {
    check("metaspace_unigram.json", METASPACE_UNIGRAM);
}

/// 批量编码必须与逐条编码逐字段一致（rayon 并行不改变顺序与偏移）。
#[test]
fn encode_batch_matches_serial_for_both_models() {
    for (file, cases) in [
        ("byte_level_bpe.json", BYTE_LEVEL_BPE),
        ("metaspace_unigram.json", METASPACE_UNIGRAM),
    ] {
        let tk: Tokenizer = load_tokenizer_json(&data_path(file)).expect("加载 tokenizer.json");
        let texts: Vec<String> = cases.iter().map(|c| c.0.to_string()).collect();
        let batch = tk.encode_batch(&texts, false).expect("encode_batch");
        assert_eq!(batch.len(), texts.len(), "批量结果条数应与输入一致");
        for (enc, (text, ids, tokens, offsets)) in batch.iter().zip(cases.iter()) {
            assert_eq!(&enc.ids[..], *ids, "批量 ids 不一致: {text:?}");
            assert_eq!(&enc.tokens[..], *tokens, "批量 tokens 不一致: {text:?}");
            assert_eq!(&enc.offsets[..], *offsets, "批量 offsets 不一致: {text:?}");
        }
    }
}

/// 长文本纵向扩展时偏移必须单调不减（回归：曾因游标回退退化为 O(n²) 并错位）。
#[test]
fn offsets_stay_monotonic_on_long_text() {
    let tk: Tokenizer = load_tokenizer_json(&data_path("byte_level_bpe.json")).expect("load");
    let unit = "hello 你好 world ";
    let text = unit.repeat(400);
    let enc = tk.encode(&text, false).expect("encode");
    assert_eq!(enc.ids.len(), enc.offsets.len());
    let mut prev = (0usize, 0usize);
    for &(s, e) in &enc.offsets {
        assert!(s <= e, "偏移区间非法: ({s}, {e})");
        // 覆盖语义允许同区间重复（一个字符拆成多片），但起点不得倒退
        assert!(s >= prev.0, "偏移起点倒退: {s} < {}", prev.0);
        assert!(e <= text.chars().count(), "偏移越界: {e}");
        prev = (s, e);
    }
    assert!(
        enc.offsets.last().map(|o| o.1).unwrap_or(0) <= text.chars().count(),
        "末位偏移应落在原文范围内"
    );
}
