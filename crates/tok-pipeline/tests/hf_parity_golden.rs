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
//!
//! 除单文件 `tokenizer.json` 快照外，`tests/data/roles_custom/` 还固化了**自定义角色 +
//! 自动推断后处理**的完整 `save_pretrained` 产物（tokenizer.json +
//! tokenizer_config.json + special_tokens_map.json）。表 `ROLES_CUSTOM` 的期望值由
//! HF `tokenizers` 用 `add_special_tokens=true` 编码得到，锁定角色语义（cls/sep 插入
//! 位置、type_ids、special_tokens_mask）而非仅词汇切分。

use std::path::PathBuf;

use tok_pipeline::{load_pretrained, load_tokenizer_json, Tokenizer};

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

/// WordPiece + BertPreTokenizer（同 `tests/data/wordpiece_bert.json`）。
///
/// 这组基准锁定 **UNK 语义**：WordPiece 对未登录词的粒度是"预分词片段"，
/// 一个片段整体退化为**单个** `[UNK]`，不会按字符/字节散开。文中
/// `"😀🎉🧠 mixed"`、`"a😀b"`、`"中文字"` 是最容易被实现细节写坏的边界：
/// 只要 BertPreTokenizer 的标点判定把 emoji / 汉字误判成标点，预分词就会
/// 在词内切开片段，UNK 数量与偏移随之裂开，与 HF 产生可见偏差。
const WORD_PIECE_BERT: &[Case] = &[
    (
        "hello world",
        &[4, 5],
        &["hello", "world"],
        &[(0, 5), (6, 11)],
    ),
    ("hellozzzz", &[1], &["[UNK]"], &[(0, 9)]),
    ("zzzz yyxx", &[1, 1], &["[UNK]", "[UNK]"], &[(0, 4), (5, 9)]),
    (
        "hello zzzz world",
        &[4, 1, 5],
        &["hello", "[UNK]", "world"],
        &[(0, 5), (6, 10), (11, 16)],
    ),
    (
        "😀🎉🧠 mixed",
        &[1, 1],
        &["[UNK]", "[UNK]"],
        &[(0, 3), (4, 9)],
    ),
    (
        "abc def ghi",
        &[1, 1, 1],
        &["[UNK]", "[UNK]", "[UNK]"],
        &[(0, 3), (4, 7), (8, 11)],
    ),
    ("hello!", &[4, 11], &["hello", "!"], &[(0, 5), (5, 6)]),
    ("中文字", &[1], &["[UNK]"], &[(0, 3)]),
    ("a😀b", &[1], &["[UNK]"], &[(0, 3)]),
    (
        "$100 ¥20 ±3",
        &[1, 1, 1, 1],
        &["[UNK]", "[UNK]", "[UNK]", "[UNK]"],
        &[(0, 1), (1, 4), (5, 8), (9, 11)],
    ),
    (
        "“引号”…—",
        &[1, 1, 1, 1, 1],
        &["[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]"],
        &[(0, 1), (1, 3), (3, 4), (4, 5), (5, 6)],
    ),
    (
        "Hello uns world",
        &[4, 13, 5],
        &["hello", "uns", "world"],
        &[(0, 5), (6, 9), (10, 15)],
    ),
];

/// 自定义角色快照（`tests/data/roles_custom/`）：wordpiece + BertPreTokenizer，
/// 8 个角色全部改写成 `<|...|>` 形态，后处理由 `--post-processor auto` 依据
/// `--role cls/sep` 在训练期自动推断为 `BertProcessing`。
///
/// 期望值来自 HF `tokenizers`：`encode(text, add_special_tokens=True)`。
/// 表中偏移里的 `(0, 0)` 就是模板插入位（HF 对插入 token 恒返回零宽区间）。
const ROLES_CUSTOM: &[Case] = &[
    (
        "hello world 你好",
        &[193, 163, 164, 48, 132, 194],
        &["<|cls|>", "hello", "world", "你", "##好", "<|sep|>"],
        &[(0, 0), (0, 5), (6, 11), (12, 13), (13, 14), (0, 0)],
    ),
    (
        "<|cls|> hello <|sep|> world <|sep|>",
        &[193, 193, 163, 194, 164, 194, 194],
        &[
            "<|cls|>", "<|cls|>", "hello", "<|sep|>", "world", "<|sep|>", "<|sep|>",
        ],
        &[
            (0, 0),
            (0, 7),
            (8, 13),
            (14, 21),
            (22, 27),
            (28, 35),
            (0, 0),
        ],
    ),
    (
        "<|begin|> hello <|end|>",
        &[193, 191, 163, 192, 194],
        &["<|cls|>", "<|begin|>", "hello", "<|end|>", "<|sep|>"],
        &[(0, 0), (0, 9), (10, 15), (16, 23), (0, 0)],
    ),
    (
        "<|mask|> 掩码位置",
        &[193, 195, 189, 194],
        &["<|cls|>", "<|mask|>", "[UNK]", "<|sep|>"],
        &[(0, 0), (0, 8), (9, 13), (0, 0)],
    ),
    (
        "café naïve résumé",
        &[
            193, 12, 85, 103, 100, 2, 85, 123, 104, 80, 9, 100, 87, 94, 93, 100, 194,
        ],
        &[
            "<|cls|>", "c", "##a", "##f", "##é", "n", "##a", "##ï", "##v", "##e", "r", "##é",
            "##s", "##u", "##m", "##é", "<|sep|>",
        ],
        &[
            (0, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (16, 17),
            (0, 0),
        ],
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

#[test]
fn wordpiece_bert_matches_hf_golden() {
    check("wordpiece_bert.json", WORD_PIECE_BERT);
}

/// UNK 语义的性质断言：连续 OOV 片段只产生一个 `[UNK]`，不按字符散开。
///
/// 与上面的表驱动基准互补——表锁死具体值，这里锁死"数量关系"这一语义本身，
/// 即使将来换掉词表数据，该性质也不应失效。
#[test]
fn wordpiece_collapses_oov_span_into_single_unk() {
    let tk: Tokenizer = load_tokenizer_json(&data_path("wordpiece_bert.json")).expect("load");
    for text in ["😀🎉🧠 mixed", "a😀b", "中文字", "hellozzzz"] {
        let enc = tk.encode(text, false).expect("encode");
        let unk_prefix: Vec<usize> = enc
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| t.as_str() == "[UNK]")
            .map(|(i, _)| i)
            .collect();
        // 连续多个 UNK 只允许来自"多个预分词片段"，绝不能来自同一个片段被切开：
        // 因此每个 UNK 的偏移区间必须互不相同。
        let mut ranges: Vec<(usize, usize)> = unk_prefix.iter().map(|&i| enc.offsets[i]).collect();
        ranges.sort_unstable();
        ranges.dedup();
        assert_eq!(
            ranges.len(),
            unk_prefix.len(),
            "{text:?} 的多个 [UNK] 落到了同一字符区间，说明预分词把单个片段切碎了"
        );
        assert!(
            !unk_prefix.is_empty() || text.is_empty(),
            "{text:?} 应至少包含一个 [UNK]"
        );
    }
}

/// 批量编码必须与逐条编码逐字段一致（rayon 并行不改变顺序与偏移）。
#[test]
fn encode_batch_matches_serial_for_both_models() {
    for (file, cases) in [
        ("byte_level_bpe.json", BYTE_LEVEL_BPE),
        ("metaspace_unigram.json", METASPACE_UNIGRAM),
        ("wordpiece_bert.json", WORD_PIECE_BERT),
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

// ---------------------------------------------------------------- 自定义角色快照

fn roles_custom_dir() -> PathBuf {
    data_path("roles_custom")
}

/// 角色语义必须能只凭 `save_pretrained` 的三份文件完整恢复。
///
/// 这是 E6 的回归闸门：一旦 `tokenizer_config.json` / `special_tokens_map.json`
/// 的字段名或优先级改动，`encode_pair` 会静默退化成"无后处理"，此处立刻失败。
#[test]
fn custom_roles_restore_from_pretrained_dir() {
    let tk = load_pretrained(&roles_custom_dir()).expect("按目录加载快照");
    let roles = tk.roles();
    assert_eq!(roles.unk_token.as_deref(), Some("[UNK]"));
    assert_eq!(roles.pad_token.as_deref(), Some("[PAD]"));
    assert_eq!(roles.cls_token.as_deref(), Some("<|cls|>"));
    assert_eq!(roles.sep_token.as_deref(), Some("<|sep|>"));
    assert_eq!(roles.mask_token.as_deref(), Some("<|mask|>"));
    assert_eq!(roles.bos_token.as_deref(), Some("<|begin|>"));
    assert_eq!(roles.eos_token.as_deref(), Some("<|end|>"));
    assert_eq!(
        roles.additional_special_tokens,
        vec!["<|system|>".to_string()],
        "additional 角色必须原样恢复"
    );

    // 角色 token 必须同时满足：在词表里、标记为 special、id 与快照一致。
    // id 由训练期决定，此处是外部快照，不随实现变化。
    let vocab = tk.vocab_map();
    let specials: Vec<&str> = tk
        .added_tokens()
        .iter()
        .filter(|t| t.special)
        .map(|t| t.content.as_str())
        .collect();
    for (tok, id) in [
        ("[UNK]", 189u32),
        ("[PAD]", 190),
        ("<|begin|>", 191),
        ("<|end|>", 192),
        ("<|cls|>", 193),
        ("<|sep|>", 194),
        ("<|mask|>", 195),
        ("<|system|>", 196),
    ] {
        assert_eq!(
            vocab.get(tok).copied(),
            Some(id),
            "{tok} 的 id 与快照不一致"
        );
        assert!(specials.contains(&tok), "{tok} 必须标记为 special");
    }
}

#[test]
fn custom_roles_encode_matches_hf_golden() {
    let tk = load_pretrained(&roles_custom_dir()).expect("按目录加载快照");
    for (text, ids, tokens, offsets) in ROLES_CUSTOM {
        let enc = tk.encode(text, true).expect("encode");
        assert_eq!(&enc.ids[..], *ids, "ids 与 HF 不一致: {text:?}");
        assert_eq!(&enc.tokens[..], *tokens, "tokens 与 HF 不一致: {text:?}");
        assert_eq!(&enc.offsets[..], *offsets, "offsets 与 HF 不一致: {text:?}");
        assert!(enc.validate().is_ok(), "编码结果应通过内部一致性校验");

        // 模板插入的 cls/sep 必须落在首尾，且被标进 special_tokens_mask；
        // 字面出现的角色 token 是否置位取决于实现口径（HF 恒 0），故只校验不变量。
        let n = enc.ids.len();
        assert_eq!(enc.ids[0], 193, "首 token 应是 <|cls|>");
        assert_eq!(enc.ids[n - 1], 194, "尾 token 应是 <|sep|>");
        assert_eq!(enc.special_tokens_mask[0], 1, "首位置必须标 special");
        assert_eq!(enc.special_tokens_mask[n - 1], 1, "尾位置必须标 special");
        assert_eq!(enc.offsets[0], (0, 0), "插入的 cls 应为零宽");
        assert_eq!(enc.offsets[n - 1], (0, 0), "插入的 sep 应为零宽");

        // skip_special 解码必须剔除全部角色 token（OOV 片段会被 [UNK] 吞掉，
        // 故不比较正文，只锁定"角色 token 一定不出现"这一不变量）
        let decoded = tk.decode(&enc.ids, true).expect("decode");
        for tok in [
            "<|cls|>",
            "<|sep|>",
            "<|begin|>",
            "<|end|>",
            "<|mask|>",
            "<|system|>",
            "[UNK]",
            "[PAD]",
        ] {
            assert!(
                !decoded.contains(tok),
                "skip_special 解码仍残留 {tok}: {decoded:?}（原文 {text:?}）"
            );
        }
    }
}

/// pair 编码：cls A sep B sep + type_ids 分段。期望值来自 HF 同参编码。
#[test]
fn custom_roles_encode_pair_matches_hf_golden() {
    let tk = load_pretrained(&roles_custom_dir()).expect("按目录加载快照");
    for (a, b, ids, type_ids) in [
        (
            "hello world",
            "你好 世界",
            &[193u32, 163, 164, 194, 48, 132, 44, 145, 194][..],
            &[0u32, 0, 0, 0, 1, 1, 1, 1, 1][..],
        ),
        (
            "<|cls|> a",
            "b <|sep|>",
            &[193u32, 193, 5, 194, 22, 194, 194][..],
            &[0u32, 0, 0, 0, 1, 1, 1][..],
        ),
    ] {
        let enc = tk.encode_pair(a, b, true).expect("encode_pair");
        assert_eq!(&enc.ids[..], ids, "pair ids 与 HF 不一致: {a:?} / {b:?}");
        assert_eq!(
            &enc.type_ids[..],
            type_ids,
            "pair type_ids 与 HF 不一致: {a:?} / {b:?}"
        );
        assert!(enc.validate().is_ok(), "pair 编码结果应通过一致性校验");
        assert_eq!(enc.sequence_ranges.len(), 2, "pair 必须登记两条序列区间");
    }
}
