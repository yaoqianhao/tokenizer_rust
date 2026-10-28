//! tok-pipeline 集成测试：装配 → 训练 → 导出 HF 三件套 → 回读 → 端到端一致。
//!
//! 覆盖架构方案的验收点：
//! * CP-1 训练产物可被 `tokenizer.json` 表达（`model.type=BPE`、vocab、merges 齐备）；
//! * CP-2 导出文件可被本实现回读，且**同一文本编码结果逐 id 一致**；
//! * CP-3 截断 / 填充 / pair 模板 / special_tokens_mask 行为符合 HF 语义。

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use tok_core::{
    byte_level::byte_alphabet, AddedToken, Model, PaddingParams, TrainerParams, TruncationParams,
};
use tok_decoders::DecoderConfig;
use tok_models::ModelConfig;
use tok_post::{PostProcessorConfig, SpecialTokenEntry, TemplatePiece};
use tok_pretok::{BehaviorSpec, PreTokenizerConfig, SplitDelimiterBehavior};
use tok_trainers::CorpusReader;

use crate::hf::{self, TokenRoles};
use crate::tokenizer::{bert_like, load_tokenizer_json, PipelineConfig, Tokenizer};

fn temp_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("tok_pipeline_{}_{}", tag, nanos));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------- 用例 1：字节级 BPE 全链路

fn byte_level_config() -> PipelineConfig {
    PipelineConfig {
        normalizer: None,
        pre_tokenizer: Some(PreTokenizerConfig::ByteLevel {
            add_prefix_space: false,
            trim_offsets: true,
            use_regex: true,
        }),
        model: ModelConfig::Bpe {
            dropout: None,
            unk_token: None,
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            fuse_unk: false,
            byte_fallback: false,
            ignore_merges: false,
            vocab: BTreeMap::new(),
            merges: Vec::new(),
        },
        post_processor: None,
        decoder: Some(DecoderConfig::ByteLevel {
            add_prefix_space: false,
            trim_offsets: true,
            use_regex: true,
        }),
        added_tokens: Vec::new(),
        truncation: None,
        padding: None,
        model_max_length: 64,
        roles: TokenRoles::default(),
    }
}

#[test]
fn byte_level_bpe_train_export_and_reload() {
    let dir = temp_dir("bpe");
    let corpus = dir.join("corpus.txt");
    std::fs::write(
        &corpus,
        "hello world\nhello there\nworld peace\nlow lower lowest\nnewer newest\nwide wider\n",
    )
    .unwrap();

    let cfg = byte_level_config();

    let mut tk = Tokenizer::from_config(&cfg).expect("装配失败");
    assert!(tk.model().get_vocab().is_empty(), "训练前词表应为空");

    let mut params = TrainerParams::of(300);
    params.special_tokens = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
    params.initial_alphabet = byte_alphabet();
    params.min_frequency = 1;

    let reader = CorpusReader::new(vec![corpus.clone()]).extensions(vec!["txt".to_string()]);
    let report = tk.train(&reader, params).expect("训练失败");
    assert!(
        report.vocab_size > 256,
        "字节级 BPE 词表应含 256 字节 + 合并产物，实际 {}",
        report.vocab_size
    );
    assert!(report.merges > 0, "应有合并规则产生");

    // 特殊 token 角色可由 added_tokens 推断
    assert_eq!(tk.roles().unk_token.as_deref(), Some("<unk>"));
    assert_eq!(tk.roles().bos_token.as_deref(), Some("<s>"));

    let model_cfg = tk.model_config();
    match &model_cfg {
        ModelConfig::Bpe { vocab, merges, .. } => {
            assert!(!vocab.is_empty());
            assert!(!merges.is_empty());
        }
        other => panic!("模型应为 BPE，实际 {:?}", other),
    }

    // 落盘 HF 三件套
    let out = dir.join("hf");
    let saved = tk.save_pretrained(&out).expect("落盘失败");
    let saved_json = saved.dir.join(hf::TOKENIZER_JSON);
    let saved_cfg = saved.dir.join(hf::TOKENIZER_CONFIG_JSON);
    let saved_map = saved.dir.join(hf::SPECIAL_TOKENS_MAP_JSON);
    assert!(saved_json.is_file());
    assert!(saved_cfg.is_file());
    assert!(saved_map.is_file());

    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&saved_json).unwrap()).unwrap();
    hf::validate_tokenizer_json(&value).expect("tokenizer.json 结构不合法");
    assert_eq!(value["model"]["type"], "BPE");
    assert!(value["model"]["vocab"].as_object().unwrap().len() > 256);
    assert_eq!(
        value["model"]["merges"].as_array().unwrap().len(),
        report.merges
    );
    // merges 采用 HF 0.20+ 的 [["a","b"], ...] 形状
    let first_merge = &value["model"]["merges"][0];
    assert!(first_merge.is_array(), "merges 每项应为二元数组");
    assert_eq!(first_merge.as_array().unwrap().len(), 2);
    assert!(value.get("added_tokens").is_some());
    assert!(value.get("normalizer").is_some());
    assert!(value.get("pre_tokenizer").is_some());
    assert_eq!(value["pre_tokenizer"]["type"], "ByteLevel");
    assert!(value.get("post_processor").is_some());
    assert_eq!(value["decoder"]["type"], "ByteLevel");

    // tokenizer_config.json
    let cfg_value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&saved_cfg).unwrap()).unwrap();
    assert_eq!(cfg_value["tokenizer_class"], "PreTrainedTokenizerFast");
    assert!(cfg_value["added_tokens_decoder"].is_object());
    assert!(cfg_value["added_tokens_decoder"].as_object().unwrap().len() >= 3);
    assert_eq!(cfg_value["model_max_length"], 64);

    // special_tokens_map.json
    let map_value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&saved_map).unwrap()).unwrap();
    assert_eq!(map_value["unk_token"], "<unk>");
    assert_eq!(map_value["bos_token"], "<s>");
    assert_eq!(map_value["eos_token"], "</s>");

    // 回读 → 逐 id 一致（CP-2）
    let tk2 = load_tokenizer_json(&saved_json).expect("回读失败");
    for text in ["hello world", "lowest newer", "world peace"] {
        let a = tk.encode(text, true).unwrap();
        let b = tk2.encode(text, true).unwrap();
        assert_eq!(
            a.ids, b.ids,
            "文本 {:?} 编码不一致: {:?} vs {:?}",
            text, a.ids, b.ids
        );
        assert_eq!(a.tokens, b.tokens);
    }

    // 解码往返（字节级 BPE 对 ASCII 应无损）
    let enc = tk2.encode("hello world", true).unwrap();
    let decoded = tk2.decode(&enc.ids, true).unwrap();
    assert_eq!(decoded, "hello world");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn byte_level_bpe_shares_vocab_with_added_specials() {
    // 空词表 + 预先声明的 added token：训练后 added token 仍须可查 id 且不在 base 区
    let mut cfg = byte_level_config();
    cfg.added_tokens = vec![
        AddedToken::new("<pad>", true),
        AddedToken::new("<mask>", true),
    ];
    cfg.roles.pad_token = Some("<pad>".to_string());
    cfg.roles.mask_token = Some("<mask>".to_string());

    let mut tk = Tokenizer::from_config(&cfg).unwrap();
    let pad_id = tk
        .model()
        .get_vocab()
        .token_to_id("<pad>")
        .expect("added token 应可查 id");
    let mask_id = tk.model().get_vocab().token_to_id("<mask>").unwrap();
    assert_ne!(pad_id, mask_id);

    let dir = temp_dir("bpe_added");
    let corpus = dir.join("c.txt");
    std::fs::write(&corpus, "alpha beta gamma\nbeta gamma delta\n").unwrap();
    let mut params = TrainerParams::of(280);
    params.initial_alphabet = byte_alphabet();
    params.min_frequency = 1;
    let reader = CorpusReader::new(vec![corpus]).extensions(vec!["txt".to_string()]);
    tk.train(&reader, params).unwrap();

    // 训练不得吞掉已登记的 added token。
    // 注意：训练会以 trainer 产物重建 base 词表（256 字节 + 合并产物占据 0..N），
    // added token 的 id 必然被重新分配到 base 之后（与 HF 语义一致：added 恒在 base 区之外），
    // 原 id 不可保留；因此验收点改为「仍可查、落在 base 之外、编号不冲突、回读稳定」。
    let pad_after = tk
        .model()
        .get_vocab()
        .token_to_id("<pad>")
        .expect("<pad> 训练后应仍可查 id");
    let mask_after = tk
        .model()
        .get_vocab()
        .token_to_id("<mask>")
        .expect("<mask> 训练后应仍可查 id");
    assert_ne!(pad_after, mask_after, "added token 编号不得冲突");
    assert!(
        pad_after >= 256,
        "added token 应落在 256 字节 base 区之外，实际 {}",
        pad_after
    );
    assert!(
        tk.special_ids().contains(&pad_after),
        "<pad> 应仍被登记为特殊 token"
    );
    assert_ne!(
        pad_after, pad_id,
        "base 词表重建后 added token 应获得 base 之外的新 id"
    );

    let out = dir.join("hf");
    tk.save_pretrained(&out).unwrap();
    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join(hf::TOKENIZER_JSON)).unwrap())
            .unwrap();
    let added = value["added_tokens"].as_array().unwrap();
    assert!(added.iter().any(|t| t["content"] == "<pad>"));
    assert!(added
        .iter()
        .all(|t| t["special"].as_bool().unwrap_or(false)));

    // 回读后 added token 的 id 必须稳定（导出/回读不重编号）
    let reloaded = load_tokenizer_json(&out.join(hf::TOKENIZER_JSON)).expect("回读失败");
    assert_eq!(
        reloaded.model().get_vocab().token_to_id("<pad>"),
        Some(pad_after),
        "回读后 <pad> 的 id 应与落盘前一致"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------- 用例 2：BERT 风格 WordPiece

fn wordpiece_vocab() -> BTreeMap<String, u32> {
    let mut v = BTreeMap::new();
    for (i, t) in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        .iter()
        .enumerate()
    {
        v.insert(t.to_string(), i as u32);
    }
    v.insert("hello".to_string(), 5);
    v.insert("world".to_string(), 6);
    v.insert("##s".to_string(), 7);
    v.insert("!".to_string(), 8);
    v.insert("token".to_string(), 9);
    v.insert("##izer".to_string(), 10);
    v
}

fn bert_config(with_template: bool) -> PipelineConfig {
    let roles = TokenRoles {
        unk_token: Some("[UNK]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        pad_token: Some("[PAD]".to_string()),
        cls_token: Some("[CLS]".to_string()),
        mask_token: Some("[MASK]".to_string()),
        bos_token: None,
        eos_token: None,
        additional_special_tokens: Vec::new(),
    };
    let mut cfg = bert_like(wordpiece_vocab(), roles);
    if !with_template {
        cfg.post_processor = None;
    }
    cfg
}

#[test]
fn wordpiece_pair_template_and_special_mask() {
    let tk = Tokenizer::from_config(&bert_config(true)).unwrap();

    let enc = tk.encode_pair("hello world", "tokenizer!", true).unwrap();
    assert_eq!(
        enc.ids.first().copied(),
        Some(2),
        "pair 编码应以 [CLS]=2 开头"
    );
    assert_eq!(enc.type_ids.first().copied(), Some(0));
    assert_eq!(enc.type_ids.last().copied(), Some(1));
    assert_eq!(
        enc.ids.iter().filter(|i| **i == 3).count(),
        2,
        "pair 应含两个 [SEP]"
    );
    let mask_sum: u32 = enc.special_tokens_mask.iter().copied().sum();
    assert_eq!(mask_sum, 3, "[CLS] + 2×[SEP] 应为 3 个特殊位");
    assert_eq!(enc.len(), enc.ids.len());
    assert!(enc.validate().is_ok());

    // 单句：一前一后
    let single = tk.encode("hello world", true).unwrap();
    assert_eq!(single.ids.first().copied(), Some(2));
    assert_eq!(single.ids.last().copied(), Some(3));

    // add_special_tokens=false 时应去掉模板特殊 token
    let raw = tk.encode_pair("hello world", "tokenizer!", false).unwrap();
    assert!(!raw.ids.contains(&2));
    assert!(!raw.ids.contains(&3));
    assert_eq!(raw.special_tokens_mask.iter().copied().sum::<u32>(), 0);

    // 未登录词 → [UNK]
    let unknown = tk.encode("zzz", false).unwrap();
    assert_eq!(unknown.ids, vec![1]);
}

#[test]
fn wordpiece_wordpiece_output_shape() {
    let tk = Tokenizer::from_config(&bert_config(false)).unwrap();
    let enc = tk.encode("hello worlds", false).unwrap();
    // "worlds" → world + ##s
    assert!(enc.tokens.contains(&"world".to_string()));
    assert!(enc.tokens.contains(&"##s".to_string()));
    // offsets 落在原文上且单调不减
    let mut last_end = 0usize;
    for (s, e) in &enc.offsets {
        assert!(*e >= *s, "offset 倒挂 {:?}；全部 {:?}", (s, e), enc.offsets);
        assert!(
            *s >= last_end,
            "offset 非单调 {:?}；全部 {:?}",
            (s, e),
            enc.offsets
        );
        last_end = *e;
    }
}

#[test]
fn truncation_and_padding_follow_hf_semantics() {
    let mut tk = Tokenizer::from_config(&bert_config(true)).unwrap();
    tk.set_truncation(Some(TruncationParams {
        max_length: 4,
        ..Default::default()
    }));
    let enc = tk.encode("hello world hello world hello", true).unwrap();
    assert!(
        enc.ids.len() <= 4,
        "截断后长度应 ≤ 4，实际 {}",
        enc.ids.len()
    );
    assert!(!enc.overflowing.is_empty(), "长序列应产生溢出片段");
    assert_eq!(enc.ids.first().copied(), Some(2), "截断须保留 [CLS]");
    assert_eq!(enc.ids.last().copied(), Some(3), "截断须保留 [SEP]");

    tk.set_padding(Some(PaddingParams {
        pad_id: 0,
        pad_token: "[PAD]".to_string(),
        pad_type_id: 0,
        strategy: Default::default(),
        direction: Default::default(),
        pad_to_multiple_of: Some(8),
    }));
    let padded = tk.encode("hello", true).unwrap();
    assert_eq!(padded.ids.len(), 8, "pad_to_multiple_of=8 应对齐到 8");
    assert_eq!(padded.attention_mask.iter().filter(|m| **m == 1).count(), 3);
    assert_eq!(*padded.ids.last().unwrap(), 0, "尾部补 [PAD]=0");
}

// ---------------------------------------------------------------- 用例 3：模板 / 序列化互操作

#[test]
fn template_processing_round_trips_through_json() {
    let cfg = bert_config(true);
    let tk = Tokenizer::from_config(&cfg).unwrap();
    let value = tk.to_tokenizer_json().unwrap();
    let text = serde_json::to_string(&value).unwrap();
    let back: serde_json::Value = serde_json::from_str(&text).unwrap();
    let tk2 = Tokenizer::from_tokenizer_json(&back).unwrap();
    assert_eq!(
        tk.encode("hello world", true).unwrap().ids,
        tk2.encode("hello world", true).unwrap().ids
    );

    // 模板节形状
    let pp = &value["post_processor"];
    assert_eq!(pp["type"], "TemplateProcessing");
    assert!(pp["single"].is_array());
    assert!(pp["pair"].is_array());
    assert_eq!(pp["special_tokens"]["[CLS]"]["ids"][0], 2);
}

#[test]
fn validate_rejects_broken_tokenizer_json() {
    let cfg = bert_config(true);
    let tk = Tokenizer::from_config(&cfg).unwrap();
    let mut value = tk.to_tokenizer_json().unwrap();

    // 去掉 model.vocab 应被拒
    let cloned = value.clone();
    value["model"].as_object_mut().unwrap().remove("vocab");
    assert!(hf::validate_tokenizer_json(&value).is_err());
    assert!(hf::validate_tokenizer_json(&cloned).is_ok());

    // merges 形状非法应被拒
    let mut bad = cloned.clone();
    bad["model"]["merges"] = serde_json::json!([["only-one"]]);
    assert!(hf::validate_tokenizer_json(&bad).is_err());
}

#[test]
fn special_tokens_already_in_vocab_are_tracked() {
    // 训练器会把 special token 直接写进 base 词表（`[UNK]` 等），这类 token 仍需同时
    // 出现在后加视图中：HF 的 `special_ids` / `skip_special_tokens` / `special_tokens_mask`
    // 都读后加视图，漏登记会让 decode(ids, skip_special=true) 输出 "<unk>" 字面。
    let tk = Tokenizer::from_config(&bert_config(false)).unwrap();
    let unk = 1u32; // [UNK]
    let hello = 5u32;
    assert!(
        tk.special_ids().contains(&unk),
        "[UNK] 已在 base 词表，也必须出现在 special_ids 中"
    );
    assert_eq!(tk.decode(&[unk], true).unwrap(), "");
    assert_eq!(tk.decode(&[unk, hello], true).unwrap(), "hello");
    // skip_special_tokens=false 时保留字面（与 HF 一致）
    assert_eq!(tk.decode(&[unk, hello], false).unwrap(), "[UNK] hello");
}

#[test]
fn unk_and_cjk_tokens_keep_real_offsets() {
    // HF 语义：`offsets` 永远指向原文真实区间，特殊/未登录 token 也不例外
    //（只有 special_tokens_mask 标 1）。曾经的实现把 is_special 的 offset 置 (0,0)，
    // 导致 [UNK] 与中文词元的 offsets 全为 0，与 HF 不一致。
    let tk = Tokenizer::from_config(&bert_config(false)).unwrap();

    let enc = tk.encode("hello   world again", true).unwrap();
    let world = enc
        .tokens
        .iter()
        .position(|t| t == "world")
        .expect("缺 world");
    assert_eq!(enc.offsets[world], (8, 13));
    let last = enc.tokens.len() - 1;
    assert_eq!(enc.tokens[last], "[UNK]");
    assert_eq!(enc.offsets[last], (14, 19), "[UNK] 的 offset 应为真实区间");
    assert_eq!(enc.special_tokens_mask[last], 1, "[UNK] 仍须标记为 special");

    let enc = tk.encode("苹果 香蕉", true).unwrap();
    assert!(enc.tokens.iter().all(|t| t == "[UNK]"));
    assert_eq!(enc.offsets[0], (0, 2));
    assert_eq!(enc.offsets[1], (3, 5));
}

#[test]
fn manual_template_processing_builds_pair_section() {
    let mut specials = BTreeMap::new();
    specials.insert(
        "[CLS]".to_string(),
        SpecialTokenEntry {
            id: "[CLS]".to_string(),
            ids: vec![2],
            tokens: vec!["[CLS]".to_string()],
        },
    );
    specials.insert(
        "[SEP]".to_string(),
        SpecialTokenEntry {
            id: "[SEP]".to_string(),
            ids: vec![3],
            tokens: vec!["[SEP]".to_string()],
        },
    );
    let single = vec![
        TemplatePiece::special("[CLS]", 0),
        TemplatePiece::sequence("A", 0),
        TemplatePiece::special("[SEP]", 0),
    ];
    let pair = vec![
        TemplatePiece::special("[CLS]", 0),
        TemplatePiece::sequence("A", 0),
        TemplatePiece::special("[SEP]", 0),
        TemplatePiece::sequence("B", 1),
        TemplatePiece::special("[SEP]", 1),
    ];
    let mut cfg = bert_config(false);
    cfg.post_processor = Some(PostProcessorConfig::TemplateProcessing {
        single,
        pair,
        special_tokens: specials,
    });

    let tk = Tokenizer::from_config(&cfg).unwrap();
    let enc = tk.encode_pair("hello", "world", true).unwrap();
    assert_eq!(enc.ids, vec![2, 5, 3, 6, 3]);
    assert_eq!(enc.type_ids, vec![0, 0, 0, 1, 1]);

    let pp = tk.to_tokenizer_json().unwrap()["post_processor"].clone();
    assert_eq!(pp["pair"].as_array().unwrap().len(), 5);
}

#[test]
fn byte_level_split_delimiter_behaviour_is_configurable() {
    // L2 切分行为：隔离标点（BERT 风格），切分结果与偏移需对齐原文
    let mut cfg = byte_level_config();
    cfg.pre_tokenizer = Some(PreTokenizerConfig::Punctuation {
        behavior: BehaviorSpec::named("isolated"),
    });
    cfg.decoder = None;
    cfg.model = tok_models::byte_level_bpe_config(
        [
            ("a".to_string(), 0u32),
            ("b".to_string(), 1u32),
            ("|".to_string(), 2u32),
        ]
        .into_iter()
        .collect(),
        Vec::new(),
    );

    let tk = Tokenizer::from_config(&cfg).unwrap();
    let enc = tk.encode("a|b", false).unwrap();
    assert_eq!(enc.tokens, vec!["a", "|", "b"]);
    assert_eq!(enc.offsets, vec![(0, 1), (1, 2), (2, 3)]);

    // 明确解析行为枚举，避免“配置静默失效”
    assert_eq!(
        BehaviorSpec::named("isolated").parse().unwrap(),
        SplitDelimiterBehavior::Isolated
    );
    assert!(BehaviorSpec::named("nonsense").parse().is_err());
}
