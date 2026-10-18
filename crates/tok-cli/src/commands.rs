//! CLI 子命令实现（train / encode / decode / inspect）。
//!
//! 所有子命令都只通过 `tok-pipeline`(门面) + `tok-compat`(互操作) 触达内核，
//! 不直接拼装 L1..L6 的运行时对象。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use tok_compat::{TOKENIZER_JSON, inventory};
use tok_core::{PaddingParams, Result, TokenizerError, TrainerParams, TruncationParams};
use tok_decoders::DecoderConfig;
use tok_models::ModelConfig;
use tok_normalizers::NormalizerConfig;
use tok_pipeline::{PipelineConfig, TokenRoles, Tokenizer, load_tokenizer_json};
use tok_pretok::{BehaviorSpec, PreTokenizerConfig};
use tok_trainers::CorpusReader;

// ------------------------------------------------------------------ train

#[derive(Debug, clap::Args)]
pub struct TrainArgs {
    /// 语料文件或目录（可多次指定）
    #[arg(long = "corpus", required = true, num_args = 1..)]
    pub corpus: Vec<PathBuf>,
    /// 输出目录（写入 tokenizer.json / tokenizer_config.json / special_tokens_map.json）
    #[arg(long)]
    pub out: PathBuf,
    /// 目标词表规模
    #[arg(long, default_value_t = 30000)]
    pub vocab_size: usize,
    /// 模型类型：bpe | wordpiece | wordlevel
    #[arg(long, default_value = "bpe")]
    pub model: String,
    /// 预分词器：bytelevel | whitespace | whitespace-split | bert | metaspace | digits | punctuation | none
    #[arg(long = "pre-tokenizer", default_value = "bytelevel")]
    pub pre_tokenizer: String,
    /// 归一化器（可重复，按顺序组成 Sequence）：lowercase | nfc | nfd | nfkc | nfkd | strip-accents
    #[arg(long = "normalizer", num_args = 0..)]
    pub normalizer: Vec<String>,
    /// 特殊 token（可重复）；同时作为训练期受保护区，不参与裁剪
    #[arg(long = "special", num_args = 0..)]
    pub special: Vec<String>,
    /// 进入合并统计的最小词频
    #[arg(long, default_value_t = 2)]
    pub min_frequency: u64,
    /// 字节级 BPE 追加 `<0xXX>` 常驻区（8 位任意字节可编码）
    #[arg(long, default_value_t = false)]
    pub byte_fallback: bool,
    /// 语料扩展名过滤（可重复）
    #[arg(long = "ext", num_args = 0.., default_value = "txt")]
    pub ext: Vec<String>,
    /// 递归遍历目录
    #[arg(long, default_value_t = false)]
    pub recursive: bool,
    /// 最多读取的行数（调试用）
    #[arg(long)]
    pub max_lines: Option<usize>,
    /// 训练后立刻做一次 encode 自检并打印结果
    #[arg(long, default_value_t = false)]
    pub smoke: bool,
    /// 训练结束后打印落盘文件清单的 JSON
    #[arg(long, default_value_t = true)]
    pub report_json: bool,
    /// `config.json` 的 model_type：auto（按模型推断）| none（不写）| 自定义值
    #[arg(long = "hf-model-type", default_value = "auto")]
    pub hf_model_type: String,
}

pub fn run_train(args: TrainArgs) -> Result<()> {
    let model = build_model(&args.model, args.byte_fallback, &args.special)?;
    let normalizer = build_normalizer(&args.normalizer)?;
    let pre_tokenizer = build_pre_tokenizer(&args.pre_tokenizer)?;
    let decoder = build_decoder(&args.model, &args.pre_tokenizer);

    let cfg = PipelineConfig {
        normalizer,
        pre_tokenizer,
        model,
        post_processor: None,
        decoder,
        added_tokens: args
            .special
            .iter()
            .map(|s| tok_core::AddedToken::new(s.clone(), true))
            .collect(),
        truncation: None,
        padding: None,
        model_max_length: 512,
        roles: TokenRoles::default(),
    };

    let mut tokenizer = Tokenizer::from_config(&cfg).map_err(|e| {
        TokenizerError::config(format!("装配失败（前置配置有误）: {e}"))
    })?;

    let mut params = TrainerParams::of(args.vocab_size);
    params.min_frequency = args.min_frequency;
    params.special_tokens = args.special.clone();
    if args.pre_tokenizer == "bytelevel" {
        // 字节级 BPE：词表至少覆盖 256 个字节 + 特殊 token
        params.initial_alphabet = tok_core::byte_level::byte_alphabet();
        let floor = 256 + args.special.len() + 8;
        if params.vocab_size < floor {
            params.vocab_size = floor;
        }
    }
    params.byte_fallback = args.byte_fallback;

    let reader = CorpusReader::new(args.corpus.clone())
        .extensions(args.ext.clone())
        .recursive(args.recursive)
        .max_lines(args.max_lines);

    let report = tokenizer.train(&reader, params)?;
    let identity: Option<(&str, &str)> = match args.hf_model_type.as_str() {
        "none" => None,
        "auto" => tokenizer.hf_identity(),
        custom => Some((custom, "PreTrainedTokenizerFast")),
    };
    let files = tokenizer.save_pretrained_with_config(&args.out, identity)?;

    println!("训练完成：{}", serde_json::to_string_pretty(&serde_json::to_value(&report)?)?);
    if args.report_json {
        println!(
            "落盘文件：{}",
            serde_json::to_string_pretty(&json!({
                "dir": files.dir,
                "files": files.files.iter().map(|(p, n)| json!({
                    "path": p, "lines": n,
                })).collect::<Vec<_>>(),
                "total_lines": files.total_lines(),
            }))?
        );
    }

    if args.smoke {
        let sample = "Hello world! 这是一个 tokenizer 冒烟测试。";
        let enc = tokenizer.encode(sample, true)?;
        println!(
            "冒烟自检：文本={:?} ids={:?} tokens={:?}",
            sample, enc.ids, enc.tokens
        );
        let decoded = tokenizer.decode(&enc.ids, true)?;
        println!("冒烟自检：decode(ids)={:?}", decoded);
    }
    Ok(())
}

// ------------------------------------------------------------------ encode

#[derive(Debug, clap::Args)]
pub struct EncodeArgs {
    /// 已训练 tokenizer 目录（含 tokenizer.json）或直接指向 tokenizer.json
    #[arg(long)]
    pub tokenizer: PathBuf,
    /// 待编码文本
    #[arg(long, conflicts_with = "file")]
    pub text: Option<String>,
    /// 待编码文本文件（UTF-8，按行编码）
    #[arg(long)]
    pub file: Option<PathBuf>,
    /// 第二段文本（启用 pair 模板）
    #[arg(long)]
    pub pair: Option<String>,
    /// 不添加特殊 token
    #[arg(long, default_value_t = false)]
    pub no_special: bool,
    /// 输出完整 JSON（含 offsets / special_tokens_mask / type_ids）
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

pub fn run_encode(args: EncodeArgs) -> Result<()> {
    let tokenizer = load_tokenizer(&args.tokenizer)?;
    let add_special = !args.no_special;

    let texts: Vec<String> = match (&args.text, &args.file) {
        (Some(t), None) => vec![t.clone()],
        (None, Some(p)) => std::fs::read_to_string(p)?
            .lines()
            .map(|l| l.to_string())
            .collect(),
        (Some(_), Some(_)) => return Err(TokenizerError::config("--text 与 --file 互斥")),
        (None, None) => return Err(TokenizerError::config("需要 --text 或 --file")),
    };

    let mut out = Vec::new();
    for text in &texts {
        let enc = match &args.pair {
            Some(b) => tokenizer.encode_pair(text, b, add_special)?,
            None => tokenizer.encode(text, add_special)?,
        };
        out.push(json!({
            "text": text,
            "ids": enc.ids,
            "tokens": enc.tokens,
            "offsets": enc.offsets,
            "type_ids": enc.type_ids,
            "attention_mask": enc.attention_mask,
            "special_tokens_mask": enc.special_tokens_mask,
        }));
    }

    if args.json || texts.len() > 1 {
        println!("{}", serde_json::to_string_pretty(&Value::Array(out))?);
    } else {
        let enc = &out[0];
        println!("ids    : {}", enc["ids"]);
        println!("tokens : {}", enc["tokens"]);
        println!("offsets: {}", enc["offsets"]);
    }
    Ok(())
}

// ------------------------------------------------------------------ decode

#[derive(Debug, clap::Args)]
pub struct DecodeArgs {
    /// 已训练 tokenizer 目录或 tokenizer.json 路径
    #[arg(long)]
    pub tokenizer: PathBuf,
    /// 逗号分隔的 id 列表，如 "1,2,3"
    #[arg(long, conflicts_with = "file")]
    pub ids: Option<String>,
    /// 每行一个 id 的文件
    #[arg(long)]
    pub file: Option<PathBuf>,
    /// 跳过特殊 token
    #[arg(long, default_value_t = true)]
    pub skip_special: bool,
}

pub fn run_decode(args: DecodeArgs) -> Result<()> {
    let tokenizer = load_tokenizer(&args.tokenizer)?;
    let ids = match (&args.ids, &args.file) {
        (Some(s), None) => parse_ids(s)?,
        (None, Some(p)) => {
            let mut v = Vec::new();
            for line in std::fs::read_to_string(p)?.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                v.extend(parse_ids(line)?);
            }
            v
        }
        (Some(_), Some(_)) => return Err(TokenizerError::config("--ids 与 --file 互斥")),
        (None, None) => return Err(TokenizerError::config("需要 --ids 或 --file")),
    };
    let text = tokenizer.decode(&ids, args.skip_special)?;
    println!("{text}");
    Ok(())
}

fn parse_ids(raw: &str) -> Result<Vec<u32>> {
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u32>()
                .map_err(|_| TokenizerError::invalid_input(format!("非法 id: '{}'", s)))
        })
        .collect()
}

// ------------------------------------------------------------------ inspect

#[derive(Debug, clap::Args)]
pub struct InspectArgs {
    /// tokenizer 目录或 tokenizer.json 路径
    #[arg(long)]
    pub tokenizer: PathBuf,
    /// 打印词表前 N 个 token（0 表示不打印）
    #[arg(long, default_value_t = 0)]
    pub head: usize,
}

pub fn run_inspect(args: InspectArgs) -> Result<()> {
    let path = resolve_tokenizer_path(&args.tokenizer);
    let raw = std::fs::read_to_string(&path)?;
    let value: Value = serde_json::from_str(&raw)?;
    tok_pipeline::validate_tokenizer_json(&value)?;

    let vocab_size = value["model"]["vocab"].as_object().map(|o| o.len()).unwrap_or(0);
    let merges = value["model"]["merges"].as_array().map(|a| a.len()).unwrap_or(0);
    let added = value["added_tokens"].as_array().map(|a| a.len()).unwrap_or(0);

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "path": path,
            "version": value["version"],
            "model_type": value["model"]["type"],
            "vocab_size": vocab_size,
            "merges": merges,
            "added_tokens": added,
            "has_normalizer": !value["normalizer"].is_null(),
            "has_pre_tokenizer": !value["pre_tokenizer"].is_null(),
            "has_post_processor": !value["post_processor"].is_null(),
            "has_decoder": !value["decoder"].is_null(),
            "added_tokens_sample": value["added_tokens"].as_array().map(|a| a.iter().take(5).cloned().collect::<Vec<_>>()),
        }))?
    );

    if let Some(parent) = path.parent() {
        if let Ok(inv) = inventory(parent) {
            println!("目录产物体检：{}", serde_json::to_string_pretty(&json!({
                "dir": parent,
                "tokenizer_json": inv.tokenizer_json,
                "tokenizer_config_json": inv.tokenizer_config_json,
                "special_tokens_map_json": inv.special_tokens_map_json,
                "complete": inv.is_complete(),
                "others": inv.others,
            }))?);
        }
    }

    if args.head > 0 {
        if let Some(obj) = value["model"]["vocab"].as_object() {
            let mut pairs: Vec<(&String, u64)> =
                obj.iter().map(|(k, v)| (k, v.as_u64().unwrap_or(u64::MAX))).collect();
            pairs.sort_by_key(|(_, id)| *id);
            for (token, id) in pairs.into_iter().take(args.head) {
                println!("{:>6}  {}", id, token);
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ 辅助

fn resolve_tokenizer_path(p: &Path) -> PathBuf {
    if p.is_dir() {
        p.join(TOKENIZER_JSON)
    } else {
        p.to_path_buf()
    }
}

fn load_tokenizer(p: &Path) -> Result<Tokenizer> {
    let path = resolve_tokenizer_path(p);
    load_tokenizer_json(&path)
}

fn build_model(kind: &str, byte_fallback: bool, specials: &[String]) -> Result<ModelConfig> {
    let unk = specials
        .iter()
        .find(|s| s.to_lowercase().contains("unk"))
        .cloned();
    match kind {
        "bpe" => Ok(ModelConfig::Bpe {
            dropout: None,
            unk_token: unk,
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            fuse_unk: false,
            byte_fallback,
            ignore_merges: false,
            vocab: BTreeMap::new(),
            merges: Vec::new(),
        }),
        "wordpiece" => Ok(ModelConfig::WordPiece {
            unk_token: unk.unwrap_or_else(|| "[UNK]".to_string()),
            continuing_subword_prefix: "##".to_string(),
            max_input_chars_per_word: 100,
            vocab: BTreeMap::new(),
        }),
        "wordlevel" => Ok(ModelConfig::WordLevel {
            unk_token: unk.unwrap_or_else(|| "[UNK]".to_string()),
            vocab: BTreeMap::new(),
        }),
        other => Err(TokenizerError::config(format!(
            "未知 --model: '{}'（可选 bpe / wordpiece / wordlevel）",
            other
        ))),
    }
}

fn build_normalizer(names: &[String]) -> Result<Option<NormalizerConfig>> {
    let mut parts = Vec::new();
    for n in names {
        parts.push(match n.as_str() {
            "identity" | "none" => NormalizerConfig::Identity,
            "lowercase" => NormalizerConfig::Lowercase,
            "nfc" => NormalizerConfig::Nfc,
            "nfd" => NormalizerConfig::Nfd,
            "nfkc" => NormalizerConfig::Nfkc,
            "nfkd" => NormalizerConfig::Nfkd,
            "strip-accents" => NormalizerConfig::StripAccents,
            other => {
                return Err(TokenizerError::config(format!(
                    "未知 --normalizer: '{}'（可选 lowercase / nfc / nfd / nfkc / nfkd / strip-accents）",
                    other
                )));
            }
        });
    }
    Ok(match parts.len() {
        0 => None,
        1 => parts.pop(),
        _ => Some(NormalizerConfig::Sequence { normalizers: parts }),
    })
}

fn build_pre_tokenizer(kind: &str) -> Result<Option<PreTokenizerConfig>> {
    Ok(match kind {
        "none" => None,
        "bytelevel" => Some(PreTokenizerConfig::ByteLevel {
            add_prefix_space: false,
            trim_offsets: true,
            use_regex: true,
        }),
        "whitespace" => Some(PreTokenizerConfig::Whitespace),
        "whitespace-split" => Some(PreTokenizerConfig::WhitespaceSplit),
        "bert" => Some(PreTokenizerConfig::BertPreTokenizer),
        "metaspace" => Some(PreTokenizerConfig::Metaspace {
            replacement: "\u{2581}".to_string(),
            add_prefix_space: true,
            prepend_scheme: Some("always".to_string()),
        }),
        "digits" => Some(PreTokenizerConfig::Digits {
            individual_digits: false,
        }),
        "punctuation" => Some(PreTokenizerConfig::Punctuation {
            behavior: BehaviorSpec::named("isolated"),
        }),
        other => {
            return Err(TokenizerError::config(format!(
                "未知 --pre-tokenizer: '{}'",
                other
            )));
        }
    })
}

fn build_decoder(model: &str, pre: &str) -> Option<DecoderConfig> {
    match (model, pre) {
        ("bpe", "bytelevel") => Some(DecoderConfig::ByteLevel {
            add_prefix_space: false,
            trim_offsets: true,
            use_regex: true,
        }),
        ("bpe", "metaspace") => Some(DecoderConfig::Metaspace {
            replacement: "\u{2581}".to_string(),
            prepend_scheme: "always".to_string(),
            split: true,
        }),
        ("wordpiece", _) => Some(DecoderConfig::WordPiece {
            prefix: "##".to_string(),
            cleanup: true,
        }),
        _ => None,
    }
}

/// 预留：把 `--truncation/--padding` 透传进 pipeline（当前仅内部自检使用）。
#[allow(dead_code)]
fn apply_io_params(tk: &mut Tokenizer, trunc: Option<TruncationParams>, pad: Option<PaddingParams>) {
    tk.set_truncation(trunc);
    tk.set_padding(pad);
}
