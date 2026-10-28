//! CLI 子命令实现（train / encode / decode / inspect）。
//!
//! 所有子命令都只通过 `tok-pipeline`(门面) + `tok-compat`(互操作) 触达内核，
//! 不直接拼装 L1..L6 的运行时对象。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use tok_compat::{inventory, TOKENIZER_JSON};
use tok_core::{PaddingParams, Result, TokenizerError, TrainerParams, TruncationParams};
use tok_decoders::DecoderConfig;
use tok_models::ModelConfig;
use tok_normalizers::NormalizerConfig;
use tok_pipeline::{load_tokenizer_json, PipelineConfig, TokenRoles, Tokenizer};
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
    /// 模型类型：bpe | wordpiece | wordlevel | unigram
    #[arg(long, default_value = "bpe")]
    pub model: String,
    /// 预分词器：bytelevel | whitespace | whitespace-split | bert | metaspace | digits | punctuation | none
    /// （缺省时按模型推断：unigram → metaspace，其余 → bytelevel）
    #[arg(long = "pre-tokenizer")]
    pub pre_tokenizer: Option<String>,
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
    /// Unigram：候选 piece 最大字符数（SentencePiece max_piece_length）
    #[arg(long = "max-piece-length", default_value_t = 16)]
    pub max_piece_length: usize,
    /// Unigram：每轮剪枝保留比例（SentencePiece shrinking_factor）
    #[arg(long = "shrinking-factor", default_value_t = 0.75)]
    pub shrinking_factor: f64,
    /// Unigram：每个剪枝阶段内的 EM 迭代次数
    #[arg(long = "sub-iterations", default_value_t = 2)]
    pub sub_iterations: usize,
}

pub fn run_train(args: TrainArgs) -> Result<()> {
    // SentencePiece 风格默认 Metaspace 预分词；其余模型沿用 bytelevel
    let pre_name = args.pre_tokenizer.clone().unwrap_or_else(|| {
        if args.model == "unigram" {
            "metaspace".to_string()
        } else {
            "bytelevel".to_string()
        }
    });
    // Unigram 必须有 unk：未登录字符需要兜底节点，否则 Viterbi 无连通路径
    let mut specials = args.special.clone();
    if args.model == "unigram" && !specials.iter().any(|s| s.to_lowercase().contains("unk")) {
        specials.insert(0, "<unk>".to_string());
    }

    let model = build_model(&args.model, args.byte_fallback, &specials)?;
    let normalizer = build_normalizer(&args.normalizer)?;
    let pre_tokenizer = build_pre_tokenizer(&pre_name)?;
    let decoder = build_decoder(&args.model, &pre_name, args.byte_fallback);

    let cfg = PipelineConfig {
        normalizer,
        pre_tokenizer,
        model,
        post_processor: None,
        decoder,
        added_tokens: specials
            .iter()
            .map(|s| tok_core::AddedToken::new(s.clone(), true))
            .collect(),
        truncation: None,
        padding: None,
        model_max_length: 512,
        roles: TokenRoles::default(),
    };

    let mut tokenizer = Tokenizer::from_config(&cfg)
        .map_err(|e| TokenizerError::config(format!("装配失败（前置配置有误）: {e}")))?;

    let mut params = TrainerParams::of(args.vocab_size);
    params.min_frequency = args.min_frequency;
    params.special_tokens = specials.clone();
    if pre_name == "bytelevel" {
        // 字节级 BPE：词表至少覆盖 256 个字节 + 特殊 token
        params.initial_alphabet = tok_core::byte_level::byte_alphabet();
        let floor = 256 + specials.len() + 8;
        if params.vocab_size < floor {
            params.vocab_size = floor;
        }
    }
    if args.model == "unigram" {
        params.max_piece_length = Some(args.max_piece_length);
        params.shrinking_factor = Some(args.shrinking_factor);
        params.n_sub_iterations = Some(args.sub_iterations);
        // Unigram 的目标规模要能容纳 special + 256 字节 token（byte_fallback 时）
        let floor = specials.len() + if args.byte_fallback { 256 } else { 0 } + 16;
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

    println!(
        "训练完成：{}",
        serde_json::to_string_pretty(&serde_json::to_value(&report)?)?
    );
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

    let vocab_size = value["model"]["vocab"]
        .as_object()
        .map(|o| o.len())
        .unwrap_or(0);
    let merges = value["model"]["merges"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    let added = value["added_tokens"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);

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
            println!(
                "目录产物体检：{}",
                serde_json::to_string_pretty(&json!({
                    "dir": parent,
                    "tokenizer_json": inv.tokenizer_json,
                    "tokenizer_config_json": inv.tokenizer_config_json,
                    "special_tokens_map_json": inv.special_tokens_map_json,
                    "complete": inv.is_complete(),
                    "others": inv.others,
                }))?
            );
        }
    }

    if args.head > 0 {
        if let Some(obj) = value["model"]["vocab"].as_object() {
            let mut pairs: Vec<(&String, u64)> = obj
                .iter()
                .map(|(k, v)| (k, v.as_u64().unwrap_or(u64::MAX)))
                .collect();
            pairs.sort_by_key(|(_, id)| *id);
            for (token, id) in pairs.into_iter().take(args.head) {
                println!("{:>6}  {}", id, token);
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ compare

#[derive(Debug, clap::Args)]
pub struct CompareArgs {
    /// 左侧 tokenizer（目录或 tokenizer.json）
    #[arg(long)]
    pub left: PathBuf,
    /// 右侧 tokenizer（目录或 tokenizer.json）；用于同实现内自洽校验或跨版本对照
    #[arg(long)]
    pub right: PathBuf,
    /// 待比对文本文件（每行一条；缺省用内置样本）
    #[arg(long = "text-file")]
    pub text_file: Option<PathBuf>,
    /// 仅打印不一致条目
    #[arg(long, default_value_t = false)]
    pub only_diff: bool,
    /// 不一致时以非零码退出（CI 用）
    #[arg(long, default_value_t = false)]
    pub strict: bool,
}

pub fn run_compare(args: CompareArgs) -> Result<()> {
    let left = load_tokenizer(&args.left)?;
    let right = load_tokenizer(&args.right)?;
    let texts = match &args.text_file {
        Some(p) => std::fs::read_to_string(p)?
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.to_string())
            .collect::<Vec<_>>(),
        None => default_compare_texts(),
    };
    let diffs = tok_compat::diff_ids(&left, &right, &texts)?;
    let mut mismatched = 0usize;
    for d in &diffs {
        if d.is_match() {
            if !args.only_diff {
                println!("OK   {:?} → {} ids", d.text, d.left.len());
            }
        } else {
            mismatched += 1;
            println!("DIFF {:?}", d.text);
            println!("  left : {:?}", d.left);
            println!("  right: {:?}", d.right);
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "texts": diffs.len(),
            "mismatched": mismatched,
            "matched": diffs.len() - mismatched,
        }))?
    );
    if args.strict && mismatched > 0 {
        return Err(TokenizerError::config(format!(
            "{mismatched} / {} 条文本编码不一致",
            diffs.len()
        )));
    }
    Ok(())
}

/// 内置对照样本：覆盖多语言、emoji、空白与重音，偏重偏移/边界易错点。
fn default_compare_texts() -> Vec<String> {
    [
        "Hello world! 这是一个中文句子。",
        "  leading and trailing  ",
        "emoji 😀🎉 mixed with café naïve résumé",
        "tabs\tand\nnewlines\r\nhere",
        "Hello   world,  多空格  测试",
        "中文English混排test123数字456",
        "The quick brown fox jumps over the lazy dog.",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
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
        // Unigram 无空词表语义：训练起点放一个 unk 占位，训练完成后由 artifact 整体替换
        "unigram" => Ok(ModelConfig::Unigram {
            unk_id: Some(0),
            byte_fallback,
            vocab: vec![(unk.unwrap_or_else(|| "<unk>".to_string()), 0.0)],
        }),
        other => Err(TokenizerError::config(format!(
            "未知 --model: '{}'（可选 bpe / wordpiece / wordlevel / unigram）",
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

fn build_decoder(model: &str, pre: &str, byte_fallback: bool) -> Option<DecoderConfig> {
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
        // SentencePiece 风格：▁ → 空格；若启用 byte_fallback，再把 <0xXX> 还原成字节
        ("unigram", "metaspace") | ("unigram", "none") => {
            let metaspace = DecoderConfig::Metaspace {
                replacement: "\u{2581}".to_string(),
                prepend_scheme: "always".to_string(),
                split: true,
            };
            if byte_fallback {
                Some(DecoderConfig::Sequence {
                    decoders: vec![metaspace, DecoderConfig::ByteFallback],
                })
            } else {
                Some(metaspace)
            }
        }
        ("wordpiece", _) => Some(DecoderConfig::WordPiece {
            prefix: "##".to_string(),
            cleanup: true,
        }),
        (_, "metaspace") => Some(DecoderConfig::Metaspace {
            replacement: "\u{2581}".to_string(),
            prepend_scheme: "always".to_string(),
            split: true,
        }),
        _ => None,
    }
}

/// 预留：把 `--truncation/--padding` 透传进 pipeline（当前仅内部自检使用）。
#[allow(dead_code)]
fn apply_io_params(
    tk: &mut Tokenizer,
    trunc: Option<TruncationParams>,
    pad: Option<PaddingParams>,
) {
    tk.set_truncation(trunc);
    tk.set_padding(pad);
}
