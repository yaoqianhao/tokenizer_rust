//! CLI 子命令实现（train / encode / decode / inspect）。
//!
//! 所有子命令都只通过 `tok-pipeline`(门面) + `tok-compat`(互操作) 触达内核，
//! 不直接拼装 L1..L6 的运行时对象。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use tok_compat::{inventory, TOKENIZER_JSON};
use tok_core::{
    Encoding, PaddingDirection, PaddingParams, PaddingStrategy, Result, TokenizerError,
    TrainerParams, TruncationDirection, TruncationParams, TruncationStrategy,
};
use tok_decoders::DecoderConfig;
use tok_models::ModelConfig;
use tok_normalizers::NormalizerConfig;
use tok_pipeline::{load_tokenizer_json, PipelineConfig, TokenRoles, Tokenizer};
use tok_post::{PostProcessorConfig, TemplatePiece};
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
    /// 预分词器：bytelevel | whitespace | whitespace-split | bert | metaspace | digits | punctuation | none；
    /// 组合/带参写法：`sequence(a,b)`、`split:pattern=<regex>;behavior=isolated;invert=false`、
    /// `bytelevel:add_prefix_space=true;trim_offsets=true;use_regex=true`
    /// （缺省时按模型推断：unigram → metaspace，其余 → bytelevel）
    #[arg(long = "pre-tokenizer")]
    pub pre_tokenizer: Option<String>,
    /// 归一化器（可重复，按顺序组成 Sequence）：lowercase | nfc | nfd | nfkc | nfkd | strip-accents |
    /// strip | `replace:pattern=<regex>;content=<s>` | `prepend:prepend=<s>`
    #[arg(long = "normalizer", num_args = 0..)]
    pub normalizer: Vec<String>,
    /// 特殊 token（可重复）；同时作为训练期受保护区，不参与裁剪
    #[arg(long = "special", num_args = 0..)]
    pub special: Vec<String>,
    /// 角色 token（可重复，`name=token`）：unk | bos | eos | pad | cls | sep | mask | additional。
    /// 声明后写入 `tokenizer_config.json` 的 `*_token` 与 `special_tokens_map.json`，
    /// 并自动登记为受保护的 special added token（`special_tokens_mask=1`、`skip_special_tokens` 生效）。
    /// 例：`--role cls=[CLS] --role sep=[SEP] --role additional=<|system|>`
    #[arg(long = "role", num_args = 0..)]
    pub role: Vec<String>,
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
    /// 后处理器：auto（默认，按 --role/--cls/--sep 显式声明推断）| none | bert | roberta |
    /// bert-template | `template:<single>[|<pair>]`（`$A`/`$B` 为序列占位，其余按特殊 token 字面值）
    #[arg(long = "post-processor", default_value = "auto")]
    pub post_processor: String,
    /// 后处理模板用的 CLS 字面值（缺省按模型推断：[CLS] 或 <s>）
    #[arg(long)]
    pub cls: Option<String>,
    /// 后处理模板用的 SEP 字面值（缺省按模型推断：[SEP] 或 </s>）
    #[arg(long)]
    pub sep: Option<String>,
    /// 解码器：auto（默认，按模型/预分词推断）| none | 逗号分隔的 Sequence |
    /// 元素形如 bytelevel / wordpiece / metaspace / bytefallback / bpe / strip，支持 `name:k=v` 传参；
    /// `fuse(a,b)` 为分组语法糖，导出成 HF 的 `Sequence`（HF 的 `Fuse` 不承载子解码器）
    #[arg(long = "decoder", default_value = "auto")]
    pub decoder: String,
    /// 截断：最终序列最大长度（缺省不截断）
    #[arg(long = "truncation-max-length")]
    pub truncation_max_length: Option<usize>,
    /// 截断策略：longest-first（默认）| only-first | only-second
    #[arg(long = "truncation-strategy", default_value = "longest-first")]
    pub truncation_strategy: String,
    /// 截断方向：right（默认）| left
    #[arg(long = "truncation-direction", default_value = "right")]
    pub truncation_direction: String,
    /// 截断重叠步长（HF `stride`）
    #[arg(long, default_value_t = 0)]
    pub stride: usize,
    /// 填充：none（默认）| batch-longest | fixed:<N>
    #[arg(long, default_value = "none")]
    pub padding: String,
    /// 填充方向：right（默认）| left
    #[arg(long = "padding-direction", default_value = "right")]
    pub padding_direction: String,
    /// 填充 token 字面值（缺省在 [PAD]/<pad>/[pad] 中取词表已有的那个）
    #[arg(long = "pad-token")]
    pub pad_token: Option<String>,
    /// 填充 token id（缺省按 pad-token 在训练后词表里的真实 id 解析）
    #[arg(long = "pad-id")]
    pub pad_id: Option<u32>,
    /// 填充长度向上取整的倍数（HF `pad_to_multiple_of`）
    #[arg(long = "pad-to-multiple-of")]
    pub pad_to_multiple_of: Option<usize>,
    /// `model_max_length`（写入 tokenizer_config.json，并作为填充长度上限）
    #[arg(long = "model-max-length", default_value_t = 512)]
    pub model_max_length: usize,
}

pub fn run_train(args: TrainArgs) -> Result<()> {
    // SentencePiece 风格默认 Metaspace 预分词；其余模型沿用 bytelevel
    let pre_spec = args.pre_tokenizer.clone().unwrap_or_else(|| {
        if args.model == "unigram" {
            "metaspace".to_string()
        } else {
            "bytelevel".to_string()
        }
    });
    // 推断用基名：`sequence(split:...,bytelevel)` 这类组合取首节点名
    let pre_base = pretok_base_name(&pre_spec);
    // 组合里出现 bytelevel 时同样需要 256 字节常驻区
    let has_bytelevel = pre_spec.contains("bytelevel");
    // Unigram 必须有 unk：未登录字符需要兜底节点，否则 Viterbi 无连通路径
    let roles = parse_roles(&args.role)?;
    let mut specials = args.special.clone();
    // 角色 token 必须进训练期词表并受保护，否则会被裁剪或被模型切碎
    for t in roles.all() {
        if !specials.iter().any(|x| x == &t) {
            specials.push(t);
        }
    }
    if args.model == "unigram" && !specials.iter().any(|s| s.to_lowercase().contains("unk")) {
        specials.insert(0, "<unk>".to_string());
    }

    let model = build_model(&args.model, args.byte_fallback, &specials)?;
    let normalizer = build_normalizers(&args.normalizer)?;
    let pre_tokenizer = build_pre_tokenizer(&pre_spec)?;
    let decoder = if args.decoder == "auto" {
        let key = if has_bytelevel {
            "bytelevel"
        } else {
            pre_base.as_str()
        };
        build_decoder(&args.model, key, args.byte_fallback)
    } else {
        parse_decoder_spec(&args.decoder, args.byte_fallback)?
    };
    // 后处理器在训练后替换，但模板引用的特殊 token 必须先进入训练期词表
    let post_cfg = if args.post_processor.trim() == "auto" {
        // 自动推断：只依据用户**显式声明**的角色（--role / --cls / --sep），
        // 不拿模型默认字面值去猜，避免凭空改写既有产物的 ids。
        infer_post_processor(
            &args.model,
            &roles,
            args.cls.as_deref(),
            args.sep.as_deref(),
        )
    } else {
        parse_post_processor_spec(
            &args.post_processor,
            &args.model,
            args.cls.as_deref(),
            args.sep.as_deref(),
        )?
    };
    if let Some(cfg) = &post_cfg {
        for s in tok_post::PostProcessorWrapper::build(cfg)?.required_specials() {
            if !specials.iter().any(|x| x == &s) {
                specials.push(s);
            }
        }
    }

    let cfg = PipelineConfig {
        normalizer,
        pre_tokenizer,
        model,
        post_processor: None,
        decoder: decoder.clone(),
        added_tokens: specials
            .iter()
            .map(|s| tok_core::AddedToken::new(s.clone(), true))
            .collect(),
        truncation: None,
        padding: None,
        model_max_length: args.model_max_length,
        roles: roles.clone(),
    };

    let mut tokenizer = Tokenizer::from_config(&cfg)
        .map_err(|e| TokenizerError::config(format!("装配失败（前置配置有误）: {e}")))?;

    let mut params = TrainerParams::of(args.vocab_size);
    params.min_frequency = args.min_frequency;
    params.special_tokens = specials.clone();
    if has_bytelevel {
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
    // E1/E2：训练产物落盘前按 CLI 参数替换后处理与解码器（词表已就绪，可精确绑定 id）
    if post_cfg.is_some() {
        tokenizer.set_post_processor(post_cfg)?;
    }
    if args.decoder != "auto" {
        tokenizer.set_decoder(decoder.clone())?;
    }
    // E5：截断/填充在训练后按 CLI 参数装配（pad id 需要训练完成后的真实词表）
    let trunc = parse_truncation(
        args.truncation_max_length,
        &args.truncation_strategy,
        &args.truncation_direction,
        args.stride,
    )?;
    let pad = parse_padding(
        &args.padding,
        &args.padding_direction,
        args.pad_token.as_deref(),
        args.pad_id,
        args.pad_to_multiple_of,
        Some(&tokenizer.vocab_map()),
        &specials,
    )?;
    apply_io_params(&mut tokenizer, trunc, pad);
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
    /// 编码期截断：覆盖 tokenizer.json 里的 max_length（缺省沿用文件配置）
    #[arg(long = "max-length")]
    pub max_length: Option<usize>,
    /// 编码期填充：none | batch-longest | fixed:<N>（覆盖 tokenizer.json 里的 padding）
    #[arg(long)]
    pub padding: Option<String>,
    /// 填充 token 字面值（缺省沿用词表里的 [PAD]/<pad>）
    #[arg(long = "pad-token")]
    pub pad_token: Option<String>,
    /// 填充 token id（缺省按 pad-token 的真实词表 id 解析）
    #[arg(long = "pad-id")]
    pub pad_id: Option<u32>,
    /// 填充长度向上取整的倍数
    #[arg(long = "pad-to-multiple-of")]
    pub pad_to_multiple_of: Option<usize>,
}

pub fn run_encode(args: EncodeArgs) -> Result<()> {
    let mut tokenizer = load_tokenizer(&args.tokenizer)?;
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

    // 编码期覆盖：--max-length 只改长度，策略/方向/stride 沿用文件里已配置的值
    if let Some(max_length) = args.max_length {
        let base = tokenizer.config().truncation.clone().unwrap_or_default();
        tokenizer.set_truncation(Some(TruncationParams { max_length, ..base }));
    }
    // 批量填充：先按文件配置编码，再按**整批**最长补 pad（HF `padding=True` 语义）
    let padding = match &args.padding {
        Some(spec) => parse_padding(
            spec,
            &tokenizer
                .config()
                .padding
                .as_ref()
                .map(|p| match p.direction {
                    PaddingDirection::Left => "left".to_string(),
                    PaddingDirection::Right => "right".to_string(),
                })
                .unwrap_or_else(|| "right".to_string()),
            args.pad_token.as_deref(),
            args.pad_id,
            args.pad_to_multiple_of,
            Some(&tokenizer.vocab_map()),
            &[],
        )?,
        None => None,
    };
    let batch_longest = matches!(
        padding.as_ref().map(|p| p.strategy),
        Some(PaddingStrategy::BatchLongest)
    );
    if !batch_longest {
        // fixed / 文件内既定策略：交给 pipeline 逐条应用
        if let Some(p) = &padding {
            tokenizer.set_padding(Some(p.clone()));
        }
    } else {
        tokenizer.set_padding(None);
    }

    let mut encodings: Vec<Encoding> = Vec::with_capacity(texts.len());
    for text in &texts {
        let enc = match &args.pair {
            Some(b) => tokenizer.encode_pair(text, b, add_special)?,
            None => tokenizer.encode(text, add_special)?,
        };
        encodings.push(enc);
    }
    if batch_longest {
        let params = padding.expect("batch_longest 时 padding 必定存在");
        let target = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        for enc in &mut encodings {
            enc.pad(&params, target);
        }
    }

    let out: Vec<Value> = texts
        .iter()
        .zip(encodings.iter())
        .map(|(text, enc)| {
            json!({
                "text": text,
                "ids": enc.ids,
                "tokens": enc.tokens,
                "offsets": enc.offsets,
                "type_ids": enc.type_ids,
                "attention_mask": enc.attention_mask,
                "special_tokens_mask": enc.special_tokens_mask,
            })
        })
        .collect();

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
    // 分隔符宽容：兼容 `1,2,3`、`1 2 3`、`[1, 2, 3]` 以及混用空白/逗号的粘贴形态，
    // 手抄 ids 时最常见的失败就是"把空格串当逗号串"。方括号仅在整串首尾成对时剥掉。
    let body = raw
        .trim()
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(raw)
        .trim();
    body.split([',', ' ', '\t', '\n', '\r'])
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

// ------------------------------------------------------------------ eval

#[derive(Debug, clap::Args)]
pub struct EvalArgs {
    /// 已训练 tokenizer 目录（含 tokenizer.json）或直接指向 tokenizer.json
    #[arg(long)]
    pub tokenizer: PathBuf,
    /// 评测语料文件或目录（可多次指定）
    #[arg(long = "corpus", required = true, num_args = 1..)]
    pub corpus: Vec<PathBuf>,
    /// 语料扩展名过滤（可重复）
    #[arg(long = "ext", num_args = 0.., default_value = "txt")]
    pub ext: Vec<String>,
    /// 递归遍历目录
    #[arg(long, default_value_t = false)]
    pub recursive: bool,
    /// 最多读取的行数（调试用）
    #[arg(long)]
    pub max_lines: Option<usize>,
    /// 编码时不插入后处理特殊 token
    #[arg(long = "no-special", default_value_t = false)]
    pub no_special: bool,
    /// 以 JSON 输出指标
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

/// 语料级评测：分词效率（tok/char）、UNK 率、round-trip 还原率与吞吐。
///
/// 这三项是替换实验（后处理 / 解码器 / 预分词）的唯一量化口径：
/// 同一个语料、同一批参数下逐项对比，才能判断某次替换是"更省 token"还是"更丢信息"。
pub fn run_eval(args: EvalArgs) -> Result<()> {
    let tokenizer = load_tokenizer(&args.tokenizer)?;
    let reader = CorpusReader::new(args.corpus.clone())
        .extensions(args.ext.clone())
        .recursive(args.recursive)
        .max_lines(args.max_lines);
    let data = reader.read()?;
    let vocab = tokenizer.vocab_map();
    let unk_id = tokenizer
        .roles()
        .unk_token
        .as_ref()
        .and_then(|t| vocab.get(t).copied());

    let add_special = !args.no_special;
    let started = std::time::Instant::now();
    let mut tokens = 0usize;
    let mut chars = 0usize;
    let mut unk = 0usize;
    let mut roundtrip_ok = 0usize;
    let mut roundtrip_squash_ok = 0usize;
    let mut max_tokens_line = 0usize;
    for line in &data.lines {
        let enc = tokenizer.encode(line, add_special)?;
        tokens += enc.ids.len();
        chars += line.chars().count();
        max_tokens_line = max_tokens_line.max(enc.ids.len());
        if let Some(u) = unk_id {
            unk += enc.ids.iter().filter(|id| **id == u).count();
        }
        let decoded = tokenizer.decode(&enc.ids, true)?;
        if decoded == *line {
            roundtrip_ok += 1;
        }
        // 忽略空白差异后再比一次：WordPiece / Metaspace 的空白语义本就不可逆，
        // 抹掉空白仍不等才说明内容真的丢了（UNK 替换、byte 未还原）。
        if squash_whitespace(&decoded) == squash_whitespace(line) {
            roundtrip_squash_ok += 1;
        }
    }
    let elapsed = started.elapsed().as_secs_f64().max(1e-9);
    let lines = data.lines.len();

    let ratio = |num: usize, den: usize| -> f64 {
        if den == 0 {
            0.0
        } else {
            ((num as f64 / den as f64) * 1e6).round() / 1e6
        }
    };
    let per_sec = |num: usize| -> f64 { ((num as f64 / elapsed) * 100.0).round() / 100.0 };

    let metrics = json!({
        "tokenizer": args.tokenizer.display().to_string(),
        "add_special_tokens": add_special,
        "lines": lines,
        "tokens": tokens,
        "chars": chars,
        "tokens_per_char": ratio(tokens, chars),
        "chars_per_token": ratio(chars, tokens),
        "max_tokens_per_line": max_tokens_line,
        "unk_id": unk_id,
        "unk_tokens": unk,
        "unk_rate": ratio(unk, tokens),
        "roundtrip_lines": roundtrip_ok,
        "roundtrip_rate": ratio(roundtrip_ok, lines),
        "roundtrip_squash_lines": roundtrip_squash_ok,
        "roundtrip_squash_rate": ratio(roundtrip_squash_ok, lines),
        "elapsed_secs": (elapsed * 1000.0).round() / 1000.0,
        "lines_per_sec": per_sec(lines),
        "tokens_per_sec": per_sec(tokens),
    });
    if args.json {
        println!("{}", serde_json::to_string_pretty(&metrics)?);
    } else {
        println!("语料级评测：{}", args.tokenizer.display());
        println!(
            "  行数={lines}  token 数={tokens}  字符数={chars}  tok/char={}  char/tok={}",
            ratio(tokens, chars),
            ratio(chars, tokens)
        );
        println!(
            "  UNK: id={:?} 次数={unk} 占比={}",
            unk_id,
            ratio(unk, tokens)
        );
        println!(
            "  round-trip: {roundtrip_ok}/{lines} = {}（含后处理特殊 token 的精确还原率）",
            ratio(roundtrip_ok, lines)
        );
        println!(
            "  round-trip(忽略空白): {roundtrip_squash_ok}/{lines} = {}（抹掉空白后仍不等即内容丢失）",
            ratio(roundtrip_squash_ok, lines)
        );
        println!(
            "  最长行 token 数={max_tokens_line}  吞吐={} 行/s {} token/s  耗时={:.3}s",
            per_sec(lines),
            per_sec(tokens),
            elapsed
        );
    }
    Ok(())
}

/// 丢弃全部空白字符，用于「忽略空白」的 round-trip 比对。
fn squash_whitespace(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
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
    // E6：目录形态一律走 load_pretrained，把 special_tokens_map.json /
    // tokenizer_config.json 里的角色语义一起恢复；单文件仍按纯 tokenizer.json 读取。
    if p.is_dir() {
        return tok_pipeline::load_pretrained(p);
    }
    load_tokenizer_json(&resolve_tokenizer_path(p))
}

/// 解析 `--role name=token`（E6）。非法名字 / 空 token / 同一 token 占两个角色都在此拦截，
/// 避免把不自洽的角色表写进 `tokenizer_config.json` 让 HF 侧静默错位。
fn parse_roles(specs: &[String]) -> Result<TokenRoles> {
    let mut roles = TokenRoles::default();
    for spec in specs {
        let (name, tok) = spec.split_once('=').ok_or_else(|| {
            TokenizerError::config(format!(
                "--role 需写成 `name=token`，实际收到 `{spec}`（例：--role cls=[CLS]）"
            ))
        })?;
        let name = name.trim().to_ascii_lowercase();
        let tok = tok.trim();
        if tok.is_empty() {
            return Err(TokenizerError::config(format!(
                "--role {name}= 的 token 不能为空"
            )));
        }
        let tok = tok.to_string();
        match name.as_str() {
            "unk" => roles.unk_token = Some(tok),
            "bos" => roles.bos_token = Some(tok),
            "eos" => roles.eos_token = Some(tok),
            "pad" => roles.pad_token = Some(tok),
            "cls" => roles.cls_token = Some(tok),
            "sep" => roles.sep_token = Some(tok),
            "mask" => roles.mask_token = Some(tok),
            "additional" | "extra" => {
                if !roles.additional_special_tokens.contains(&tok) {
                    roles.additional_special_tokens.push(tok);
                }
            }
            other => {
                return Err(TokenizerError::config(format!(
                    "未知角色名 `{other}`；可用：unk|bos|eos|pad|cls|sep|mask|additional"
                )))
            }
        }
    }
    roles.validate()?;
    Ok(roles)
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

fn build_normalizers(names: &[String]) -> Result<Option<NormalizerConfig>> {
    let mut parts = Vec::new();
    for n in names {
        parts.push(parse_normalizer_node(n)?);
    }
    Ok(match parts.len() {
        0 => None,
        1 => parts.pop(),
        _ => Some(NormalizerConfig::Sequence { normalizers: parts }),
    })
}

/// 单个归一化节点：`name` 或 `name:k=v;k2=v2`。
fn parse_normalizer_node(node: &str) -> Result<NormalizerConfig> {
    let (name, args) = split_name_args(node)?;
    Ok(match name.as_str() {
        "identity" | "none" => NormalizerConfig::Identity,
        "lowercase" => NormalizerConfig::Lowercase,
        "nfc" => NormalizerConfig::Nfc,
        "nfd" => NormalizerConfig::Nfd,
        "nfkc" => NormalizerConfig::Nfkc,
        "nfkd" => NormalizerConfig::Nfkd,
        "strip-accents" => NormalizerConfig::StripAccents,
        "strip" => NormalizerConfig::Strip,
        "bytelevel" | "byte-level" | "byte_level" => NormalizerConfig::ByteLevel,
        "replace" => {
            // 两种 pattern 形态：literal=<字面量>（HF `{"String": ...}`）
            // 与 pattern=<regex>（HF `{"Regex": ...}`）。
            let spec = if let Some(literal) = take_arg(&args, &["literal", "string"]) {
                if take_arg(&args, &["pattern", "regex"]).is_some() {
                    return Err(TokenizerError::config(
                        "`--normalizer replace:...` 的 literal= 与 pattern= 互斥，只能二选一"
                            .to_string(),
                    ));
                }
                tok_core::PatternSpec::literal(literal)
            } else {
                let pattern = take_arg(&args, &["pattern", "regex"]).ok_or_else(|| {
                    TokenizerError::config(
                        "`--normalizer replace:...` 需要 literal=<字面量> 或 pattern=<regex>\
                         （例：replace:literal=.;content=_ / replace:pattern=\\s+;content= ）"
                            .to_string(),
                    )
                })?;
                tok_core::PatternSpec::new(pattern)
            };
            let content = take_arg(&args, &["content", "with"]).unwrap_or_default();
            NormalizerConfig::Replace {
                pattern: spec,
                content,
            }
        }
        "prepend" => {
            let prepend = take_arg(&args, &["prepend", "prefix"]).ok_or_else(|| {
                TokenizerError::config(
                    "`--normalizer prepend:...` 需要 prepend=<字符串>".to_string(),
                )
            })?;
            NormalizerConfig::Prepend { prepend }
        }
        other => {
            return Err(TokenizerError::config(format!(
                "未知 --normalizer: '{}'（可选 lowercase / nfc / nfd / nfkc / nfkd / strip-accents / \
                 strip / bytelevel / replace:literal=<s>;content=<s> / \
                 replace:pattern=<regex>;content=<s> / prepend:prepend=<s>）",
                other
            )));
        }
    })
}

fn build_pre_tokenizer(spec: &str) -> Result<Option<PreTokenizerConfig>> {
    let spec = spec.trim();
    if spec.is_empty() || spec == "none" {
        return Ok(None);
    }
    Ok(Some(parse_pretok_node(spec)?))
}

/// 预分词节点：`name`、`name:k=v;k2=v2` 或 `name(k=v;k2=v2)`。
fn parse_pretok_node(node: &str) -> Result<PreTokenizerConfig> {
    let node = node.trim();
    // sequence(...)：按顶层逗号组合
    if let Some(inner) = node.strip_prefix("sequence(") {
        let inner = inner
            .strip_suffix(')')
            .ok_or_else(|| TokenizerError::config(format!("预分词表达式括号未闭合: '{}'", node)))?;
        let mut items = Vec::new();
        for part in split_top_level(inner)? {
            items.push(parse_pretok_node(&part)?);
        }
        if items.is_empty() {
            return Err(TokenizerError::config(
                "`sequence(...)` 内至少需要一个预分词器".to_string(),
            ));
        }
        return Ok(if items.len() == 1 {
            items.pop().expect("len=1")
        } else {
            PreTokenizerConfig::Sequence {
                pretokenizers: items,
            }
        });
    }
    let (name, args) = split_name_args(node)?;
    let no_args = |a: &BTreeMap<String, String>| -> Result<()> {
        if !a.is_empty() {
            return Err(TokenizerError::config(format!(
                "预分词器 '{}' 不接受参数: {:?}",
                name, a
            )));
        }
        Ok(())
    };
    Ok(match name.as_str() {
        "bytelevel" => PreTokenizerConfig::ByteLevel {
            add_prefix_space: bool_arg(&args, "add_prefix_space", false)?,
            trim_offsets: bool_arg(&args, "trim_offsets", true)?,
            use_regex: bool_arg(&args, "use_regex", true)?,
        },
        "whitespace" => {
            no_args(&args)?;
            PreTokenizerConfig::Whitespace
        }
        "whitespace-split" | "whitespace_split" => {
            no_args(&args)?;
            PreTokenizerConfig::WhitespaceSplit
        }
        "bert" => {
            no_args(&args)?;
            PreTokenizerConfig::BertPreTokenizer
        }
        "metaspace" => PreTokenizerConfig::Metaspace {
            replacement: take_arg(&args, &["replacement"])
                .unwrap_or_else(|| "\u{2581}".to_string()),
            add_prefix_space: bool_arg(&args, "add_prefix_space", true)?,
            prepend_scheme: Some(
                take_arg(&args, &["prepend_scheme"]).unwrap_or_else(|| "always".to_string()),
            ),
        },
        "digits" => PreTokenizerConfig::Digits {
            individual_digits: bool_arg(&args, "individual_digits", false)?,
        },
        "punctuation" => PreTokenizerConfig::Punctuation {
            behavior: BehaviorSpec::named(
                &take_arg(&args, &["behavior"]).unwrap_or_else(|| "isolated".to_string()),
            ),
        },
        "split" => {
            let spec = if let Some(literal) = take_arg(&args, &["literal", "string"]) {
                tok_core::PatternSpec::literal(literal)
            } else {
                let pattern = take_arg(&args, &["pattern", "regex"]).ok_or_else(|| {
                    TokenizerError::config(
                        "`--pre-tokenizer split:...` 需要 literal=<字面量> 或 pattern=<regex>\
                         （逗号请写成 \\x2C，避免与 sequence 分隔冲突）"
                            .to_string(),
                    )
                })?;
                tok_core::PatternSpec::new(pattern)
            };
            PreTokenizerConfig::Split {
                pattern: spec,
                behavior: BehaviorSpec::named(
                    &take_arg(&args, &["behavior"]).unwrap_or_else(|| "removed".to_string()),
                ),
                invert: bool_arg(&args, "invert", false)?,
            }
        }
        other => {
            return Err(TokenizerError::config(format!(
                "未知 --pre-tokenizer: '{}'（可选 bytelevel / whitespace / whitespace-split / bert / \
                 metaspace / digits / punctuation / split:pattern=<regex> / sequence(a,b) / none）",
                other
            )));
        }
    })
}

/// 拆出 `name` 与参数表：支持 `name:k=v;...` 与 `name(k=v;...)` 两种写法。
fn split_name_args(node: &str) -> Result<(String, BTreeMap<String, String>)> {
    let node = node.trim();
    if let Some(open) = node.find('(') {
        if !node.ends_with(')') {
            return Err(TokenizerError::config(format!(
                "表达式括号未闭合: '{}'",
                node
            )));
        }
        let name = node[..open].trim().to_string();
        let inner = &node[open + 1..node.len() - 1];
        return Ok((name, parse_kv_args(inner)?));
    }
    match node.split_once(':') {
        Some((name, rest)) => Ok((name.trim().to_string(), parse_kv_args(rest)?)),
        None => Ok((node.to_string(), BTreeMap::new())),
    }
}

/// 取参数（任一别名命中即取出并从表里移除，避免"未知参数"误报）。
fn take_arg(args: &BTreeMap<String, String>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| args.get(*k).cloned())
}

fn bool_arg(args: &BTreeMap<String, String>, key: &str, default: bool) -> Result<bool> {
    match args.get(key) {
        None => Ok(default),
        Some(v) => match v.as_str() {
            "true" | "1" | "yes" | "on" => Ok(true),
            "false" | "0" | "no" | "off" => Ok(false),
            other => Err(TokenizerError::config(format!(
                "参数 {} 需要布尔值，实际为 '{}'",
                key, other
            ))),
        },
    }
}

/// 推断用基名：`sequence(a,b)` 取首段，`split:...` 取 `split`。
fn pretok_base_name(spec: &str) -> String {
    let s = spec.trim();
    if let Some(inner) = s.strip_prefix("sequence(") {
        if let Some(first) = split_top_level(inner.trim_end_matches(')'))
            .ok()
            .and_then(|v| v.into_iter().next())
        {
            return pretok_base_name(&first);
        }
    }
    let cut = s.find([':', '(']).unwrap_or(s.len());
    s[..cut].trim().to_string()
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

/// 后处理器自动推断（`--post-processor auto`，默认值）。
///
/// 仅在用户**显式**给出了成对的 CLS/SEP 时才装配：
/// * 来源优先级：`--cls` / `--sep` > `--role cls=...` / `--role sep=...`；
/// * 只给其中一个 → 不推断（宁可不加，也不与另一半的自定义角色拼错）；
/// * 都不给 → 返回 `None`，与历史 `none` 行为完全一致，不改变既有产物。
///
/// 装配形态跟随模型家族：WordPiece / WordLevel 走 `BertProcessing`，
/// 其余（BPE / Unigram）按 RoBERTa 约定走 `RobertaProcessing`。
fn infer_post_processor(
    model: &str,
    roles: &TokenRoles,
    cls: Option<&str>,
    sep: Option<&str>,
) -> Option<PostProcessorConfig> {
    let cls = cls
        .map(str::to_string)
        .or_else(|| roles.cls_token.clone())?;
    let sep = sep
        .map(str::to_string)
        .or_else(|| roles.sep_token.clone())?;
    Some(match model {
        "wordpiece" | "wordlevel" => PostProcessorConfig::BertProcessing {
            sep: (sep, 0),
            cls: (cls, 0),
        },
        _ => PostProcessorConfig::RobertaProcessing {
            sep: (sep, 0),
            cls: (cls, 0),
            trim_offsets: true,
            add_prefix_space: true,
        },
    })
}

/// 后处理器 spec → 配置（`none` 返回 `None`）。
///
/// 支持：
/// * `none` / 空串：不启用后处理
/// * `auto`：按显式声明的角色推断（见 [`infer_post_processor`]）
/// * `bert` / `roberta`：走对应的内置处理器（`--cls` / `--sep` 覆盖字面值）
/// * `bert-template`：与 `bert` 语义等价的手写模板，用于 Bert ↔ Template 自洽校验
/// * `template:<single>[|<pair>]`：`$A` / `$B` 为序列占位，其余 token 按特殊 token 字面值；
///   `$B` 之后自动切到 `type_id = 1`
fn parse_post_processor_spec(
    spec: &str,
    model: &str,
    cls: Option<&str>,
    sep: Option<&str>,
) -> Result<Option<PostProcessorConfig>> {
    let spec = spec.trim();
    if spec.is_empty() || spec == "none" {
        return Ok(None);
    }
    // 角色 token 缺省值：WordPiece / WordLevel 系沿用 BERT 约定，其余按 RoBERTa 约定
    let (cls_default, sep_default) = match model {
        "wordpiece" | "wordlevel" => ("[CLS]", "[SEP]"),
        _ => ("<s>", "</s>"),
    };
    let cls = cls.unwrap_or(cls_default).to_string();
    let sep = sep.unwrap_or(sep_default).to_string();
    Ok(Some(match spec {
        "bert" => PostProcessorConfig::BertProcessing {
            sep: (sep, 0),
            cls: (cls, 0),
        },
        "roberta" => PostProcessorConfig::RobertaProcessing {
            sep: (sep, 0),
            cls: (cls, 0),
            trim_offsets: true,
            add_prefix_space: true,
        },
        "bert-template" => PostProcessorConfig::TemplateProcessing {
            single: vec![
                TemplatePiece::special(cls.clone(), 0),
                TemplatePiece::sequence("A", 0),
                TemplatePiece::special(sep.clone(), 0),
            ],
            pair: vec![
                TemplatePiece::special(cls, 0),
                TemplatePiece::sequence("A", 0),
                TemplatePiece::special(sep.clone(), 0),
                TemplatePiece::sequence("B", 1),
                TemplatePiece::special(sep, 1),
            ],
            special_tokens: Default::default(),
        },
        other => {
            let body = other.strip_prefix("template:").ok_or_else(|| {
                TokenizerError::config(format!(
                    "未知 --post-processor: '{}'（可选 auto / none / bert / roberta / bert-template / template:...）",
                    other
                ))
            })?;
            let (single, pair) = match body.split_once('|') {
                Some((s, p)) => (s, Some(p)),
                None => (body, None),
            };
            let single = parse_template(single)?;
            let pair = match pair {
                Some(p) => parse_template(p)?,
                None => Vec::new(),
            };
            PostProcessorConfig::TemplateProcessing {
                single,
                pair,
                special_tokens: Default::default(),
            }
        }
    }))
}

/// `"[CLS] $A [SEP]"` → 模板片段；`$B` 出现后后续片段使用 `type_id = 1`。
fn parse_template(spec: &str) -> Result<Vec<TemplatePiece>> {
    let mut out = Vec::new();
    let mut type_id = 0u32;
    for token in spec.split_whitespace() {
        match token {
            "$A" => out.push(TemplatePiece::sequence("A", type_id)),
            "$B" => {
                type_id = 1;
                out.push(TemplatePiece::sequence("B", 1));
            }
            literal => out.push(TemplatePiece::special(literal, type_id)),
        }
    }
    if !out
        .iter()
        .any(|p| matches!(p, TemplatePiece::Sequence(s) if s.sequence.id == "A"))
    {
        return Err(TokenizerError::config(format!(
            "后处理模板 '{}' 缺少序列占位 $A",
            spec
        )));
    }
    Ok(out)
}

/// 解码器 spec → 配置（`auto` 由调用方处理，`none` 返回 `None`）。
fn parse_decoder_spec(spec: &str, byte_fallback: bool) -> Result<Option<DecoderConfig>> {
    let spec = spec.trim();
    if spec.is_empty() || spec == "none" {
        return Ok(None);
    }
    let nodes = split_top_level(spec)?;
    let decoders: Vec<DecoderConfig> = nodes
        .iter()
        .map(|n| parse_decoder_node(n, byte_fallback))
        .collect::<Result<_>>()?;
    Ok(match decoders.len() {
        0 => None,
        1 => decoders.into_iter().next(),
        _ => Some(DecoderConfig::Sequence { decoders }),
    })
}

fn parse_decoder_node(node: &str, byte_fallback: bool) -> Result<DecoderConfig> {
    let node = node.trim();
    if let Some(inner) = node.strip_prefix("fuse(").and_then(|s| s.strip_suffix(')')) {
        // HF 的 `Fuse` 解码器是无参的"原样拼接"占位符，不承载子解码器；
        // 因此 `fuse(a,b)` 只是输入侧的分组语法糖，导出时必须落成 HF 的 `Sequence`。
        let decoders = split_top_level(inner)?
            .iter()
            .map(|n| parse_decoder_node(n, byte_fallback))
            .collect::<Result<Vec<_>>>()?;
        return Ok(DecoderConfig::Sequence { decoders });
    }
    let (name, args) = match node.split_once(':') {
        Some((n, rest)) => (n.trim(), parse_kv_args(rest)?),
        None => (node, BTreeMap::new()),
    };
    let flag = |key: &str, default: bool| -> bool {
        args.get(key)
            .map(|v| v == "true" || v == "1")
            .unwrap_or(default)
    };
    let text = |key: &str, default: &str| -> String {
        args.get(key)
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };
    Ok(match name {
        "bytelevel" => DecoderConfig::ByteLevel {
            add_prefix_space: flag("add_prefix_space", false),
            trim_offsets: flag("trim_offsets", true),
            use_regex: flag("use_regex", true),
        },
        "wordpiece" => DecoderConfig::WordPiece {
            prefix: text("prefix", "##"),
            cleanup: flag("cleanup", true),
        },
        "metaspace" => DecoderConfig::Metaspace {
            replacement: text("replacement", "\u{2581}"),
            prepend_scheme: text("prepend_scheme", "always"),
            split: flag("split", true),
        },
        "bpe" => DecoderConfig::BPEDecoder {
            suffix: text("suffix", "</w>"),
        },
        "ctc" => DecoderConfig::CTC {
            pad_token: text("pad_token", "<pad>"),
            word_delimiter_token: text("word_delimiter", "|"),
            cleanup: flag("cleanup", true),
        },
        "bytefallback" => {
            if !byte_fallback {
                // 词表没有 `<0xXX>` 常驻区时该解码器无副作用，仅提示（不改语义）
                eprintln!("提示: 未启用 --byte-fallback，ByteFallback 解码器不会命中任何 token");
            }
            DecoderConfig::ByteFallback
        }
        "strip" => DecoderConfig::Strip {
            content: text("content", " "),
            start: text("start", "0").parse().unwrap_or(0),
            stop: text("stop", "1").parse().unwrap_or(1),
        },
        other => {
            return Err(TokenizerError::config(format!(
                "未知解码器节点: '{}'（可选 bytelevel / wordpiece / metaspace / bpe / ctc / bytefallback / strip，或用 fuse(...) 组合）",
                other
            )));
        }
    })
}

/// `prefix=##;cleanup=false` → 参数表（`;` 分隔，避免与 Sequence 的 `,` 冲突）。
///
/// 解码器 / 预分词器 / 归一化器共用同一套参数语法。
fn parse_kv_args(raw: &str) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for pair in raw.split(';').filter(|s| !s.trim().is_empty()) {
        let (k, v) = pair.split_once('=').ok_or_else(|| {
            TokenizerError::config(format!("参数 '{}' 不是 k=v 形式", pair.trim()))
        })?;
        out.insert(k.trim().to_string(), parse_arg_value(v));
    }
    Ok(out)
}

/// 参数值解码：默认去空白；`"..."` / `'...'` 引号包裹时原样保留内部内容
/// （用于 `replace:content=" "` 这类需要"一个空格"的字面值）。
///
/// 只在 CLI 表达式层做引号剥离，落盘到 `tokenizer.json` 的仍是字面值，与 HF 一致。
fn parse_arg_value(raw: &str) -> String {
    let v = raw.trim();
    for q in ['"', '\''] {
        if v.len() >= 2 && v.starts_with(q) && v.ends_with(q) {
            return v[1..v.len() - 1].to_string();
        }
    }
    v.to_string()
}

/// 截断参数：`--truncation-max-length` 缺省时导出 `truncation: null`（不截断）。
fn parse_truncation(
    max_length: Option<usize>,
    strategy: &str,
    direction: &str,
    stride: usize,
) -> Result<Option<TruncationParams>> {
    let Some(max_length) = max_length else {
        return Ok(None);
    };
    if max_length == 0 {
        return Err(TokenizerError::config(
            "--truncation-max-length 必须大于 0".to_string(),
        ));
    }
    let strategy = match strategy.trim().replace('_', "-").as_str() {
        "longest-first" => TruncationStrategy::LongestFirst,
        "only-first" => TruncationStrategy::OnlyFirst,
        "only-second" => {
            return Err(TokenizerError::config(
                "--truncation-strategy only-second 当前仅支持 pair 语义，单序列训练期无意义"
                    .to_string(),
            ));
        }
        other => {
            return Err(TokenizerError::config(format!(
                "未知 --truncation-strategy: '{}'（可选 longest-first / only-first / only-second）",
                other
            )));
        }
    };
    Ok(Some(TruncationParams {
        max_length,
        strategy,
        stride,
        direction: parse_direction(direction, "--truncation-direction")?,
    }))
}

fn parse_direction(raw: &str, flag: &str) -> Result<TruncationDirection> {
    match raw.trim() {
        "left" => Ok(TruncationDirection::Left),
        "right" => Ok(TruncationDirection::Right),
        other => Err(TokenizerError::config(format!(
            "未知 {}: '{}'（可选 left / right）",
            flag, other
        ))),
    }
}

fn parse_padding_direction(raw: &str) -> Result<PaddingDirection> {
    match raw.trim() {
        "left" => Ok(PaddingDirection::Left),
        "right" => Ok(PaddingDirection::Right),
        other => Err(TokenizerError::config(format!(
            "未知 --padding-direction: '{}'（可选 left / right）",
            other
        ))),
    }
}

/// 填充参数：`none` | `batch-longest` | `fixed:<N>`。
///
/// `pad_id` 的解析顺序：显式 `--pad-id` → 词表里 `--pad-token` 的真实 id → 0。
/// 训练前词表为空时（`vocab = None`）不解析，交给训练后的第二次调用。
fn parse_padding(
    spec: &str,
    direction: &str,
    pad_token: Option<&str>,
    pad_id: Option<u32>,
    pad_to_multiple_of: Option<usize>,
    vocab: Option<&BTreeMap<String, u32>>,
    specials: &[String],
) -> Result<Option<PaddingParams>> {
    let spec = spec.trim();
    if spec.is_empty() || spec == "none" {
        return Ok(None);
    }
    let strategy = if spec == "batch-longest" || spec == "batch_longest" {
        PaddingStrategy::BatchLongest
    } else if let Some(n) = spec.strip_prefix("fixed:") {
        let n: usize = n.trim().parse().map_err(|_| {
            TokenizerError::config(format!("--padding fixed:<N> 的 N 不是整数: '{}'", spec))
        })?;
        if n == 0 {
            return Err(TokenizerError::config(
                "--padding fixed:<N> 的 N 必须大于 0".to_string(),
            ));
        }
        PaddingStrategy::Fixed(n)
    } else {
        return Err(TokenizerError::config(format!(
            "未知 --padding: '{}'（可选 none / batch-longest / fixed:<N>）",
            spec
        )));
    };

    // pad token 字面值：显式指定优先，否则按 HF 常见写法在候选里挑
    let token = match pad_token {
        Some(t) => t.to_string(),
        None => ["[PAD]", "<pad>", "[pad]"]
            .into_iter()
            .find(|c| {
                vocab.map(|v| v.contains_key(*c)).unwrap_or(false)
                    || specials.iter().any(|s| s == c)
            })
            .map(|c| c.to_string())
            .ok_or_else(|| {
                TokenizerError::config(
                    "启用 --padding 需要能确定 pad token：请传 --pad-token <字面值>（词表里没有 [PAD]/<pad>）"
                        .to_string(),
                )
            })?,
    };
    let id = match pad_id {
        Some(id) => id,
        None => vocab.and_then(|v| v.get(&token).copied()).unwrap_or(0),
    };
    Ok(Some(PaddingParams {
        strategy,
        direction: parse_padding_direction(direction)?,
        pad_to_multiple_of,
        pad_id: id,
        pad_type_id: 0,
        pad_token: token,
    }))
}

/// 按顶层逗号切分（括号内的逗号不切）。
fn split_top_level(spec: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut cur = String::new();
    for ch in spec.chars() {
        match ch {
            '(' => {
                depth += 1;
                cur.push(ch);
            }
            ')' => {
                depth = depth.checked_sub(1).ok_or_else(|| {
                    TokenizerError::config(format!("表达式括号不匹配: '{}'", spec))
                })?;
                cur.push(ch);
            }
            ',' if depth == 0 => {
                let item = cur.trim();
                if !item.is_empty() {
                    out.push(item.to_string());
                }
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    if depth != 0 {
        return Err(TokenizerError::config(format!(
            "表达式括号未闭合: '{}'",
            spec
        )));
    }
    let item = cur.trim();
    if !item.is_empty() {
        out.push(item.to_string());
    }
    Ok(out)
}

/// 把 `--truncation/--padding` 透传进 pipeline（E5：截断/填充内核早已实现，此前无 CLI 入口）。
fn apply_io_params(
    tk: &mut Tokenizer,
    trunc: Option<TruncationParams>,
    pad: Option<PaddingParams>,
) {
    tk.set_truncation(trunc);
    tk.set_padding(pad);
}

#[cfg(test)]
mod arg_syntax_tests {
    use super::*;

    #[test]
    fn kv_args_support_quoted_whitespace_value() {
        // `--normalizer 'replace:pattern=\s+;content=" "'`：引号内部必须是**一个空格**字面值，
        // 否则会在去空白时被吃掉，落盘的 content 变成空串、与 HF 行为不符。
        let args = parse_kv_args(r#"pattern=\s+;content=" ""#).unwrap();
        assert_eq!(args.get("pattern").map(String::as_str), Some(r#"\s+"#));
        assert_eq!(args.get("content").map(String::as_str), Some(" "));

        // 单引号等价；裸值仍按惯例去空白。
        let args = parse_kv_args("content=' ';flag=  true  ").unwrap();
        assert_eq!(args.get("content").map(String::as_str), Some(" "));
        assert_eq!(args.get("flag").map(String::as_str), Some("true"));
    }

    #[test]
    fn kv_args_reject_non_pair() {
        assert!(parse_kv_args("pattern").is_err());
    }
}
