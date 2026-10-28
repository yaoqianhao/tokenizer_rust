//! 门面 `Tokenizer`：装配 L1..L6 并实现端到端 `encode` / `decode` / `train` / 落盘。
//!
//! 依赖方向：`tok-pipeline` → 六个实现 crate；实现 crate 之间互不依赖。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use tok_core::{
    layer, AddedToken, AddedTokenWithId, Decoder, Encoding, Model, NormalizedString, Normalizer,
    PaddingParams, PaddingStrategy, PostProcessor, PreTokenizer, Result, TokenizerError,
    TrainerParams, TruncationParams,
};
use tok_decoders::{DecoderConfig, DecoderWrapper};
use tok_models::{ModelConfig, ModelWrapper};
use tok_normalizers::{NormalizerConfig, NormalizerWrapper};
use tok_post::{PostProcessorConfig, PostProcessorWrapper};
use tok_pretok::{PreTokenizerConfig, PreTokenizerWrapper};
use tok_trainers::{train_model, CorpusReader, TrainOptions, TrainerReport};

use crate::hf::{self, HfTokenizerJson, TokenRoles, HF_SCHEMA_VERSION};

fn default_max_len() -> usize {
    512
}

/// 流水线装配配置：与 `tokenizer.json` 的可见字段一一对应（另含模型最大长度与角色 token）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfig {
    #[serde(default)]
    pub normalizer: Option<NormalizerConfig>,
    #[serde(default)]
    pub pre_tokenizer: Option<PreTokenizerConfig>,
    pub model: ModelConfig,
    #[serde(default)]
    pub post_processor: Option<PostProcessorConfig>,
    #[serde(default)]
    pub decoder: Option<DecoderConfig>,
    #[serde(default)]
    pub added_tokens: Vec<AddedToken>,
    #[serde(default)]
    pub truncation: Option<TruncationParams>,
    #[serde(default)]
    pub padding: Option<PaddingParams>,
    #[serde(default = "default_max_len")]
    pub model_max_length: usize,
    #[serde(default)]
    pub roles: TokenRoles,
}

/// 落盘结果：文件路径 + 行数。
#[derive(Debug, Clone, Default)]
pub struct SavedFiles {
    pub dir: PathBuf,
    pub files: Vec<(PathBuf, usize)>,
}

impl SavedFiles {
    pub fn total_lines(&self) -> usize {
        self.files.iter().map(|(_, n)| *n).sum()
    }

    pub fn names(&self) -> Vec<String> {
        self.files
            .iter()
            .map(|(p, _)| {
                p.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default()
            })
            .collect()
    }

    /// 便于日志/测试断言的紧凑描述：`tokenizer.json(421 行), ...`
    pub fn describe(&self) -> String {
        self.files
            .iter()
            .map(|(p, n)| {
                format!(
                    "{}({} 行)",
                    p.file_name()
                        .map(|x| x.to_string_lossy().to_string())
                        .unwrap_or_default(),
                    n
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// HF fast tokenizer 门面。
pub struct Tokenizer {
    /// 装配配置（序列化时的真值来源；训练后刷新 model 节）
    config: PipelineConfig,
    normalizer: Option<NormalizerWrapper>,
    pre_tokenizer: Option<PreTokenizerWrapper>,
    model: ModelWrapper,
    post_processor: PostProcessorWrapper,
    decoder: Option<DecoderWrapper>,
    added_tokens: Vec<AddedToken>,
    roles: TokenRoles,
    truncation: Option<TruncationParams>,
    padding: Option<PaddingParams>,
    model_max_length: usize,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("normalizer", &self.normalizer.as_ref().map(|n| n.name()))
            .field(
                "pre_tokenizer",
                &self.pre_tokenizer.as_ref().map(|p| p.name()),
            )
            .field("model", &self.model.type_name())
            .field("post_processor", &self.post_processor.name())
            .field("decoder", &self.decoder.as_ref().map(|d| d.name()))
            .field("vocab_size", &self.model.get_vocab().len())
            .finish()
    }
}

impl Tokenizer {
    // ---------------------------------------------------------------- 装配

    /// 依据配置装配流水线，并把 `added_tokens` 登记进词表（HF 语义：added token 必须可查 id）。
    pub fn from_config(cfg: &PipelineConfig) -> Result<Self> {
        // 训练前的空词表是合法起点（HF 同构：先建空 tokenizer，再 train 出词表）
        if !model_config_is_empty(&cfg.model) {
            cfg.model.validate()?;
        }

        let normalizer = match &cfg.normalizer {
            None => None,
            Some(NormalizerConfig::Identity) => None,
            Some(n) => Some(NormalizerWrapper::build(n)?),
        };
        let pre_tokenizer = match &cfg.pre_tokenizer {
            None => None,
            Some(p) => Some(PreTokenizerWrapper::build(p)?),
        };
        let model = ModelWrapper::build(&cfg.model)?;
        let post_processor = match &cfg.post_processor {
            None => PostProcessorWrapper::identity(),
            Some(p) => PostProcessorWrapper::build(p)?,
        };
        let decoder = match &cfg.decoder {
            None => None,
            Some(d) => Some(DecoderWrapper::build(d)?),
        };

        let mut tk = Self {
            config: cfg.clone(),
            normalizer,
            pre_tokenizer,
            model,
            post_processor,
            decoder,
            added_tokens: Vec::new(),
            roles: cfg.roles.clone(),
            truncation: cfg.truncation.clone(),
            padding: cfg.padding.clone(),
            model_max_length: cfg.model_max_length,
        };

        // 1) 登记配置里声明的 added token
        let mut to_register: Vec<AddedToken> = cfg.added_tokens.clone();
        // 2) 后处理器模板引用的特殊 token 必须可解析
        for s in tk.post_processor.required_specials() {
            if !to_register.iter().any(|t| t.content == s) {
                to_register.push(AddedToken::new(s, true));
            }
        }
        // 3) 角色 token 兜底登记
        let mut roles = tk.roles.clone();
        if roles.unk_token.is_none() {
            roles.unk_token = model_unk_token(&cfg.model);
        }
        roles.ensure_in_added(&mut to_register);
        tk.roles = roles;

        tk.register_added(&to_register)?;
        if tk.roles.is_empty() {
            tk.roles = TokenRoles::infer(&tk.added_tokens);
        }
        tk.post_processor.bind_vocab(tk.model.get_vocab())?;
        Ok(tk)
    }

    /// 由 HF `tokenizer.json` 反装（用于跨实现校验回读）。
    pub fn from_tokenizer_json(value: &Value) -> Result<Self> {
        hf::validate_tokenizer_json(value)?;
        let model_json = value
            .get("model")
            .ok_or_else(|| TokenizerError::config("缺 model 节"))?;
        let model_wrapper = ModelWrapper::from_hf_json(model_json)?;
        let model_cfg = model_wrapper.to_model_config()?;

        let normalizer = match value.get("normalizer") {
            None | Some(Value::Null) => None,
            Some(v) => Some(serde_json::from_value::<NormalizerConfig>(v.clone())?),
        };
        let pre_tokenizer = match value.get("pre_tokenizer") {
            None | Some(Value::Null) => None,
            Some(v) => Some(serde_json::from_value::<PreTokenizerConfig>(v.clone())?),
        };
        let post_processor = match value.get("post_processor") {
            None | Some(Value::Null) => None,
            Some(v) => Some(serde_json::from_value::<PostProcessorConfig>(v.clone())?),
        };
        let decoder = match value.get("decoder") {
            None | Some(Value::Null) => None,
            Some(v) => Some(serde_json::from_value::<DecoderConfig>(v.clone())?),
        };
        let added_tokens: Vec<AddedToken> = match value.get("added_tokens") {
            Some(Value::Array(items)) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    // HF `tokenizer.json` 的 added_tokens 元素带 `id` 字段；
                    // 优先按带 id 形态解析（id 由词表登记顺序决定，不直接采用），
                    // 失败时回退到无 id 的简化形态。
                    match serde_json::from_value::<AddedTokenWithId>(it.clone()) {
                        Ok(t) => out.push(t.to_token()),
                        Err(_) => out.push(serde_json::from_value::<AddedToken>(it.clone())?),
                    }
                }
                out
            }
            _ => Vec::new(),
        };
        let cfg = PipelineConfig {
            normalizer,
            pre_tokenizer,
            model: model_cfg,
            post_processor,
            decoder,
            added_tokens,
            truncation: serde_json::from_value(
                value.get("truncation").cloned().unwrap_or(Value::Null),
            )
            .ok(),
            padding: serde_json::from_value(value.get("padding").cloned().unwrap_or(Value::Null))
                .ok(),
            model_max_length: default_max_len(),
            roles: TokenRoles::default(),
        };
        Self::from_config(&cfg)
    }

    /// 导出为可再次装配的 `PipelineConfig`（词表取训练后的实时状态）。
    pub fn to_config(&self) -> Result<PipelineConfig> {
        let mut cfg = self.config.clone();
        cfg.model = self.model.to_model_config()?;
        cfg.added_tokens = self.added_tokens.clone();
        cfg.roles = self.roles.clone();
        cfg.truncation = self.truncation.clone();
        cfg.padding = self.padding.clone();
        Ok(cfg)
    }

    /// 原始装配配置（不含训练后的词表变化）。
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// 设置截断参数（`None` 关闭截断）。
    pub fn set_truncation(&mut self, params: Option<TruncationParams>) {
        self.truncation = params.clone();
        self.config.truncation = params;
    }

    /// 设置填充参数（`None` 关闭填充）。
    pub fn set_padding(&mut self, params: Option<PaddingParams>) {
        self.padding = params.clone();
        self.config.padding = params;
    }

    pub fn model(&self) -> &ModelWrapper {
        &self.model
    }

    pub fn model_config(&self) -> ModelConfig {
        self.model
            .to_model_config()
            .unwrap_or_else(|_| unreachable_config())
    }

    pub fn roles(&self) -> &TokenRoles {
        &self.roles
    }

    pub fn added_tokens(&self) -> &[AddedToken] {
        &self.added_tokens
    }

    pub fn model_max_length(&self) -> usize {
        self.model_max_length
    }

    // ---------------------------------------------------------------- 登记

    /// 登记后加 token：分配 id（追加在词表末尾，不重编号既有 ID）并写入后加视图。
    pub fn register_added(&mut self, tokens: &[AddedToken]) -> Result<usize> {
        let mut added = self.added_tokens.clone();
        let mut count = 0usize;
        for t in tokens {
            if let Some(existing) = self.model.get_vocab().token_to_id(&t.content) {
                // token 已在 base 词表（例如训练器把 special token 写进了 base vocab）：
                // 必须补登记到后加视图，否则 `special_ids` / `skip_special_tokens` /
                // `special_tokens_mask` 都会把它当普通 token（HF 中它同时在 vocab 与 added_tokens）。
                self.model
                    .get_vocab_mut()
                    .register_added_view(t.clone(), existing)?;
                if !added.iter().any(|x| x.content == t.content) {
                    added.push(t.clone());
                }
                continue;
            }
            let mut id = self.model.get_vocab().max_id() + 1;
            while self.model.get_vocab().id_to_token(id).is_some() {
                id += 1;
            }
            self.model.get_vocab_mut().add_special(t.clone(), id)?;
            added.push(t.clone());
            count += 1;
        }
        self.added_tokens = added;
        Ok(count)
    }

    /// 特殊 token 的 id 列表（`skip_special_tokens` 语义）。
    pub fn special_ids(&self) -> Vec<u32> {
        self.model
            .get_vocab()
            .added()
            .tokens()
            .iter()
            .filter(|(t, _)| t.special)
            .map(|(_, id)| *id)
            .collect()
    }

    // ---------------------------------------------------------------- 前向

    /// S0..S9：规范化 → 后加 token 前置切分 → 预分词 → 子词切分 → 截断 → 后处理。
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        let segs = self.encode_segments(&[text.to_string()])?;
        let enc = self
            .post_processor
            .process_encodings(segs, add_special_tokens)?;
        self.truncate_final(enc, add_special_tokens)
    }

    /// 双句编码（pair 语义）。
    pub fn encode_pair(&self, a: &str, b: &str, add_special_tokens: bool) -> Result<Encoding> {
        let segs = self.encode_segments(&[a.to_string(), b.to_string()])?;
        let enc = self
            .post_processor
            .process_encodings(segs, add_special_tokens)?;
        self.truncate_final(enc, add_special_tokens)
    }

    /// 批量编码（串行；`rayon` 并行留待 CLI 侧按批切分）。
    pub fn encode_batch(
        &self,
        texts: &[String],
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.encode(t, add_special_tokens)?);
        }
        Ok(out)
    }

    /// 只要子词字面值（调试/校验用）。
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.encode(text, false)?.tokens)
    }

    /// 截断 + 填充（S8 / S9）。
    ///
    /// HF 语义：`max_length` 是**最终序列长度**，模板新增的特殊 token（[CLS]/[SEP] 等）
    /// 由 `Encoding::truncate` 依据 `special_tokens_mask` 原位保留，不占用内容预算，
    /// 因此这里**不再**从 `max_length` 里预扣特殊 token 数（旧实现会双重扣减，导致
    /// 4 = [CLS] hello [SEP] + pad 这类用例被裁短）。
    fn truncate_final(&self, mut enc: Encoding, add_special_tokens: bool) -> Result<Encoding> {
        if let Some(trunc) = &self.truncation {
            let mut parts = enc.truncate(trunc);
            if parts.len() > 1 {
                let overflow = parts.split_off(1);
                enc = parts.pop().unwrap_or_default();
                enc.overflowing = overflow;
            } else {
                enc = parts.pop().unwrap_or_default();
            }
        }
        let _ = add_special_tokens;
        self.apply_padding(enc)
    }

    /// 填充：策略（`BatchLongest` / `Fixed`）+ `pad_to_multiple_of`，并受 `model_max_length` 兜底约束。
    fn apply_padding(&self, mut enc: Encoding) -> Result<Encoding> {
        let Some(p) = &self.padding else {
            return Ok(enc);
        };
        let mut target = match p.strategy {
            PaddingStrategy::BatchLongest => enc.len(),
            PaddingStrategy::Fixed(n) => n,
        };
        if let Some(m) = p.pad_to_multiple_of {
            if m > 0 && target % m != 0 {
                target = ((target / m) + 1) * m;
            }
        }
        if self.model_max_length > 0 && target > self.model_max_length {
            target = self.model_max_length;
        }
        enc.pad(p, target);
        Ok(enc)
    }

    /// 把 1~2 段文本编码为**未后处理**的 `Encoding`。
    fn encode_segments(&self, texts: &[String]) -> Result<Vec<Encoding>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.encode_one(t)?);
        }
        Ok(out)
    }

    fn encode_one(&self, text: &str) -> Result<Encoding> {
        // ---- S0 规范化（保持 original↔normalized 对齐）
        let mut ns = NormalizedString::new(text);
        if let Some(n) = &self.normalizer {
            ns = n.normalize(ns);
        }
        // ---- S1 后加 token 前置切分（最长左匹配）
        let units = self.model.get_vocab().added().split(ns.normalized());
        let mut enc = Encoding::default();
        let mut degraded = ns.is_degraded();

        for unit in units {
            match unit.id {
                Some(id) => {
                    let off = ns
                        .convert_offsets((unit.start, unit.end))
                        .map(|o| (o.start, o.end));
                    if off.is_none() {
                        degraded = true;
                    }
                    let (s, e) = off.unwrap_or((0, 0));
                    let special = self
                        .model
                        .get_vocab()
                        .added()
                        .tokens()
                        .iter()
                        .any(|(t, i)| *i == id && t.special);
                    enc.ids.push(id);
                    enc.type_ids.push(0);
                    enc.tokens.push(unit.text.clone());
                    enc.offsets.push((s, e));
                    enc.attention_mask.push(1);
                    enc.special_tokens_mask.push(if special { 1 } else { 0 });
                }
                None => {
                    // ---- S2 预分词（段内切分，偏移按整串对齐表换算，保证精确）
                    let seg = &ns.normalized()[unit.start..unit.end];
                    if seg.is_empty() {
                        continue;
                    }
                    let mut local = NormalizedString::new(seg);
                    if let Some(p) = &self.pre_tokenizer {
                        p.pre_tokenize(&mut local)?;
                    }
                    for pt in local.pretokens() {
                        let abs = (unit.start + pt.offsets.0, unit.start + pt.offsets.1);
                        let outer = match ns.convert_offsets(abs) {
                            Some(o) => (o.start, o.end),
                            None => {
                                degraded = true;
                                (0, 0)
                            }
                        };
                        // ---- S3 子词切分
                        let subwords = self.model.tokenize(&pt.value)?;
                        for sw in subwords {
                            let is_special = self
                                .model
                                .get_vocab()
                                .added()
                                .tokens()
                                .iter()
                                .any(|(t, i)| *i == sw.id && t.special);
                            enc.ids.push(sw.id);
                            enc.type_ids.push(0);
                            enc.tokens.push(sw.value.clone());
                            // 子词偏移是**映射串**上的区间：须经 alignment 换算回原文，
                            // 不能线性叠加。byte-level 下映射字符数 ≠ 原始字符数
                            // （汉字 1 字符 → 3 个映射字符），线性叠加会让偏移逐 token 漂移。
                            // 覆盖语义与 HF 一致：多个碎片落在同一原始字符内时返回同一区间。
                            // 坐标系统一为**字节**：pt.n_offsets 是预分词串上的字节区间
                            // （current_segments/splits 均为字节，见 types.rs），各 Model 的
                            // token.offsets 也以 pt.value 的字节计（bpe 用 char_indices 的字节跨度、
                            // wordpiece/unigram 亦然），因此直接相加，不做字节→字符换算。
                            let rel =
                                (pt.n_offsets.0 + sw.offsets.0, pt.n_offsets.0 + sw.offsets.1);
                            let (s, e) = match local.convert_offsets(rel) {
                                Some(o) => {
                                    let abs = (unit.start + o.start, unit.start + o.end);
                                    match ns.convert_offsets(abs) {
                                        Some(o2) => (o2.start, o2.end),
                                        None => {
                                            degraded = true;
                                            (0, 0)
                                        }
                                    }
                                }
                                // 对齐表未命中：退化为整个预分词单元区间
                                None => (outer.0, outer.1),
                            };
                            // 对外语义与 HF 对齐：offsets 输出**原文字符**区间（内部全程为字节）。
                            // 注意：特殊 token 也保留真实区间（HF 的 offsets 不因 special 置零，
                            // 只有 special_tokens_mask 标 1），因此这里不再对 is_special 特判。
                            let (s, e) = (byte_to_char_index(text, s), byte_to_char_index(text, e));
                            enc.offsets.push((s, e));
                            enc.attention_mask.push(1);
                            enc.special_tokens_mask.push(if is_special { 1 } else { 0 });
                        }
                    }
                }
            }
        }
        enc.offsets_degraded = degraded;
        enc.sequence_ranges.insert(0, (0, enc.ids.len()));
        enc.validate()?;
        Ok(enc)
    }

    // ---------------------------------------------------------------- 反向

    /// 解码：ids → 文本（`skip_special_tokens` 时先剔除特殊 id）。
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let vocab = self.model.get_vocab();
        match &self.decoder {
            Some(d) => {
                if skip_special_tokens {
                    d.decode_ids_skip_special(ids, vocab, &self.special_ids())
                } else {
                    d.decode_ids(ids, vocab)
                }
            }
            None => {
                let specials = self.special_ids();
                let mut out = String::new();
                for id in ids {
                    if skip_special_tokens && specials.contains(id) {
                        continue;
                    }
                    let t = vocab.id_to_token(*id).ok_or_else(|| {
                        TokenizerError::layer(
                            layer::DECODER,
                            format!("解码失败：词表中不存在 id {}", id),
                        )
                    })?;
                    out.push_str(&t);
                }
                Ok(out)
            }
        }
    }

    // ---------------------------------------------------------------- 训练

    /// 训练：`模型按 self 的 L1/L2 预处理语料 → L3 训练器 → 回填词表`，随后重建后处理/解码绑定。
    pub fn train(&mut self, reader: &CorpusReader, params: TrainerParams) -> Result<TrainerReport> {
        let data = reader.read()?;
        let mut opts = TrainOptions::new(params.clone());
        if let Some(n) = &self.normalizer {
            opts = opts.with_normalizer(n);
        }
        if let Some(p) = &self.pre_tokenizer {
            opts = opts.with_pre_tokenizer(p);
        }
        let report = train_model(&mut self.model, &data.lines, &opts)?;

        // 训练产物落盘后：特殊 token 常驻校验 + 重新绑定模板/角色
        self.model.reserve_tokens(
            &params
                .special_tokens
                .iter()
                .map(|s| AddedToken::new(s.clone(), true))
                .collect::<Vec<_>>(),
        )?;
        let mut to_register: Vec<AddedToken> = self.added_tokens.clone();
        // 训练参数里声明的 special token 必须进入登记表（tokenizer.json 的 added_tokens
        // 与 added_tokens_decoder 都以此为准）。
        for s in &params.special_tokens {
            if !to_register.iter().any(|t| t.content == *s) {
                to_register.push(AddedToken::new(s.clone(), true));
            }
        }
        for s in self.post_processor.required_specials() {
            if !to_register.iter().any(|t| t.content == s) {
                to_register.push(AddedToken::new(s, true));
            }
        }
        self.register_added(&to_register)?;

        // 训练后兜底补全角色：trainer 保留的 special token 与登记表都可能携带角色 token，
        // 训练前 roles 为空（如空词表起步）时必须在此重新推断，否则 special_tokens_map 会缺项。
        let mut observed: Vec<AddedToken> = self.added_tokens.clone();
        observed.extend(
            self.model
                .get_vocab()
                .added()
                .tokens()
                .iter()
                .map(|(t, _)| t.clone()),
        );
        let inferred = TokenRoles::infer(&observed);
        self.roles.fill_missing(&inferred);

        self.post_processor = rebuild_post(&self.post_processor)?;
        self.post_processor.bind_vocab(self.model.get_vocab())?;
        self.refresh_decoder_binding()?;
        Ok(report)
    }

    /// 解码器与词表的绑定检查（如 WordPiece 的 `##` 前缀需与模型配置一致）。
    fn refresh_decoder_binding(&self) -> Result<()> {
        if let Some(DecoderConfig::WordPiece { prefix, .. }) =
            self.decoder.as_ref().and_then(|d| d.config())
        {
            if let ModelConfig::WordPiece {
                continuing_subword_prefix,
                ..
            } = self.model_config()
            {
                if prefix != &continuing_subword_prefix {
                    return Err(TokenizerError::config(format!(
                        "解码器 WordPiece 前缀 '{}' 与模型配置 '{}' 不一致",
                        prefix, continuing_subword_prefix
                    )));
                }
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------- 导出

    /// `tokenizer.json`（HF fast 格式）。
    pub fn to_tokenizer_json(&self) -> Result<Value> {
        let normalizer = match &self.normalizer {
            None => Value::Null,
            Some(n) => n.to_config_json()?,
        };
        let pre_tokenizer = match &self.pre_tokenizer {
            None => Value::Null,
            Some(p) => p.to_config_json()?,
        };
        let model = self.model.to_hf_json()?;
        let post_processor = self.post_processor.to_config_json()?;
        let decoder = match &self.decoder {
            None => Value::Null,
            Some(d) => d.to_config_json()?,
        };
        let added = self.added_with_ids();
        let added_json: Vec<Value> = added
            .iter()
            .map(|(t, id)| {
                serde_json::to_value(AddedTokenWithId::from_token(t, *id)).unwrap_or(Value::Null)
            })
            .collect();

        let doc = HfTokenizerJson {
            version: HF_SCHEMA_VERSION.to_string(),
            truncation: match &self.truncation {
                None => Value::Null,
                Some(t) => serde_json::to_value(t)?,
            },
            padding: match &self.padding {
                None => Value::Null,
                Some(p) => serde_json::to_value(p)?,
            },
            added_tokens: added_json,
            normalizer,
            pre_tokenizer,
            post_processor,
            decoder,
            model,
        };
        let value = serde_json::to_value(&doc)?;
        hf::validate_tokenizer_json(&value)?;
        Ok(value)
    }

    /// `tokenizer_config.json`。
    pub fn to_tokenizer_config_json(&self) -> Result<Value> {
        Ok(hf::tokenizer_config(
            &self.roles,
            &self.added_with_ids(),
            self.model_max_length,
            self.model.type_name(),
        ))
    }

    /// `special_tokens_map.json`。
    pub fn to_special_tokens_map_json(&self) -> Result<Value> {
        Ok(hf::special_tokens_map(&self.roles))
    }

    /// added token 及其 id（按 id 升序）。
    pub fn added_with_ids(&self) -> Vec<(AddedToken, u32)> {
        let vocab = self.model.get_vocab();
        let mut out: Vec<(AddedToken, u32)> = self
            .added_tokens
            .iter()
            .filter_map(|t| vocab.token_to_id(&t.content).map(|id| (t.clone(), id)))
            .collect();
        out.sort_by_key(|(_, id)| *id);
        out
    }

    /// 落盘 `save_pretrained`：三份 HF 文件 + 模型侧文件（如 BPE 的 vocab.json / merges.txt）。
    pub fn save_pretrained(&self, dir: &Path) -> Result<SavedFiles> {
        std::fs::create_dir_all(dir)?;
        let mut saved = SavedFiles {
            dir: dir.to_path_buf(),
            files: Vec::new(),
        };

        let tj = self.to_tokenizer_json()?;
        saved
            .files
            .push(hf::write_json(&dir.join(hf::TOKENIZER_JSON), &tj)?);

        let tc = self.to_tokenizer_config_json()?;
        saved
            .files
            .push(hf::write_json(&dir.join(hf::TOKENIZER_CONFIG_JSON), &tc)?);

        let stm = self.to_special_tokens_map_json()?;
        saved.files.push(hf::write_json(
            &dir.join(hf::SPECIAL_TOKENS_MAP_JSON),
            &stm,
        )?);

        for name in self.model.save(dir)? {
            let p = dir.join(&name);
            let lines = std::fs::read_to_string(&p)
                .map(|s| s.lines().count())
                .unwrap_or(0);
            saved.files.push((p, lines));
        }
        Ok(saved)
    }

    /// 落盘 `save_pretrained`，并按需补写 `config.json`。
    ///
    /// `hf_identity` 为 `(model_type, tokenizer_class)`（如 `("gpt2", "GPT2Tokenizer")`）；
    /// 传 `None` 则只写 tokenizer 三件套，由下游自行提供 `config.json`。
    /// 补写 `config.json` 的目的是让 `AutoTokenizer.from_pretrained(dir)` 与
    /// 需要模型类型串的评测框架无需猜目录名即可走 `fast` 路径。
    pub fn save_pretrained_with_config(
        &self,
        dir: &Path,
        hf_identity: Option<(&str, &str)>,
    ) -> Result<SavedFiles> {
        let mut saved = self.save_pretrained(dir)?;
        if let Some((model_type, tokenizer_class)) = hf_identity {
            let value = hf::model_config(model_type, tokenizer_class);
            saved
                .files
                .push(hf::write_json(&dir.join(hf::CONFIG_JSON), &value)?);
        }
        Ok(saved)
    }

    /// 由模型实现类型推断 HF 身份（供 `config.json` 使用）。
    pub fn hf_identity(&self) -> Option<(&'static str, &'static str)> {
        hf::infer_hf_model_type(self.model.type_name())
    }

    /// 词表快照（`id → token`），供跨实现比对。
    pub fn vocab_map(&self) -> BTreeMap<String, u32> {
        self.model.get_vocab().to_id_map()
    }
}

/// 重建后处理器（保留模板表，避免训练后丢失特殊 token 绑定）。
fn rebuild_post(old: &PostProcessorWrapper) -> Result<PostProcessorWrapper> {
    match old.config() {
        None => Ok(PostProcessorWrapper::identity()),
        Some(cfg) => PostProcessorWrapper::build(cfg),
    }
}

fn model_config_is_empty(cfg: &ModelConfig) -> bool {
    match cfg {
        ModelConfig::Bpe { vocab, .. }
        | ModelConfig::WordPiece { vocab, .. }
        | ModelConfig::WordLevel { vocab, .. } => vocab.is_empty(),
        ModelConfig::Unigram { vocab, .. } => vocab.is_empty(),
    }
}

fn model_unk_token(cfg: &ModelConfig) -> Option<String> {
    match cfg {
        ModelConfig::Bpe { unk_token, .. } => unk_token.clone(),
        ModelConfig::WordPiece { unk_token, .. } => Some(unk_token.clone()),
        ModelConfig::WordLevel { unk_token, .. } => Some(unk_token.clone()),
        ModelConfig::Unigram { .. } => None,
    }
}

fn unreachable_config() -> ModelConfig {
    ModelConfig::WordLevel {
        unk_token: "<unk>".to_string(),
        vocab: Default::default(),
    }
}

/// 便捷构造：字节级 BPE + ByteLevel 预分词/解码 + ByteLevel 后处理（GPT-2 同构）。
pub fn byte_level_bpe(
    vocab: BTreeMap<String, u32>,
    merges: Vec<tok_models::MergeSpec>,
    roles: TokenRoles,
) -> PipelineConfig {
    PipelineConfig {
        normalizer: None,
        pre_tokenizer: Some(PreTokenizerConfig::ByteLevel {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        }),
        model: ModelConfig::Bpe {
            dropout: None,
            unk_token: roles.unk_token.clone(),
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            fuse_unk: false,
            byte_fallback: true,
            ignore_merges: false,
            vocab,
            merges,
        },
        post_processor: None,
        decoder: Some(DecoderConfig::ByteLevel {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        }),
        added_tokens: Vec::new(),
        truncation: None,
        padding: None,
        model_max_length: 512,
        roles,
    }
}

/// 便捷构造：BERT 风格（BertPreTokenizer + WordPiece + TemplateProcessing + WordPiece 解码）。
pub fn bert_like(vocab: BTreeMap<String, u32>, roles: TokenRoles) -> PipelineConfig {
    let unk = roles
        .unk_token
        .clone()
        .unwrap_or_else(|| "[UNK]".to_string());
    let cls = roles
        .cls_token
        .clone()
        .unwrap_or_else(|| "[CLS]".to_string());
    let sep = roles
        .sep_token
        .clone()
        .unwrap_or_else(|| "[SEP]".to_string());
    PipelineConfig {
        normalizer: Some(NormalizerConfig::Sequence {
            normalizers: vec![
                NormalizerConfig::Nfc,
                NormalizerConfig::Lowercase,
                NormalizerConfig::StripAccents,
            ],
        }),
        pre_tokenizer: Some(PreTokenizerConfig::BertPreTokenizer),
        model: ModelConfig::WordPiece {
            unk_token: unk,
            continuing_subword_prefix: "##".to_string(),
            max_input_chars_per_word: 100,
            vocab,
        },
        post_processor: Some(PostProcessorConfig::TemplateProcessing {
            single: vec![
                tok_post::TemplatePiece::special(cls.clone(), 0),
                tok_post::TemplatePiece::sequence("A", 0),
                tok_post::TemplatePiece::special(sep.clone(), 0),
            ],
            pair: vec![
                tok_post::TemplatePiece::special(cls, 0),
                tok_post::TemplatePiece::sequence("A", 0),
                tok_post::TemplatePiece::special(sep.clone(), 0),
                tok_post::TemplatePiece::sequence("B", 1),
                // HF 约定：句对模板收尾的 [SEP] 归第二段（type_id = 1）
                tok_post::TemplatePiece::special(sep, 1),
            ],
            special_tokens: Default::default(),
        }),
        decoder: Some(DecoderConfig::WordPiece {
            prefix: "##".to_string(),
            cleanup: true,
        }),
        added_tokens: Vec::new(),
        truncation: None,
        padding: None,
        model_max_length: 512,
        roles,
    }
}

/// 便捷构造：`tokenizer.json` 的直接 JSON 读取（跨实现校验用）。
pub fn load_tokenizer_json(path: &Path) -> Result<Tokenizer> {
    let text = std::fs::read_to_string(path)?;
    let value: Value = serde_json::from_str(&text)?;
    Tokenizer::from_tokenizer_json(&value)
}

/// 便捷构造：把 CLI 的 `--special` 之类的裸字符串转成 AddedToken。
pub fn specials(tokens: &[&str]) -> Vec<AddedToken> {
    tokens.iter().map(|s| AddedToken::new(*s, true)).collect()
}

/// 字节偏移 → 字符索引（非字符边界时回退到最近的边界，容错）。
/// 各 Model 输出的 offsets 以输入串的**字节**为单位，而 NormalizedString 的
/// 对齐表以**字符**为单位，此处负责单位换算。
fn byte_to_char_index(s: &str, byte: usize) -> usize {
    let byte = byte.min(s.len());
    let mut idx = byte;
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    s[..idx].chars().count()
}
