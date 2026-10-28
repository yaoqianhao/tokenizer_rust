//! 门面 `Tokenizer`：装配 L1..L6 并实现端到端 `encode` / `decode` / `train` / 落盘。
//!
//! 依赖方向：`tok-pipeline` → 六个实现 crate；实现 crate 之间互不依赖。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ahash::AHashSet;
use rayon::prelude::*;
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
    /// 特殊 token 的 id 集合（`encode` 热路径用 O(1) 判定，避免逐 token 遍历 added 表）。
    /// 由 `register_added` / `train` 后经 `rebuild_special_set` 刷新。
    special_set: AHashSet<u32>,
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
            special_set: AHashSet::new(),
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
        // 显式配置的角色优先，缺失槽位用 added_tokens 的启发式识别补齐。
        // 旧实现只在 `roles` 整体为空时才 infer，导致外部模型经 `from_tokenizer_json`
        // 装配时（roles 全空但 unk 能从 model 推出来）丢掉 [CLS]/[SEP]/[MASK] 等角色。
        let inferred = TokenRoles::infer(&tk.added_tokens);
        tk.roles.fill_missing(&inferred);
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
        // 特殊 token 角色：HF `tokenizer.json` 无独立 roles 字段，从 added_tokens 推断；
        // unk 缺失时回退到模型节声明的 unk_token（BPE / WordPiece / WordLevel 才带）。
        let mut roles = TokenRoles::infer(&added_tokens);
        if roles.unk_token.is_none() {
            roles.unk_token = match &model_cfg {
                ModelConfig::Bpe { unk_token, .. } => unk_token.clone(),
                ModelConfig::WordPiece { unk_token, .. } => Some(unk_token.clone()),
                ModelConfig::WordLevel { unk_token, .. } => Some(unk_token.clone()),
                ModelConfig::Unigram { .. } => None,
            };
        }
        // 截断/填充：仅当字段存在且非 null 时解析，解析失败按配置错误上抛（不再静默丢弃）
        let truncation = match value.get("truncation") {
            None | Some(Value::Null) => None,
            Some(v) => Some(serde_json::from_value::<TruncationParams>(v.clone())?),
        };
        let padding = match value.get("padding") {
            None | Some(Value::Null) => None,
            Some(v) => Some(serde_json::from_value::<PaddingParams>(v.clone())?),
        };
        let cfg = PipelineConfig {
            normalizer,
            pre_tokenizer,
            model: model_cfg,
            post_processor,
            decoder,
            added_tokens,
            truncation,
            padding,
            model_max_length: default_max_len(),
            roles,
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

    /// 替换后处理器（`None` 关闭）。
    ///
    /// 训练前设置会在 `from_config` 阶段因空词表无法 `bind_vocab`，因此典型用法是
    /// 训练完成后调用：写入的配置会同步到 `PipelineConfig`，再 `bind_vocab` 到当前词表。
    pub fn set_post_processor(&mut self, cfg: Option<PostProcessorConfig>) -> Result<()> {
        let normalized = match cfg {
            None => None,
            Some(c) => Some(self.bind_post_processor(c)?),
        };
        let wrapper = match &normalized {
            None => PostProcessorWrapper::identity(),
            Some(c) => {
                let mut w = PostProcessorWrapper::build(c)?;
                w.bind_vocab(self.model.get_vocab())?;
                w
            }
        };
        self.post_processor = wrapper;
        self.config.post_processor = normalized;
        Ok(())
    }

    /// 替换解码器（`None` 关闭）；与词表强相关的约束（如 WordPiece `##` 前缀）在此校验。
    pub fn set_decoder(&mut self, cfg: Option<DecoderConfig>) -> Result<()> {
        let wrapper = match &cfg {
            None => None,
            Some(d) => Some(DecoderWrapper::build(d)?),
        };
        self.decoder = wrapper;
        self.config.decoder = cfg;
        self.refresh_decoder_binding()
    }

    /// 把配置里的占位 id 换成词表真实 id，并给模板补齐 `special_tokens` 表。
    ///
    /// HF 的 `BertProcessing` / `RobertaProcessing` 把 `(token, id)` 写死进配置，
    /// 直接沿用用户传的占位 id 会让导出的 `tokenizer.json` 与实际词表错位，因此统一回填。
    fn bind_post_processor(&self, cfg: PostProcessorConfig) -> Result<PostProcessorConfig> {
        let vocab = self.model.get_vocab();
        let mut probe = PostProcessorWrapper::build(&cfg)?;
        probe.bind_vocab(vocab)?;
        let fixed = |pair: &(String, u32)| -> Result<(String, u32)> {
            Ok((pair.0.clone(), probe.resolve_id(&pair.0)?))
        };
        Ok(match cfg {
            PostProcessorConfig::BertProcessing { sep, cls } => {
                PostProcessorConfig::BertProcessing {
                    sep: fixed(&sep)?,
                    cls: fixed(&cls)?,
                }
            }
            PostProcessorConfig::RobertaProcessing {
                sep,
                cls,
                trim_offsets,
                add_prefix_space,
            } => PostProcessorConfig::RobertaProcessing {
                sep: fixed(&sep)?,
                cls: fixed(&cls)?,
                trim_offsets,
                add_prefix_space,
            },
            PostProcessorConfig::TemplateProcessing {
                single,
                pair,
                mut special_tokens,
            } => {
                for p in single.iter().chain(pair.iter()) {
                    if let tok_post::TemplatePiece::SpecialToken(s) = p {
                        let literal = s.special_token.id.clone();
                        if !special_tokens.contains_key(&literal) {
                            let id = probe.resolve_id(&literal)?;
                            special_tokens.insert(
                                literal.clone(),
                                tok_post::SpecialTokenEntry {
                                    id: literal.clone(),
                                    ids: vec![id],
                                    tokens: vec![literal],
                                },
                            );
                        }
                    }
                }
                PostProcessorConfig::TemplateProcessing {
                    single,
                    pair,
                    special_tokens,
                }
            }
            PostProcessorConfig::Sequence { processors } => PostProcessorConfig::Sequence {
                processors: processors
                    .into_iter()
                    .map(|p| self.bind_post_processor(p))
                    .collect::<Result<Vec<_>>>()?,
            },
            other => other,
        })
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
        self.rebuild_special_set();
        Ok(count)
    }

    /// 显式设定角色 token（E6 端到端联动）。
    ///
    /// 语义：
    /// 1. 先做自洽性校验（同一 token 不得占两个角色槽位，也不得与
    ///    `additional_special_tokens` 重复）；
    /// 2. 每个角色 token 都登记为 **special** 的 added token——HF 的
    ///    `special_tokens_mask` / `skip_special_tokens` / `added_tokens_decoder`
    ///    都按 added 视图判定，仅写进 `tokenizer_config.json` 而不登记会与 HF 判定不一致；
    /// 3. 若该 token 已作为**非 special** 的 added token 存在（如用户用 `--special`
    ///    加进来、但语义上应是 cls），就地升级为 special，保持 id 不变。
    pub fn set_roles(&mut self, roles: TokenRoles) -> Result<()> {
        roles.validate()?;
        let wanted = roles.all();
        // 逐个对齐 special 标记：把既有的非 special 后加 token 就地升级（保 id），
        // 再补登记完全缺失的角色 token。
        for tok in &wanted {
            let already_special = self
                .added_tokens
                .iter()
                .any(|t| t.content == *tok && t.special);
            if already_special {
                continue;
            }
            if self.model.get_vocab_mut().promote_added_special(tok) {
                if let Some(slot) = self.added_tokens.iter_mut().find(|t| t.content == *tok) {
                    slot.special = true;
                }
            }
        }
        let mut to_register: Vec<AddedToken> = Vec::new();
        roles.ensure_in_added(&mut to_register);
        let to_register: Vec<AddedToken> = to_register
            .into_iter()
            .filter(|t| !self.added_tokens.iter().any(|a| a.content == t.content))
            .collect();
        if !to_register.is_empty() {
            self.register_added(&to_register)?;
        }
        self.roles = roles;
        self.rebuild_special_set();
        Ok(())
    }

    /// 重建特殊 token id 集合（`register_added` / `train` 之后调用）。
    fn rebuild_special_set(&mut self) {
        let set: AHashSet<u32> = self
            .model
            .get_vocab()
            .added()
            .tokens()
            .iter()
            .filter(|(t, _)| t.special)
            .map(|(_, id)| *id)
            .collect();
        self.special_set = set;
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

    /// 批量编码（`rayon` 并行，结果顺序与输入严格对应）。
    pub fn encode_batch(
        &self,
        texts: &[String],
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        // 每条文本的编码相互独立、且 `encode` 只读 `&self`，可直接并行。
        // 顺序由 rayon 的 indexed collect 保证，与串行结果逐条对应。
        // 不做显式分块：任务粒度由 rayon 的工作窃取自行切分，每条文本的中间缓冲
        // 随元素生命周期结束即释放，内存峰值由**单条最长文本**决定，与批量大小无关。
        texts
            .par_iter()
            .map(|t| self.encode(t, add_special_tokens))
            .collect()
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
        // S1 → S0 顺序与 HF 一致：**先按原文切出后加 token，再对每个正文片段单独规范化**。
        //
        // HF 的 `AddedVocabulary::extract_and_normalize` 就是「先 split added tokens，
        // 再逐片 normalize」，因此 `Prepend` / `Strip` 这类非幂等归一化会作用在**每个片段**
        // 上（`"a [MASK] b"` → `"a [MASK]"` 与 `" b"` 两片各自 prepend 出 `▁`）。
        // 早期实现是「整串先规范、再切 added token」，导致带 added token 的文本归一化结果
        // 与 HF 不一致（E4 对照：`▁` 只出现在开头）。
        let units = self.model.get_vocab().added().split(text);
        let mut enc = Encoding::default();
        let mut degraded = false;
        // 字符起点表：字节偏移 → 字符索引的换算从「每次从头数」变成「一次构建 + 二分」，
        // 否则每个子词都要 `chars().count()` 一遍原文，编码整体退化为 O(n²)。
        let char_starts = char_index_table(text);

        for unit in units {
            match unit.id {
                Some(id) => {
                    // 后加 token 直接落在原文区间上，无需经归一化对齐表换算
                    let (s, e) = (
                        byte_to_char_with(&char_starts, unit.start),
                        byte_to_char_with(&char_starts, unit.end),
                    );
                    let special = self.special_set.contains(&id);
                    enc.ids.push(id);
                    enc.type_ids.push(0);
                    enc.tokens.push(unit.text.clone());
                    enc.offsets.push((s, e));
                    enc.attention_mask.push(1);
                    enc.special_tokens_mask.push(if special { 1 } else { 0 });
                }
                None => {
                    // ---- S0 规范化（片段内，保持 original↔normalized 对齐）
                    let chunk = &text[unit.start..unit.end];
                    if chunk.is_empty() {
                        continue;
                    }
                    let mut local = NormalizedString::new(chunk);
                    if let Some(n) = &self.normalizer {
                        local = n.normalize(local);
                    }
                    degraded |= local.is_degraded();
                    // ---- S2 预分词（片段内切分）
                    if let Some(p) = &self.pre_tokenizer {
                        p.pre_tokenize(&mut local)?;
                    }
                    // 子词区间在同一片段内单调递增，游标可复用（摊还 O(1)）。
                    let mut sw_cursor = 0usize;
                    for pt in local.pretokens() {
                        // `pretokens()` 给出的 `offsets` 已是**本片段原文**坐标（内部已换算），
                        // 这里只需加片段基址；`n_offsets` 才是规范化串坐标，留给子词级换算。
                        let outer = (unit.start + pt.offsets.0, unit.start + pt.offsets.1);
                        // ---- S3 子词切分
                        let subwords = self.model.tokenize(&pt.value)?;
                        for sw in subwords {
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
                            if std::env::var("TOK_DEBUG_OFF").is_ok() {
                                let rr = local.convert_offsets_from(rel, &mut 0usize);
                                eprintln!("DBG rel={:?} -> {:?} (no-cursor)", rel, rr);
                            }
                            let (s, e) = match local.convert_offsets_from(rel, &mut sw_cursor) {
                                Some(o) => (unit.start + o.start, unit.start + o.end),
                                // 对齐表未命中：退化为整个预分词单元区间
                                None => outer,
                            };
                            // 对外语义与 HF 对齐：offsets 输出**原文字符**区间（内部全程为字节）。
                            // 注意：特殊 token 也保留真实区间（HF 的 offsets 不因 special 置零，
                            // 只有 special_tokens_mask 标 1），因此这里不再对 is_special 特判。
                            let (s, e) = byte_range_to_chars(&char_starts, s, e);
                            enc.offsets.push((s, e));
                            enc.attention_mask.push(1);
                            // 模型产出的 token 一律 mask=0：HF 的 special_tokens_mask 只标
                            // 1）模板插入的特殊 token，2）在原文中字面出现的 added special token
                            // （走上面的 added-token 分支）。未登录词替换出来的 [UNK] 属于
                            // 模型产出，HF 标记为 0；若误标 1，`Encoding::truncate` 会把它当
                            // 特殊 token 排除在内容预算外，导致截断长度与 HF 差一（E5 对照暴露）。
                            enc.special_tokens_mask.push(0);
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
                // HF 在 decoder 为 None 时不会做任何拼接语义处理，而是把 token 以**单个空格**连接
                // （`TokenizerImpl::decode` 末尾统一 `tokens.join(" ")`）。实测 ▁Hello/▁world!
                // 两 token 的 HF 结果是 `▁Hello ▁world!`，故此处必须用空格而非直接 concat。
                let specials = self.special_ids();
                let mut kept: Vec<String> = Vec::new();
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
                    kept.push(t);
                }
                Ok(kept.join(" "))
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

/// 从 HF 目录（`save_pretrained` 的产物）**完整**恢复一个 `Tokenizer`。
///
/// 与 `load_tokenizer_json` 的差别：`tokenizer.json` 只描述模型/流水线/added_tokens，
/// **不承载角色语义**（哪个是 cls、哪个是 sep）。角色在 HF 侧由
/// `special_tokens_map.json`（或 `tokenizer_config.json`）声明，所以按目录加载时必须
/// 把这两份文件读回来，否则 `encode_pair` 会退化成"无后处理"、`pad_token_id` 也拿不到。
///
/// 恢复优先级：`special_tokens_map.json` 显式值 → `tokenizer_config.json` 显式值 →
/// 由 added_tokens 名字推断（`TokenRoles::infer`）→ `fill_missing` 补齐其余。
pub fn load_pretrained(dir: &Path) -> Result<Tokenizer> {
    let dir = if dir.is_dir() {
        dir.to_path_buf()
    } else {
        dir.parent().unwrap_or(Path::new(".")).to_path_buf()
    };
    let tj_path = dir.join(hf::TOKENIZER_JSON);
    let mut tk = load_tokenizer_json(&tj_path)?;

    let read_json = |name: &str| -> Option<Value> {
        std::fs::read_to_string(dir.join(name))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
    };

    let from_map = read_json(hf::SPECIAL_TOKENS_MAP_JSON)
        .map(|v| TokenRoles::from_hf_json(&v))
        .unwrap_or_default();
    let from_cfg = read_json(hf::TOKENIZER_CONFIG_JSON).unwrap_or(Value::Null);

    let inferred = TokenRoles::infer(tk.added_tokens());
    let mut roles = from_map;
    roles.fill_missing(&TokenRoles::from_hf_json(&from_cfg));
    roles.fill_missing(&inferred);
    tk.set_roles(roles)?;

    if let Some(len) = from_cfg
        .get("model_max_length")
        .and_then(|v| v.as_u64())
        .filter(|v| *v > 0)
    {
        tk.model_max_length = len as usize;
    }
    Ok(tk)
}

/// 便捷构造：把 CLI 的 `--special` 之类的裸字符串转成 AddedToken。
pub fn specials(tokens: &[&str]) -> Vec<AddedToken> {
    tokens.iter().map(|s| AddedToken::new(*s, true)).collect()
}

/// 字符起点表：每个字符的起始**字节**偏移（升序）。
/// 各 Model 输出的 offsets 以输入串的**字节**为单位，对外则要求**字符**索引；
/// 一次构建此表可让每次换算走二分，避免逐 token 重数字符串。
fn char_index_table(s: &str) -> Vec<usize> {
    s.char_indices().map(|(i, _)| i).collect()
}

/// 字节区间 → 字符区间：起点**向下**取到所在字符，终点**向上**取到字符末尾。
///
/// 子词在字节级映射后可能只覆盖某个多字节字符的一部分（例如 `你` 的 3 个 UTF-8 字节
/// 各自成一个 token），此时 HF 的 offsets 仍给出该字符的完整区间（三个 token 都是
/// `(0,1)`）。用 `byte_to_char_with` 同时处理两端会把起点推到下一个字符，产出 `(1,1)`
/// 这类零宽/错位区间，因此起止必须分别向下 / 向上取整。
fn byte_range_to_chars(
    char_starts: &[usize],
    start_byte: usize,
    end_byte: usize,
) -> (usize, usize) {
    let start = char_starts
        .partition_point(|&b| b <= start_byte)
        .saturating_sub(1);
    let end = char_starts.partition_point(|&b| b < end_byte);
    (start, end.max(start))
}

/// 字节偏移 → 字符索引（非字符边界时向下取到最近的边界，容错）。
fn byte_to_char_with(char_starts: &[usize], byte: usize) -> usize {
    // 统计起点严格小于 byte 的字符数：边界处得到该字符的序号，
    // 非边界处自然向下取整，与旧的 `s[..idx].chars().count()` 等价。
    char_starts.partition_point(|&start| start < byte)
}
