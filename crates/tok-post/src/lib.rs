//! # tok-post
//!
//! L5 后处理层：把 `Encoding`（可选 pair）装配成最终序列，
//! 同步维护 `type_ids` / `special_tokens_mask` / `sequence_ranges`。
//!
//! 支持 HF 的组网方式：`TemplateProcessing`（主流）、`BertProcessing`、`RobertaProcessing`、
//! `ByteLevel`（占位，实际裁剪在 L2 完成）、`Sequence`。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use tok_core::{layer, Encoding, PostProcessor, Result, TokenizerError};

/// 模板元素：特殊 token 或序列占位（`A` / `B`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TemplatePiece {
    SpecialToken(SpecialTokenPiece),
    Sequence(SequencePiece),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialTokenPiece {
    #[serde(rename = "SpecialToken")]
    pub special_token: SpecialTokenId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialTokenId {
    pub id: String,
    #[serde(default)]
    pub type_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequencePiece {
    #[serde(rename = "Sequence")]
    pub sequence: SequenceId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceId {
    /// `"A"` 或 `"B"`
    pub id: String,
    #[serde(default)]
    pub type_id: u32,
}

impl TemplatePiece {
    pub fn special(content: impl Into<String>, type_id: u32) -> Self {
        TemplatePiece::SpecialToken(SpecialTokenPiece {
            special_token: SpecialTokenId {
                id: content.into(),
                type_id,
            },
        })
    }
    pub fn sequence(id: &str, type_id: u32) -> Self {
        TemplatePiece::Sequence(SequencePiece {
            sequence: SequenceId {
                id: id.to_string(),
                type_id,
            },
        })
    }
}

/// `special_tokens` 表中的一项（HF 形状）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialTokenEntry {
    pub id: String,
    #[serde(default)]
    pub ids: Vec<u32>,
    #[serde(default)]
    pub tokens: Vec<String>,
}

/// 后处理器配置（HF `post_processor` 节）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PostProcessorConfig {
    #[serde(rename = "TemplateProcessing")]
    TemplateProcessing {
        #[serde(default)]
        single: Vec<TemplatePiece>,
        #[serde(default)]
        pair: Vec<TemplatePiece>,
        #[serde(default)]
        special_tokens: BTreeMap<String, SpecialTokenEntry>,
    },
    #[serde(rename = "BertProcessing")]
    BertProcessing {
        sep: (String, u32),
        cls: (String, u32),
    },
    #[serde(rename = "RobertaProcessing")]
    RobertaProcessing {
        sep: (String, u32),
        cls: (String, u32),
        #[serde(default)]
        trim_offsets: bool,
        #[serde(default)]
        add_prefix_space: bool,
    },
    #[serde(rename = "ByteLevel")]
    ByteLevel {
        #[serde(default)]
        add_prefix_space: bool,
        #[serde(default)]
        trim_offsets: bool,
        #[serde(default = "default_true")]
        use_regex: bool,
    },
    #[serde(rename = "Sequence")]
    Sequence {
        processors: Vec<PostProcessorConfig>,
    },
}

fn default_true() -> bool {
    true
}

/// 运行时后处理器：保留原始配置（保证 `to_config_json` 精确往返）+ 编译后的特殊 token 表。
#[derive(Debug, Clone)]
pub struct PostProcessorWrapper {
    cfg: Option<PostProcessorConfig>,
    /// 特殊 token 字面值 → id（`TemplateProcessing.special_tokens` 展开结果）
    table: BTreeMap<String, u32>,
}

impl PostProcessorWrapper {
    /// 无后处理：直接透传（HF 中 `post_processor` 缺省）。
    pub fn identity() -> Self {
        Self {
            cfg: None,
            table: BTreeMap::new(),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.cfg.is_none()
    }

    pub fn name(&self) -> &'static str {
        match &self.cfg {
            None => "None",
            Some(PostProcessorConfig::TemplateProcessing { .. }) => "TemplateProcessing",
            Some(PostProcessorConfig::BertProcessing { .. }) => "BertProcessing",
            Some(PostProcessorConfig::RobertaProcessing { .. }) => "RobertaProcessing",
            Some(PostProcessorConfig::ByteLevel { .. }) => "ByteLevel",
            Some(PostProcessorConfig::Sequence { .. }) => "Sequence",
        }
    }

    pub fn config(&self) -> Option<&PostProcessorConfig> {
        self.cfg.as_ref()
    }

    pub fn build(cfg: &PostProcessorConfig) -> Result<Self> {
        let mut table = BTreeMap::new();
        if let PostProcessorConfig::TemplateProcessing {
            single,
            pair,
            special_tokens,
        } = cfg
        {
            for (literal, entry) in special_tokens {
                let id = entry.ids.first().copied().ok_or_else(|| {
                    TokenizerError::config(format!(
                        "special_tokens['{}'].ids 为空，无法确定 id",
                        literal
                    ))
                })?;
                table.insert(literal.clone(), id);
                table.insert(entry.id.clone(), id);
            }
            for p in single.iter().chain(pair.iter()) {
                if let TemplatePiece::SpecialToken(s) = p {
                    if s.special_token.id.is_empty() {
                        return Err(TokenizerError::config("模板中的特殊 token id 为空"));
                    }
                }
            }
            if pair
                .iter()
                .any(|p| matches!(p, TemplatePiece::Sequence(s) if s.sequence.id == "B"))
                && single
                    .iter()
                    .any(|p| matches!(p, TemplatePiece::Sequence(s) if s.sequence.id == "B"))
            {
                return Err(TokenizerError::config("single 模板中不应出现序列 B"));
            }
        }
        Ok(Self {
            cfg: Some(cfg.clone()),
            table,
        })
    }

    /// 模板引用的特殊 token 字面值（`tok-pipeline` 用于补登记 added_tokens 后再调用本层）。
    pub fn required_specials(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut push = |s: String| {
            if !out.contains(&s) {
                out.push(s);
            }
        };
        match &self.cfg {
            Some(PostProcessorConfig::TemplateProcessing { single, pair, .. }) => {
                for p in single.iter().chain(pair.iter()) {
                    if let TemplatePiece::SpecialToken(s) = p {
                        push(s.special_token.id.clone());
                    }
                }
            }
            Some(PostProcessorConfig::BertProcessing { sep, cls }) => {
                push(cls.0.clone());
                push(sep.0.clone());
            }
            Some(PostProcessorConfig::RobertaProcessing { sep, cls, .. }) => {
                push(cls.0.clone());
                push(sep.0.clone());
            }
            Some(PostProcessorConfig::Sequence { processors }) => {
                for p in processors {
                    if let Ok(w) = PostProcessorWrapper::build(p) {
                        for s in w.required_specials() {
                            push(s);
                        }
                    }
                }
            }
            _ => {}
        }
        out
    }

    /// 特殊 token 字面值 → id。优先用模板表；缺失时要求调用方已把它登记进 `table`。
    pub fn resolve_id(&self, literal: &str) -> Result<u32> {
        self.table.get(literal).copied().ok_or_else(|| {
            TokenizerError::layer(
                layer::POST_PROCESSOR,
                format!(
                    "后处理器无法解析特殊 token '{}'：请先登记到模板 special_tokens 或词表",
                    literal
                ),
            )
        })
    }

    /// 用 `Vocabulary` 补齐模板表（HF 行为：模板中的特殊 token 必须存在于词表）。
    pub fn bind_vocab(&mut self, vocab: &tok_core::Vocabulary) -> Result<()> {
        for s in self.required_specials() {
            if !self.table.contains_key(&s) {
                let id = vocab.token_to_id(&s).ok_or_else(|| {
                    TokenizerError::layer(
                        layer::POST_PROCESSOR,
                        format!("特殊 token '{}' 不在词表中，无法装配模板", s),
                    )
                })?;
                self.table.insert(s, id);
            }
        }
        Ok(())
    }

    /// 主入口：把 1~2 条 `Encoding` 装配为最终 `Encoding`。
    pub fn process_encodings(
        &self,
        encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let is_pair = encodings.len() >= 2;
        match &self.cfg {
            None => Ok(concat_plain(encodings)),
            Some(PostProcessorConfig::TemplateProcessing { single, pair, .. }) => {
                let tpl = if is_pair { pair } else { single };
                let tpl: Vec<TemplatePiece> = if add_special_tokens {
                    tpl.clone()
                } else {
                    tpl.iter()
                        .filter(|p| !matches!(p, TemplatePiece::SpecialToken(_)))
                        .cloned()
                        .collect()
                };
                self.compose(&tpl, encodings)
            }
            Some(PostProcessorConfig::BertProcessing { sep, cls }) => {
                let tpl = bert_template(cls, sep, add_special_tokens, is_pair);
                self.compose(&tpl, encodings)
            }
            Some(PostProcessorConfig::RobertaProcessing { sep, cls, .. }) => {
                let tpl = roberta_template(cls, sep, add_special_tokens, is_pair);
                self.compose(&tpl, encodings)
            }
            Some(PostProcessorConfig::ByteLevel { .. }) => Ok(concat_plain(encodings)),
            Some(PostProcessorConfig::Sequence { processors }) => {
                let mut cur = encodings;
                for p in processors {
                    let mut w = PostProcessorWrapper::build(p)?;
                    w.table = self.table.clone();
                    cur = vec![w.process_encodings(cur, add_special_tokens)?];
                }
                Ok(cur.into_iter().next().unwrap_or_default())
            }
        }
    }

    fn compose(&self, tpl: &[TemplatePiece], encodings: Vec<Encoding>) -> Result<Encoding> {
        let mut out = Encoding::default();
        let mut seq_idx = 0usize;
        for piece in tpl {
            match piece {
                TemplatePiece::SpecialToken(s) => {
                    let id = self.resolve_id(&s.special_token.id)?;
                    push_special(&mut out, id, &s.special_token.id, s.special_token.type_id);
                }
                TemplatePiece::Sequence(seq) => {
                    if seq_idx >= encodings.len() {
                        continue;
                    }
                    let src = &encodings[seq_idx];
                    let start = out.ids.len();
                    let ty = seq.sequence.type_id;
                    for i in 0..src.ids.len() {
                        out.ids.push(src.ids[i]);
                        out.type_ids.push(ty);
                        out.tokens.push(src.tokens[i].clone());
                        out.offsets.push(src.offsets[i]);
                        out.attention_mask
                            .push(src.attention_mask.get(i).copied().unwrap_or(1));
                        out.special_tokens_mask
                            .push(src.special_tokens_mask.get(i).copied().unwrap_or(0));
                    }
                    out.offsets_degraded |= src.offsets_degraded;
                    let key = if seq.sequence.id.eq_ignore_ascii_case("B") {
                        1
                    } else {
                        0
                    };
                    out.sequence_ranges.insert(key, (start, out.ids.len()));
                    seq_idx += 1;
                }
            }
        }
        // 未被模板引用的后续序列按原顺序追加，避免静默丢数据
        while seq_idx < encodings.len() {
            let src = &encodings[seq_idx];
            let start = out.ids.len();
            for i in 0..src.ids.len() {
                out.ids.push(src.ids[i]);
                out.type_ids.push(src.type_ids.get(i).copied().unwrap_or(0));
                out.tokens.push(src.tokens[i].clone());
                out.offsets.push(src.offsets[i]);
                out.attention_mask
                    .push(src.attention_mask.get(i).copied().unwrap_or(1));
                out.special_tokens_mask
                    .push(src.special_tokens_mask.get(i).copied().unwrap_or(0));
            }
            out.offsets_degraded |= src.offsets_degraded;
            out.sequence_ranges.insert(seq_idx, (start, out.ids.len()));
            seq_idx += 1;
        }
        out.validate()?;
        Ok(out)
    }
}

fn bert_template(
    cls: &(String, u32),
    sep: &(String, u32),
    add_special: bool,
    pairs: bool,
) -> Vec<TemplatePiece> {
    let mut v = Vec::new();
    if add_special {
        v.push(TemplatePiece::special(cls.0.clone(), cls.1));
    }
    v.push(TemplatePiece::sequence("A", 0));
    if add_special {
        v.push(TemplatePiece::special(sep.0.clone(), sep.1));
    }
    if pairs {
        // HF `BertProcessing` 的 pair 形态：`[CLS] A [SEP] B [SEP]`，
        // B 段及其后的 [SEP] 均属 type_id = 1（首段含首个 [SEP] 为 0）。
        v.push(TemplatePiece::sequence("B", 1));
        if add_special {
            v.push(TemplatePiece::special(sep.0.clone(), 1));
        }
    }
    v
}

fn roberta_template(
    cls: &(String, u32),
    sep: &(String, u32),
    add_special: bool,
    pairs: bool,
) -> Vec<TemplatePiece> {
    let mut v = Vec::new();
    if add_special {
        v.push(TemplatePiece::special(cls.0.clone(), cls.1));
    }
    v.push(TemplatePiece::sequence("A", 0));
    if add_special {
        v.push(TemplatePiece::special(sep.0.clone(), sep.1));
    }
    if pairs {
        // HF `RobertaProcessing` 的 pair 形态：`<s> A </s> </s> B </s>`，type_id 全 0。
        if add_special {
            v.push(TemplatePiece::special(sep.0.clone(), sep.1));
        }
        v.push(TemplatePiece::sequence("B", 0));
        if add_special {
            v.push(TemplatePiece::special(sep.0.clone(), sep.1));
        }
    }
    v
}

/// Bert/Roberta 的 pair 模板由 `process_encodings` 内部按条数选择，此处只需处理"两段拼接"。
fn concat_plain(encodings: Vec<Encoding>) -> Encoding {
    let mut out = Encoding::default();
    for (idx, src) in encodings.into_iter().enumerate() {
        let start = out.ids.len();
        for i in 0..src.ids.len() {
            out.ids.push(src.ids[i]);
            out.type_ids.push(src.type_ids.get(i).copied().unwrap_or(0));
            out.tokens.push(src.tokens[i].clone());
            out.offsets.push(src.offsets[i]);
            out.attention_mask
                .push(src.attention_mask.get(i).copied().unwrap_or(1));
            out.special_tokens_mask
                .push(src.special_tokens_mask.get(i).copied().unwrap_or(0));
        }
        out.offsets_degraded |= src.offsets_degraded;
        out.sequence_ranges.insert(idx, (start, out.ids.len()));
    }
    out
}

fn push_special(out: &mut Encoding, id: u32, token: &str, type_id: u32) {
    out.ids.push(id);
    out.type_ids.push(type_id);
    out.tokens.push(token.to_string());
    out.offsets.push((0, 0));
    out.attention_mask.push(1);
    out.special_tokens_mask.push(1);
}

impl PostProcessor for PostProcessorWrapper {
    fn process(
        &self,
        encoding: Encoding,
        pair: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let mut v = vec![encoding];
        if let Some(p) = pair {
            v.push(p);
        }
        self.process_encodings(v, add_special_tokens)
    }

    fn added_tokens(&self, is_pair: bool) -> usize {
        match &self.cfg {
            None => 0,
            Some(PostProcessorConfig::TemplateProcessing { single, pair, .. }) => {
                let t = if is_pair { pair } else { single };
                t.iter()
                    .filter(|p| matches!(p, TemplatePiece::SpecialToken(_)))
                    .count()
            }
            Some(PostProcessorConfig::BertProcessing { .. }) => 2,
            Some(PostProcessorConfig::RobertaProcessing { .. }) => {
                if is_pair {
                    4
                } else {
                    2
                }
            }
            Some(PostProcessorConfig::ByteLevel { .. }) => 0,
            Some(PostProcessorConfig::Sequence { processors }) => processors
                .iter()
                .filter_map(|p| PostProcessorWrapper::build(p).ok())
                .map(|w| w.added_tokens(is_pair))
                .sum(),
        }
    }

    fn to_config_json(&self) -> Result<serde_json::Value> {
        match &self.cfg {
            None => Ok(serde_json::Value::Null),
            Some(PostProcessorConfig::TemplateProcessing {
                single,
                pair,
                special_tokens,
            }) => {
                // HF 要求 `TemplateProcessing.special_tokens` 自带 ids/tokens：把 `bind_vocab`
                // 解析出的 id 回填后才能被外部实现（HF `tokenizers`）自解析；
                // 已有条目保持原样，保证配置精确往返。
                let mut map = special_tokens.clone();
                for literal in self.required_specials() {
                    let Some(id) = self.table.get(&literal).copied() else {
                        continue;
                    };
                    let entry = map
                        .entry(literal.clone())
                        .or_insert_with(|| SpecialTokenEntry {
                            id: literal.clone(),
                            ids: Vec::new(),
                            tokens: Vec::new(),
                        });
                    if entry.ids.is_empty() {
                        entry.ids.push(id);
                    }
                    if entry.tokens.is_empty() {
                        entry.tokens.push(literal.clone());
                    }
                }
                let filled = PostProcessorConfig::TemplateProcessing {
                    single: single.clone(),
                    pair: pair.clone(),
                    special_tokens: map,
                };
                Ok(serde_json::to_value(&filled)?)
            }
            Some(c) => Ok(serde_json::to_value(c)?),
        }
    }

    fn name(&self) -> &'static str {
        PostProcessorWrapper::name(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tok_core::{AddedToken, PostProcessor, Vocabulary};

    fn enc(ids: &[u32], toks: &[&str]) -> Encoding {
        let mut e = Encoding::default();
        for (i, id) in ids.iter().enumerate() {
            e.ids.push(*id);
            e.type_ids.push(0);
            e.tokens.push(toks[i].to_string());
            e.offsets.push((i, i + 1));
            e.attention_mask.push(1);
            e.special_tokens_mask.push(0);
        }
        e
    }

    fn cls_sep_vocab() -> Vocabulary {
        let mut v = Vocabulary::new();
        v.add_special(AddedToken::new("[CLS]", true), 0).unwrap();
        v.add_special(AddedToken::new("[SEP]", true), 1).unwrap();
        v
    }

    #[test]
    fn bert_processor_wraps_single_and_pair() {
        let mut p = PostProcessorWrapper::build(&PostProcessorConfig::BertProcessing {
            sep: ("[SEP]".into(), 0),
            cls: ("[CLS]".into(), 0),
        })
        .unwrap();
        // 模板引用的特殊 token 必须先绑定词表才能解析出 id（L5 契约）
        p.bind_vocab(&cls_sep_vocab()).unwrap();
        let single = p
            .process_encodings(vec![enc(&[10, 11], &["a", "b"])], true)
            .unwrap();
        assert_eq!(single.ids, vec![0, 10, 11, 1]);
        let pair = p
            .process_encodings(vec![enc(&[10, 11], &["a", "b"]), enc(&[12], &["c"])], true)
            .unwrap();
        assert_eq!(pair.ids, vec![0, 10, 11, 1, 12, 1]);
        assert_eq!(pair.type_ids, vec![0, 0, 0, 0, 1, 1]);
        assert_eq!(pair.special_tokens_mask, vec![1, 0, 0, 1, 0, 1]);
        pair.validate().unwrap();
        assert_eq!(pair.sequence_ranges.get(&0).copied(), Some((1, 3)));
        assert_eq!(pair.sequence_ranges.get(&1).copied(), Some((4, 5)));
    }

    #[test]
    fn add_special_tokens_false_drops_specials() {
        let p = PostProcessorWrapper::build(&PostProcessorConfig::BertProcessing {
            sep: ("[SEP]".into(), 0),
            cls: ("[CLS]".into(), 0),
        })
        .unwrap();
        let out = p
            .process_encodings(vec![enc(&[10], &["a"])], false)
            .unwrap();
        assert_eq!(out.ids, vec![10]);
        assert_eq!(out.special_tokens_mask, vec![0]);
    }

    #[test]
    fn template_processing_round_trip() {
        let cfg: PostProcessorConfig = serde_json::from_str(
            r#"{
              "type": "TemplateProcessing",
              "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
              "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
              "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}, "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}}
            }"#,
        )
        .unwrap();
        let p = PostProcessorWrapper::build(&cfg).unwrap();
        assert_eq!(
            p.required_specials(),
            vec!["<s>".to_string(), "</s>".to_string()]
        );
        assert_eq!(p.added_tokens(false), 2);
        assert_eq!(p.added_tokens(true), 2);
        let out = p.process_encodings(vec![enc(&[5], &["x"])], true).unwrap();
        assert_eq!(out.ids, vec![1, 5, 2]);
        // 精确往返
        let back: PostProcessorConfig =
            serde_json::from_value(p.to_config_json().unwrap()).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn bind_vocab_fills_missing_special() {
        let cfg = PostProcessorConfig::BertProcessing {
            sep: ("[SEP]".into(), 0),
            cls: ("[CLS]".into(), 0),
        };
        let mut p = PostProcessorWrapper::build(&cfg).unwrap();
        let mut v = cls_sep_vocab();
        v.add_special(AddedToken::new("[MASK]", true), 2).unwrap();
        p.bind_vocab(&v).unwrap();
        assert_eq!(p.resolve_id("[CLS]").unwrap(), 0);
        assert!(p.process_encodings(vec![enc(&[9], &["z"])], true).is_ok());
    }

    #[test]
    fn identity_passthrough() {
        let p = PostProcessorWrapper::identity();
        assert!(p.is_identity());
        let out = p
            .process_encodings(vec![enc(&[3, 4], &["a", "b"])], true)
            .unwrap();
        assert_eq!(out.ids, vec![3, 4]);
        assert_eq!(out.sequence_ranges.get(&0).copied(), Some((0, 2)));
        assert_eq!(p.to_config_json().unwrap(), serde_json::Value::Null);
    }
}
