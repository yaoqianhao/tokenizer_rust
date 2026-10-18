//! # tok-models
//!
//! L3 模型层：BPE / WordPiece / WordLevel / Unigram 四种算法实现与对应训练器。
//!
//! 设计要点（架构方案 §5）：
//! - L3 是**唯一**允许出现算法分歧的层，其余层只依赖 `Model` trait；
//! - `ModelConfig` 的 JSON 形状与 HF `tokenizer.json` 的 `model` 节一致，可双向互转；
//! - 训练产物统一走 `TrainerArtifact`，模型侧用 `apply_artifact` 消费。

pub mod bpe;
pub mod unigram;
pub mod wordlevel;
pub mod wordpiece;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use tok_core::{Model, Result, Token, TokenizerError, Trainer, TrainerArtifact, TrainerParams, Vocabulary, layer};

pub use bpe::{BpeModel, BpeTrainer, MergeSpec, Symbol, bpe_from_artifact, encode_words};
pub use unigram::{UnigramModel, UnigramTrainer};
pub use wordlevel::{WordLevelModel, WordLevelTrainer};
pub use wordpiece::{WordPieceLegacyConfig, WordPieceModel, WordPieceTrainer};

fn default_continue_prefix() -> String {
    "##".to_string()
}
fn default_max_chars() -> usize {
    100
}

/// 模型配置：与 HF `tokenizer.json` 的 `model` 节同构（`type` 作为标签）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModelConfig {
    #[serde(rename = "BPE")]
    Bpe {
        #[serde(default)]
        dropout: Option<f32>,
        #[serde(default)]
        unk_token: Option<String>,
        #[serde(default)]
        continuing_subword_prefix: Option<String>,
        #[serde(default)]
        end_of_word_suffix: Option<String>,
        #[serde(default)]
        fuse_unk: bool,
        #[serde(default)]
        byte_fallback: bool,
        #[serde(default)]
        ignore_merges: bool,
        vocab: BTreeMap<String, u32>,
        merges: Vec<MergeSpec>,
    },
    #[serde(rename = "WordPiece")]
    WordPiece {
        #[serde(default)]
        unk_token: String,
        #[serde(default = "default_continue_prefix")]
        continuing_subword_prefix: String,
        #[serde(default = "default_max_chars")]
        max_input_chars_per_word: usize,
        vocab: BTreeMap<String, u32>,
    },
    #[serde(rename = "WordLevel")]
    WordLevel {
        #[serde(default)]
        unk_token: String,
        vocab: BTreeMap<String, u32>,
    },
    #[serde(rename = "Unigram")]
    Unigram {
        #[serde(default)]
        unk_id: Option<u32>,
        #[serde(default)]
        byte_fallback: bool,
        vocab: Vec<(String, f64)>,
    },
}

impl ModelConfig {
    /// 校验：词表非空、ID 连续无空洞（INV-3 的配置侧前置检查）。
    pub fn validate(&self) -> Result<()> {
        match self {
            ModelConfig::Bpe { vocab, merges, .. } => {
                check_vocab_map(vocab)?;
                for m in merges {
                    m.to_pair()?;
                }
            }
            ModelConfig::WordPiece { vocab, .. } => check_vocab_map(vocab)?,
            ModelConfig::WordLevel { vocab, .. } => check_vocab_map(vocab)?,
            ModelConfig::Unigram { vocab, .. } => {
                if vocab.is_empty() {
                    return Err(TokenizerError::config("Unigram 词表为空"));
                }
            }
        }
        Ok(())
    }

    /// 词表是否为空。BPE / WordPiece / WordLevel 的**空词表是合法训练起点**，
    /// 此时豁免 INV-3（ID 连续性）检查；Unigram 无空词表语义，一律交由 `validate` 报错。
    pub fn is_empty_vocab(&self) -> bool {
        match self {
            ModelConfig::Bpe { vocab, .. }
            | ModelConfig::WordPiece { vocab, .. }
            | ModelConfig::WordLevel { vocab, .. } => vocab.is_empty(),
            ModelConfig::Unigram { .. } => false,
        }
    }
}

fn check_vocab_map(vocab: &BTreeMap<String, u32>) -> Result<()> {
    if vocab.is_empty() {
        return Err(TokenizerError::config("模型词表为空"));
    }
    let max = vocab.values().copied().max().unwrap_or(0) as usize;
    let mut used = vec![false; max + 1];
    for id in vocab.values() {
        used[*id as usize] = true;
    }
    if let Some(pos) = used.iter().position(|u| !u) {
        return Err(TokenizerError::config(format!(
            "词表 ID 存在空洞：id {} 缺失（共 {} 个 token，最大 id {}）",
            pos,
            vocab.len(),
            max
        )));
    }
    Ok(())
}

/// 模型包装器：把四种算法统一到 `Model` trait 之后，供 `tok-pipeline` 装配。
pub enum ModelWrapper {
    Bpe(BpeModel),
    WordPiece(WordPieceModel),
    WordLevel(WordLevelModel),
    Unigram(UnigramModel),
}

impl std::fmt::Debug for ModelWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ModelWrapper({})", self.type_name())
    }
}

impl ModelWrapper {
    pub fn type_name(&self) -> &'static str {
        match self {
            ModelWrapper::Bpe(_) => "BPE",
            ModelWrapper::WordPiece(_) => "WordPiece",
            ModelWrapper::WordLevel(_) => "WordLevel",
            ModelWrapper::Unigram(_) => "Unigram",
        }
    }

    pub fn as_bpe(&self) -> Option<&BpeModel> {
        match self {
            ModelWrapper::Bpe(m) => Some(m),
            _ => None,
        }
    }

    /// 由配置构造运行时模型（BPE 的 merges 在此编译为 rank 表）。
    ///
    /// 空词表（训练起点）豁免 ID 连续性检查，与 `Tokenizer::from_config` 的口径保持一致；
    /// 非空词表仍走完整 `validate`，保证"已装配的模型"必满足 INV-3。
    pub fn build(cfg: &ModelConfig) -> Result<Self> {
        if !cfg.is_empty_vocab() {
            cfg.validate()?;
        }
        Ok(match cfg {
            ModelConfig::Bpe {
                dropout,
                unk_token,
                continuing_subword_prefix,
                end_of_word_suffix,
                fuse_unk,
                byte_fallback,
                ignore_merges,
                vocab,
                merges,
            } => {
                let mut v = Vocabulary::new();
                for (t, id) in vocab {
                    v.insert(t, *id)?;
                }
                let mut pairs = Vec::with_capacity(merges.len());
                for m in merges {
                    pairs.push(m.to_pair()?);
                }
                let mut m = BpeModel::new(v, pairs);
                m.dropout = *dropout;
                m.unk_token = unk_token.clone();
                m.continuing_subword_prefix = continuing_subword_prefix.clone();
                m.end_of_word_suffix = end_of_word_suffix.clone();
                m.fuse_unk = *fuse_unk;
                m.byte_fallback = *byte_fallback;
                m.ignore_merges = *ignore_merges;
                ModelWrapper::Bpe(m)
            }
            ModelConfig::WordPiece { unk_token, continuing_subword_prefix, max_input_chars_per_word, vocab } => {
                let mut v = Vocabulary::new();
                for (t, id) in vocab {
                    v.insert(t, *id)?;
                }
                let mut m = WordPieceModel::new(v, unk_token.clone());
                m.continuing_subword_prefix = continuing_subword_prefix.clone();
                m.max_input_chars_per_word = *max_input_chars_per_word;
                ModelWrapper::WordPiece(m)
            }
            ModelConfig::WordLevel { unk_token, vocab } => {
                let mut v = Vocabulary::new();
                for (t, id) in vocab {
                    v.insert(t, *id)?;
                }
                ModelWrapper::WordLevel(WordLevelModel::new(v, unk_token.clone()))
            }
            ModelConfig::Unigram { vocab, .. } => {
                ModelWrapper::Unigram(UnigramModel::from_pieces(vocab.clone(), "<unk>")?)
            }
        })
    }

    /// 从 HF `tokenizer.json` 的 `model` 节加载（含 `merges` 两种写法、`vocab` 两种形态）。
    pub fn from_hf_json(value: &serde_json::Value) -> Result<Self> {
        let cfg: ModelConfig = serde_json::from_value(value.clone())
            .map_err(|e| TokenizerError::config(format!("解析 model 节失败: {}", e)))?;
        Self::build(&cfg)
    }

    /// 导出为 HF `tokenizer.json` 的 `model` 节。
    pub fn to_config_json(&self) -> serde_json::Value {
        match self {
            ModelWrapper::Bpe(m) => m.to_config_json(),
            ModelWrapper::WordPiece(m) => m.to_config_json(),
            ModelWrapper::WordLevel(m) => m.to_config_json(),
            ModelWrapper::Unigram(m) => m.to_config_json(),
        }
    }

    /// 回收为 `ModelConfig`（写回配置 / 往返校验用）。
    pub fn to_model_config(&self) -> Result<ModelConfig> {
        serde_json::from_value(self.to_config_json())
            .map_err(|e| TokenizerError::config(format!("model 节无法回收为配置: {}", e)))
    }

    /// 按全局词表在模型层追加特殊 token（训练完成后由 pipeline 调用）。
    pub fn push_special_to_vocab(&mut self, token: tok_core::AddedToken, id: u32) -> Result<()> {
        self.get_vocab_mut().add_special(token, id)
    }
}

impl Model for ModelWrapper {
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        match self {
            ModelWrapper::Bpe(m) => m.tokenize(sequence),
            ModelWrapper::WordPiece(m) => m.tokenize(sequence),
            ModelWrapper::WordLevel(m) => m.tokenize(sequence),
            ModelWrapper::Unigram(m) => m.tokenize(sequence),
        }
    }

    fn get_vocab(&self) -> &Vocabulary {
        match self {
            ModelWrapper::Bpe(m) => m.get_vocab(),
            ModelWrapper::WordPiece(m) => m.get_vocab(),
            ModelWrapper::WordLevel(m) => m.get_vocab(),
            ModelWrapper::Unigram(m) => m.get_vocab(),
        }
    }

    fn get_vocab_mut(&mut self) -> &mut Vocabulary {
        match self {
            ModelWrapper::Bpe(m) => m.get_vocab_mut(),
            ModelWrapper::WordPiece(m) => m.get_vocab_mut(),
            ModelWrapper::WordLevel(m) => m.get_vocab_mut(),
            ModelWrapper::Unigram(m) => m.get_vocab_mut(),
        }
    }

    fn to_hf_json(&self) -> Result<serde_json::Value> {
        Ok(self.to_config_json())
    }

    fn save(&self, dir: &std::path::Path) -> Result<Vec<String>> {
        match self {
            ModelWrapper::Bpe(m) => m.save(dir),
            ModelWrapper::WordPiece(m) => m.save(dir),
            ModelWrapper::WordLevel(m) => m.save(dir),
            ModelWrapper::Unigram(m) => m.save(dir),
        }
    }

    fn get_trainer(&self, params: &TrainerParams) -> Option<Box<dyn Trainer>> {
        match self {
            ModelWrapper::Bpe(m) => m.get_trainer(params),
            ModelWrapper::WordPiece(m) => m.get_trainer(params),
            ModelWrapper::WordLevel(m) => m.get_trainer(params),
            ModelWrapper::Unigram(m) => m.get_trainer(params),
        }
    }

    fn apply_artifact(&mut self, artifact: &TrainerArtifact) -> Result<()> {
        match (self, artifact) {
            (ModelWrapper::Bpe(m), a @ TrainerArtifact::Bpe { .. }) => m.apply_artifact(a),
            (ModelWrapper::WordPiece(m), a @ TrainerArtifact::WordPiece { .. }) => m.apply_artifact(a),
            (ModelWrapper::WordLevel(m), a @ TrainerArtifact::WordLevel { .. }) => m.apply_artifact(a),
            (ModelWrapper::Unigram(m), a @ TrainerArtifact::Unigram { .. }) => m.apply_artifact(a),
            (w, a) => Err(TokenizerError::layer(
                layer::MODEL,
                format!("模型类型 {} 与产物类型 {} 不匹配", w.type_name(), artifact_kind(a)),
            )),
        }
    }
}

fn artifact_kind(a: &TrainerArtifact) -> &'static str {
    match a {
        TrainerArtifact::Bpe { .. } => "Bpe",
        TrainerArtifact::WordPiece { .. } => "WordPiece",
        TrainerArtifact::WordLevel { .. } => "WordLevel",
        TrainerArtifact::Unigram { .. } => "Unigram",
    }
}

/// 便捷构造：字节级 BPE 的 HF 常用配置。
pub fn byte_level_bpe_config(vocab: BTreeMap<String, u32>, merges: Vec<MergeSpec>) -> ModelConfig {
    ModelConfig::Bpe {
        dropout: None,
        unk_token: None,
        continuing_subword_prefix: None,
        end_of_word_suffix: None,
        fuse_unk: false,
        byte_fallback: false,
        ignore_merges: false,
        vocab,
        merges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use tok_core::AddedToken;

    fn tiny_bpe() -> ModelConfig {
        let mut vocab = BTreeMap::new();
        for (t, id) in [("a", 0u32), ("b", 1), ("ab", 2), ("<unk>", 3)] {
            vocab.insert(t.to_string(), id);
        }
        byte_level_bpe_config(
            vocab,
            vec![MergeSpec::Pair(vec!["a".into(), "b".into()]), MergeSpec::Text("b a".into())],
        )
    }

    #[test]
    fn build_and_tokenize_bpe() {
        let w = ModelWrapper::build(&tiny_bpe()).unwrap();
        assert_eq!(w.type_name(), "BPE");
        let toks = w.tokenize("ab").unwrap();
        assert_eq!(toks[0].value, "ab");
    }

    #[test]
    fn round_trip_hf_json() {
        let w = ModelWrapper::build(&tiny_bpe()).unwrap();
        let j = w.to_hf_json().unwrap();
        assert_eq!(j["type"], "BPE");
        assert_eq!(j["merges"][0][0], "a");
        let back = ModelWrapper::from_hf_json(&j).unwrap();
        assert_eq!(back.to_hf_json().unwrap(), j);
        // 两种 merges 写法都要能解析
        assert_eq!(back.as_bpe().unwrap().merges().len(), 2);
    }

    #[test]
    fn rejects_vocab_holes() {
        let mut vocab = BTreeMap::new();
        vocab.insert("a".to_string(), 0u32);
        vocab.insert("b".to_string(), 5u32);
        let cfg = byte_level_bpe_config(vocab, vec![]);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn dispatch_apply_artifact() {
        let mut w = ModelWrapper::build(&tiny_bpe()).unwrap();
        let art = TrainerArtifact::WordPiece { vocab: vec![("x".into(), 0)] };
        assert!(w.apply_artifact(&art).is_err());
        let ok = TrainerArtifact::Bpe {
            vocab: vec![("a".into(), 0), ("b".into(), 1)],
            merges: vec![("a".into(), "b".into())],
        };
        w.apply_artifact(&ok).unwrap();
        assert_eq!(w.get_vocab().len(), 2);
        w.push_special_to_vocab(AddedToken::new("<unk>", true), 2).unwrap();
        assert_eq!(w.get_vocab().token_to_id("<unk>"), Some(2));
    }
}
