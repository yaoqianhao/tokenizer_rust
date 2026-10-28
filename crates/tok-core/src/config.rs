//! 共享配置结构：版本、截断、填充、特殊 token、训练参数与训练产物。
//!
//! 各层的组件配置（NormalizerConfig / ModelConfig / ...）定义在各自的实现 crate 中，
//! 完整流水线配置 `TokenizerConfig` 位于 `tok-pipeline`（它能看到所有层）。

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// 配置版本号。序列化为字符串（如 `"1.0"`），与 HF `tokenizer.json` 的 `version` 一致。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ConfigVersion(pub String);

impl Default for ConfigVersion {
    fn default() -> Self {
        ConfigVersion("1.0".to_string())
    }
}

impl ConfigVersion {
    pub fn new(s: impl Into<String>) -> Self {
        ConfigVersion(s.into())
    }
    /// 解析为 (major, minor, patch)
    pub fn parts(&self) -> (u32, u32, u32) {
        let mut it = self.0.split('.');
        let a = it.next().and_then(|s| s.trim().parse().ok()).unwrap_or(1);
        let b = it.next().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        let c = it.next().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        (a, b, c)
    }
    /// 仅主版本相同即视为可加载（AD-3：未知字段直接拒绝，不做隐式兼容）
    pub fn is_compatible_with(&self, other: &ConfigVersion) -> bool {
        self.parts().0 == other.parts().0
    }
}

/// 截断准则。字段序列化为 PascalCase，与 HF 一致。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TruncationStrategy {
    #[default]
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TruncationDirection {
    Left,
    #[default]
    Right,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TruncationParams {
    pub max_length: usize,
    #[serde(default)]
    pub strategy: TruncationStrategy,
    #[serde(default)]
    pub stride: usize,
    #[serde(default)]
    pub direction: TruncationDirection,
}

impl Default for TruncationParams {
    fn default() -> Self {
        Self {
            max_length: 512,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: TruncationDirection::Right,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PaddingStrategy {
    #[default]
    BatchLongest,
    Fixed(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PaddingDirection {
    Left,
    #[default]
    Right,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PaddingParams {
    #[serde(default)]
    pub strategy: PaddingStrategy,
    #[serde(default)]
    pub direction: PaddingDirection,
    #[serde(default)]
    pub pad_to_multiple_of: Option<usize>,
    pub pad_id: u32,
    #[serde(default)]
    pub pad_type_id: u32,
    pub pad_token: String,
}

/// 正则/字面量模式：HF 的 `pattern` 是带标签枚举，落盘只接受
/// `{"Regex": "..."}` 与 `{"String": "..."}` 两种形态。
///
/// * `Tagged(RegexSpec)` —— `{"Regex": "..."}`，按正则编译；
/// * `Literal(LiteralSpec)` —— `{"String": "..."}`，按**字面量**匹配（内部转义后编译）；
/// * `Plain(String)` —— 裸字符串。HF 侧无此形态（`tokenizer.json` 加载会报
///   `invalid type: string`），仅作旧版本配置文件的兼容入口，语义沿用历史实现
///   （按正则处理），落盘时统一改写为 `{"Regex": ...}`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PatternSpec {
    Tagged(RegexSpec),
    Literal(LiteralSpec),
    Plain(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegexSpec {
    #[serde(rename = "Regex")]
    pub regex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiteralSpec {
    #[serde(rename = "String")]
    pub string: String,
}

impl PatternSpec {
    /// 正则模式（历史入口，等价于 `{"Regex": regex}`）。
    pub fn new(regex: impl Into<String>) -> Self {
        PatternSpec::Tagged(RegexSpec {
            regex: regex.into(),
        })
    }

    /// 字面量模式（`{"String": ...}`）：内容里的 `.` `\` `*` 等一律不解释。
    pub fn literal(text: impl Into<String>) -> Self {
        PatternSpec::Literal(LiteralSpec {
            string: text.into(),
        })
    }

    /// 匹配内容（不含标签）；字面量返回原文，正则返回表达式。
    pub fn as_str(&self) -> &str {
        match self {
            PatternSpec::Plain(s) => s.as_str(),
            PatternSpec::Tagged(r) => r.regex.as_str(),
            PatternSpec::Literal(l) => l.string.as_str(),
        }
    }

    /// 是否为字面量匹配（`String` 模式）。
    pub fn is_literal(&self) -> bool {
        matches!(self, PatternSpec::Literal(_))
    }

    /// 编译为 regex：字面量先 `escape`，保证 `.` 之类的字符不被当作元字符。
    /// 编译失败直接报错，不静默降级。
    pub fn compile(&self) -> Result<regex::Regex> {
        let pattern = match self {
            PatternSpec::Literal(l) => regex::escape(&l.string),
            other => other.as_str().to_string(),
        };
        regex::Regex::new(&pattern)
            .map_err(|e| crate::error::TokenizerError::Regex(format!("{}: {}", pattern, e)))
    }

    /// HF `tokenizer.json` 里的 `pattern` 形态。
    ///
    /// 裸字符串（`Plain`）在 HF 侧无法反序列化，落盘时按历史语义写成 `Regex`。
    pub fn to_hf_json(&self) -> serde_json::Value {
        match self {
            PatternSpec::Literal(l) => serde_json::json!({ "String": l.string }),
            PatternSpec::Tagged(r) => serde_json::json!({ "Regex": r.regex }),
            PatternSpec::Plain(s) => serde_json::json!({ "Regex": s }),
        }
    }
}

/// 后加 token（特殊 token 与训练后追加的 token）的登记信息。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddedToken {
    pub content: String,
    #[serde(default)]
    pub single_word: bool,
    #[serde(default)]
    pub lstrip: bool,
    #[serde(default)]
    pub rstrip: bool,
    #[serde(default)]
    pub normalized: bool,
    #[serde(default = "default_true")]
    pub special: bool,
}

fn default_true() -> bool {
    true
}

impl AddedToken {
    pub fn new(content: impl Into<String>, special: bool) -> Self {
        Self {
            content: content.into(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special,
        }
    }
    pub fn with_flags(mut self, single_word: bool, lstrip: bool, rstrip: bool) -> Self {
        self.single_word = single_word;
        self.lstrip = lstrip;
        self.rstrip = rstrip;
        self
    }
}

/// 带 ID 的 AddedToken，对应 HF `tokenizer.json` 的 `added_tokens` 数组元素。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddedTokenWithId {
    pub id: u32,
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

impl AddedTokenWithId {
    pub fn from_token(t: &AddedToken, id: u32) -> Self {
        Self {
            id,
            content: t.content.clone(),
            single_word: t.single_word,
            lstrip: t.lstrip,
            rstrip: t.rstrip,
            normalized: t.normalized,
            special: t.special,
        }
    }

    /// 丢弃 id，还原为门面层使用的 `AddedToken`（id 由词表登记顺序决定）。
    pub fn to_token(&self) -> AddedToken {
        AddedToken {
            content: self.content.clone(),
            single_word: self.single_word,
            lstrip: self.lstrip,
            rstrip: self.rstrip,
            normalized: self.normalized,
            special: self.special,
        }
    }
}

/// 训练参数（由门面在调用 `Model::get_trainer` 时注入）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerParams {
    pub vocab_size: usize,
    #[serde(default = "default_min_freq")]
    pub min_frequency: u64,
    #[serde(default)]
    pub special_tokens: Vec<String>,
    /// 初始字母表（字节级 BPE 传 256 个字节字符；普通 BPE 传空）
    #[serde(default)]
    pub initial_alphabet: Vec<char>,
    /// WordPiece 的最大词长保护
    #[serde(default)]
    pub max_token_length: Option<usize>,
    #[serde(default)]
    pub continuing_subword_prefix: Option<String>,
    #[serde(default)]
    pub end_of_word_suffix: Option<String>,
    #[serde(default)]
    pub limit_alphabet: Option<usize>,
    /// 是否生成 256 个字节级 token `<0x00>..<0xFF>`（字节级 BPE / Unigram byte-fallback）
    #[serde(default)]
    pub byte_fallback: bool,
    /// Unigram：候选 piece 的最大字符数（SentencePiece `max_piece_length`，默认 16）
    #[serde(default)]
    pub max_piece_length: Option<usize>,
    /// Unigram：每轮剪枝保留比例（SentencePiece `shrinking_factor`，默认 0.75）
    #[serde(default)]
    pub shrinking_factor: Option<f64>,
    /// Unigram：每个剪枝阶段内部的 EM 迭代次数（默认 2）
    #[serde(default)]
    pub n_sub_iterations: Option<usize>,
}

impl TrainerParams {
    /// 常用构造：给定词表规模
    pub fn of(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            ..Default::default()
        }
    }
}

fn default_min_freq() -> u64 {
    2
}

impl Default for TrainerParams {
    fn default() -> Self {
        Self {
            vocab_size: 30000,
            min_frequency: 2,
            special_tokens: Vec::new(),
            initial_alphabet: Vec::new(),
            max_token_length: None,
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            limit_alphabet: None,
            byte_fallback: false,
            max_piece_length: None,
            shrinking_factor: None,
            n_sub_iterations: None,
        }
    }
}

/// 训练产物：由 Trainer 产出、由 Model 消费（`Model::apply_artifact`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum TrainerArtifact {
    WordLevel {
        vocab: Vec<(String, u32)>,
    },
    Bpe {
        vocab: Vec<(String, u32)>,
        merges: Vec<(String, String)>,
    },
    WordPiece {
        vocab: Vec<(String, u32)>,
    },
    Unigram {
        vocab: Vec<(String, f64)>,
    },
}
