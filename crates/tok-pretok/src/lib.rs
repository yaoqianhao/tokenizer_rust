//! # tok-pretok
//!
//! L2 预分词层实现。契约：切分后 `splits` 的拼接必须等于 normalized 全文，
//! 且每个切分片都能映射回原始偏移（`convert_offsets`）。

use fancy_regex::Regex as FancyRegex;
use serde::{Deserialize, Serialize};
use tok_core::{
    layer, NormalizedString, PatternSpec, PreTokenizer, Result, TokenizerError,
    GPT2_BYTE_LEVEL_PATTERN,
};
// 重新导出核心层的分隔符行为，L2 使用者无需直接依赖 `tok-core`。
pub use tok_core::SplitDelimiterBehavior;

/// 分隔符行为：兼容 HF 的 `"behavior": "removed"` 与 `{"SplitDelimiterBehavior":"removed"}`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BehaviorSpec {
    Plain(String),
    Tagged { #[serde(rename = "SplitDelimiterBehavior")] behavior: String },
}

impl BehaviorSpec {
    pub fn named(s: &str) -> Self {
        BehaviorSpec::Plain(s.to_string())
    }
    pub fn parse(&self) -> Result<SplitDelimiterBehavior> {
        let s = match self {
            BehaviorSpec::Plain(s) => s.as_str(),
            BehaviorSpec::Tagged { behavior } => behavior.as_str(),
        };
        match s {
            "removed" | "Removed" => Ok(SplitDelimiterBehavior::Removed),
            "isolated" | "Isolated" => Ok(SplitDelimiterBehavior::Isolated),
            "merged_with_previous" | "MergedWithPrevious" => Ok(SplitDelimiterBehavior::MergedWithPrevious),
            "merged_with_next" | "MergedWithNext" => Ok(SplitDelimiterBehavior::MergedWithNext),
            other => Err(TokenizerError::config(format!("未知 SplitDelimiterBehavior: {}", other))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PreTokenizerConfig {
    #[serde(rename = "Whitespace")]
    Whitespace,
    #[serde(rename = "WhitespaceSplit")]
    WhitespaceSplit,
    #[serde(rename = "BertPreTokenizer")]
    BertPreTokenizer,
    #[serde(rename = "ByteLevel")]
    ByteLevel {
        #[serde(default)]
        add_prefix_space: bool,
        #[serde(default)]
        trim_offsets: bool,
        #[serde(default = "default_true")]
        use_regex: bool,
    },
    #[serde(rename = "Metaspace")]
    Metaspace {
        #[serde(default = "default_replacement")]
        replacement: String,
        #[serde(default)]
        add_prefix_space: bool,
        #[serde(default)]
        prepend_scheme: Option<String>,
    },
    #[serde(rename = "Digits")]
    Digits {
        #[serde(default)]
        individual_digits: bool,
    },
    #[serde(rename = "Punctuation")]
    Punctuation {
        #[serde(default = "default_isolated")]
        behavior: BehaviorSpec,
    },
    #[serde(rename = "Split")]
    Split {
        pattern: PatternSpec,
        #[serde(default = "default_isolated")]
        behavior: BehaviorSpec,
        #[serde(default)]
        invert: bool,
    },
    #[serde(rename = "Sequence")]
    Sequence {
        pretokenizers: Vec<PreTokenizerConfig>,
    },
}

fn default_true() -> bool {
    true
}
fn default_replacement() -> String {
    "\u{2581}".to_string()
}
fn default_isolated() -> BehaviorSpec {
    BehaviorSpec::named("isolated")
}

/// 运行时组件。
pub enum PreTokenizerWrapper {
    Whitespace,
    WhitespaceSplit,
    BertPreTokenizer,
    ByteLevel {
        regex: FancyRegex,
        add_prefix_space: bool,
        trim_offsets: bool,
        use_regex: bool,
    },
    Metaspace {
        replacement: String,
        add_prefix_space: bool,
        prepend_scheme: String,
    },
    Digits {
        individual_digits: bool,
    },
    Punctuation {
        behavior: SplitDelimiterBehavior,
    },
    Split {
        regex: FancyRegex,
        behavior: SplitDelimiterBehavior,
        invert: bool,
    },
    Sequence(Vec<PreTokenizerWrapper>),
    Custom(Box<dyn PreTokenizer>),
}

impl std::fmt::Debug for PreTokenizerWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PreTokenizerWrapper({})", self.name())
    }
}

fn fancy(pattern: &str) -> Result<FancyRegex> {
    FancyRegex::new(pattern).map_err(|e| TokenizerError::Regex(format!("{}: {}", pattern, e)))
}

fn find_all_str(hay: &str, needle: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if needle.is_empty() {
        return out;
    }
    let mut from = 0usize;
    while let Some(pos) = hay[from..].find(needle) {
        let s = from + pos;
        let e = s + needle.len();
        out.push((s, e));
        from = e;
    }
    out
}

impl PreTokenizerWrapper {
    pub fn build(cfg: &PreTokenizerConfig) -> Result<Self> {
        Ok(match cfg {
            PreTokenizerConfig::Whitespace => PreTokenizerWrapper::Whitespace,
            PreTokenizerConfig::WhitespaceSplit => PreTokenizerWrapper::WhitespaceSplit,
            PreTokenizerConfig::BertPreTokenizer => PreTokenizerWrapper::BertPreTokenizer,
            PreTokenizerConfig::ByteLevel { add_prefix_space, trim_offsets, use_regex } => {
                PreTokenizerWrapper::ByteLevel {
                    regex: fancy(GPT2_BYTE_LEVEL_PATTERN)?,
                    add_prefix_space: *add_prefix_space,
                    trim_offsets: *trim_offsets,
                    use_regex: *use_regex,
                }
            }
            PreTokenizerConfig::Metaspace { replacement, add_prefix_space, prepend_scheme } => {
                PreTokenizerWrapper::Metaspace {
                    replacement: replacement.clone(),
                    add_prefix_space: *add_prefix_space,
                    prepend_scheme: prepend_scheme.clone().unwrap_or_else(|| {
                        if *add_prefix_space { "always".to_string() } else { "never".to_string() }
                    }),
                }
            }
            PreTokenizerConfig::Digits { individual_digits } => {
                PreTokenizerWrapper::Digits { individual_digits: *individual_digits }
            }
            PreTokenizerConfig::Punctuation { behavior } => {
                PreTokenizerWrapper::Punctuation { behavior: behavior.parse()? }
            }
            PreTokenizerConfig::Split { pattern, behavior, invert } => PreTokenizerWrapper::Split {
                regex: fancy(pattern.as_str())?,
                behavior: behavior.parse()?,
                invert: *invert,
            },
            PreTokenizerConfig::Sequence { pretokenizers } => {
                let mut v = Vec::with_capacity(pretokenizers.len());
                for p in pretokenizers {
                    v.push(PreTokenizerWrapper::build(p)?);
                }
                PreTokenizerWrapper::Sequence(v)
            }
        })
    }

    /// ByteLevel 的 `add_prefix_space` 需要在**原始文本**层面处理，由门面在 S1 之前调用。
    pub fn needs_prefix_space(&self) -> bool {
        match self {
            PreTokenizerWrapper::ByteLevel { add_prefix_space, .. } => *add_prefix_space,
            PreTokenizerWrapper::Sequence(v) => v.iter().any(|p| p.needs_prefix_space()),
            _ => false,
        }
    }

    fn run_one(&self, ns: &mut NormalizedString) -> Result<()> {
        match self {
            PreTokenizerWrapper::Whitespace => {
                ns.split_on_pred(|c| c.is_whitespace(), SplitDelimiterBehavior::Removed);
                Ok(())
            }
            PreTokenizerWrapper::WhitespaceSplit => {
                // HF `WhitespaceSplit`：按空白切分且**丢弃**空白段（不产出 piece），
                // 与 `Whitespace` 的差别仅在连续空白/首尾空白的处理细节上。
                ns.split_on_pred(|c| c.is_whitespace(), SplitDelimiterBehavior::Removed);
                Ok(())
            }
            PreTokenizerWrapper::BertPreTokenizer => {
                // HF `BertPreTokenizer` 语义 = Whitespace(removed) + Punctuation(isolated)：
                // 空白段**丢弃**（不产出 piece），标点各自独立成 piece。
                ns.split_on_pred(|c| c.is_whitespace(), SplitDelimiterBehavior::Removed);
                ns.split_on_pred(is_punctuation, SplitDelimiterBehavior::Isolated);
                Ok(())
            }
            PreTokenizerWrapper::ByteLevel { regex, use_regex, add_prefix_space, trim_offsets: _ } => {
                if *add_prefix_space && !ns.normalized().starts_with(' ') {
                    ns.prepend(" ");
                }
                if *use_regex {
                    // HF 顺序：正则先作用于**原始文本**，再对 piece 做字节级映射。
                    // 若先映射（Ġ U+0120 属 \p{L}），"Hello world" 会被 \p{L}+ 误并成一个 piece。
                    let mut hits = Vec::new();
                    for m in regex.find_iter(ns.normalized()) {
                        match m {
                            Ok(m) => hits.push((m.start(), m.end())),
                            Err(e) => {
                                return Err(TokenizerError::Regex(format!("ByteLevel 匹配失败: {}", e)))
                            }
                        }
                    }
                    ns.byte_level_map_pieces(hits);
                } else {
                    ns.byte_level_expand();
                }
                Ok(())
            }
            PreTokenizerWrapper::Metaspace { replacement, add_prefix_space, prepend_scheme } => {
                let repl = replacement.clone();
                ns.map_chars(move |c| {
                    if c.is_whitespace() && c.to_string() != repl {
                        Some(repl.clone())
                    } else {
                        None
                    }
                });
                let need_prefix = match prepend_scheme.as_str() {
                    "always" => true,
                    "never" => false,
                    _ => {
                        // "first"：仅首个切片需要前缀
                        *add_prefix_space && !ns.normalized().starts_with(&replacement[..])
                    }
                };
                if need_prefix && !ns.normalized().starts_with(replacement.as_str()) {
                    ns.prepend(replacement);
                }
                if !replacement.is_empty() {
                    let delims = find_all_str(ns.normalized(), replacement);
                    if !delims.is_empty() {
                        ns.split_by_delims(delims, SplitDelimiterBehavior::MergedWithNext);
                    }
                }
                Ok(())
            }
            PreTokenizerWrapper::Digits { individual_digits } => {
                if *individual_digits {
                    ns.split_on_pred(|c| c.is_numeric(), SplitDelimiterBehavior::Isolated);
                    let digits = ns.normalized();
                    let mut single: Vec<(usize, usize)> = Vec::new();
                    for (i, c) in digits.char_indices() {
                        if c.is_numeric() {
                            single.push((i, i + c.len_utf8()));
                        }
                    }
                    ns.split_by_delims(single, SplitDelimiterBehavior::Isolated);
                } else {
                    ns.split_on_pred(|c| c.is_numeric(), SplitDelimiterBehavior::Isolated);
                }
                Ok(())
            }
            PreTokenizerWrapper::Punctuation { behavior } => {
                ns.split_on_pred(|c| is_punctuation(c), *behavior);
                Ok(())
            }
            PreTokenizerWrapper::Split { regex, behavior, invert } => {
                let mut delims = Vec::new();
                for m in regex.find_iter(ns.normalized()) {
                    match m {
                        Ok(m) => delims.push((m.start(), m.end())),
                        Err(e) => return Err(TokenizerError::Regex(format!("Split 匹配失败: {}", e))),
                    }
                }
                if *invert {
                    // invert：保留命中片段本身（等价于用命中做正向切分）
                    ns.split_by_matches(delims);
                } else {
                    ns.split_by_delims(delims, *behavior);
                }
                Ok(())
            }
            PreTokenizerWrapper::Sequence(list) => {
                for p in list {
                    p.run_one(ns)?;
                }
                Ok(())
            }
            PreTokenizerWrapper::Custom(c) => c.pre_tokenize(ns),
        }
    }
}

fn is_punctuation(c: char) -> bool {
    c.is_ascii_punctuation() || (!c.is_alphanumeric() && !c.is_whitespace() && !c.is_control())
}

impl PreTokenizer for PreTokenizerWrapper {
    fn pre_tokenize(&self, ns: &mut NormalizedString) -> Result<()> {
        self.run_one(ns)
    }

    fn is_byte_level(&self) -> bool {
        match self {
            PreTokenizerWrapper::ByteLevel { .. } => true,
            PreTokenizerWrapper::Sequence(v) => v.iter().any(|p| p.is_byte_level()),
            _ => false,
        }
    }

    fn to_config_json(&self) -> Result<serde_json::Value> {
        match self {
            PreTokenizerWrapper::Whitespace => Ok(serde_json::json!({"type": "Whitespace"})),
            PreTokenizerWrapper::WhitespaceSplit => Ok(serde_json::json!({"type": "WhitespaceSplit"})),
            PreTokenizerWrapper::BertPreTokenizer => Ok(serde_json::json!({"type": "BertPreTokenizer"})),
            PreTokenizerWrapper::ByteLevel { add_prefix_space, trim_offsets, use_regex, .. } => {
                Ok(serde_json::json!({
                    "type": "ByteLevel",
                    "add_prefix_space": add_prefix_space,
                    "trim_offsets": trim_offsets,
                    "use_regex": use_regex,
                }))
            }
            PreTokenizerWrapper::Metaspace { replacement, prepend_scheme, .. } => Ok(serde_json::json!({
                "type": "Metaspace",
                "replacement": replacement,
                "prepend_scheme": prepend_scheme,
                "add_prefix_space": *prepend_scheme == "always",
            })),
            PreTokenizerWrapper::Digits { individual_digits } => Ok(serde_json::json!({
                "type": "Digits",
                "individual_digits": individual_digits,
            })),
            PreTokenizerWrapper::Punctuation { behavior } => Ok(serde_json::json!({
                "type": "Punctuation",
                "behavior": behavior_name(*behavior),
            })),
            PreTokenizerWrapper::Split { regex, behavior, invert } => Ok(serde_json::json!({
                "type": "Split",
                "pattern": {"Regex": regex.as_str()},
                "behavior": behavior_name(*behavior),
                "invert": invert,
            })),
            PreTokenizerWrapper::Sequence(list) => {
                let mut items = Vec::with_capacity(list.len());
                for p in list {
                    items.push(p.to_config_json()?);
                }
                Ok(serde_json::json!({"type": "Sequence", "pretokenizers": items}))
            }
            PreTokenizerWrapper::Custom(_) => Err(TokenizerError::layer(
                layer::PRE_TOKENIZER,
                "Custom pre_tokenizer 无法导出为 HF 配置",
            )),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            PreTokenizerWrapper::Whitespace => "Whitespace",
            PreTokenizerWrapper::WhitespaceSplit => "WhitespaceSplit",
            PreTokenizerWrapper::BertPreTokenizer => "BertPreTokenizer",
            PreTokenizerWrapper::ByteLevel { .. } => "ByteLevel",
            PreTokenizerWrapper::Metaspace { .. } => "Metaspace",
            PreTokenizerWrapper::Digits { .. } => "Digits",
            PreTokenizerWrapper::Punctuation { .. } => "Punctuation",
            PreTokenizerWrapper::Split { .. } => "Split",
            PreTokenizerWrapper::Sequence(_) => "Sequence",
            PreTokenizerWrapper::Custom(_) => "Custom",
        }
    }
}

fn behavior_name(b: SplitDelimiterBehavior) -> &'static str {
    match b {
        SplitDelimiterBehavior::Removed => "removed",
        SplitDelimiterBehavior::Isolated => "isolated",
        SplitDelimiterBehavior::MergedWithPrevious => "merged_with_previous",
        SplitDelimiterBehavior::MergedWithNext => "merged_with_next",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(cfg: &PreTokenizerConfig, text: &str) -> Vec<String> {
        let p = PreTokenizerWrapper::build(cfg).unwrap();
        let mut ns = NormalizedString::from_str(text);
        p.pre_tokenize(&mut ns).unwrap();
        ns.pretokens().into_iter().map(|t| t.value).collect()
    }

    #[test]
    fn whitespace_discards_separators() {
        let v = pt(&PreTokenizerConfig::Whitespace, "hello  world");
        assert_eq!(v, vec!["hello", "world"]);
    }

    #[test]
    fn bert_splits_punctuation_and_space() {
        let v = pt(&PreTokenizerConfig::BertPreTokenizer, "hi, there!");
        assert_eq!(v, vec!["hi", ",", "there", "!"]);
    }

    #[test]
    fn byte_level_expands_utf8_bytes() {
        let cfg = PreTokenizerConfig::ByteLevel { add_prefix_space: false, trim_offsets: false, use_regex: true };
        let v = pt(&cfg, "hi");
        assert_eq!(v, vec!["hi"]);
        let v2 = pt(&cfg, "中");
        // HF 顺序：正则先作用于**原始文本** —— "中" 命中 `\p{L}+` → 单个 piece；
        // 再对该 piece 做字节级映射 → "ä¸Ń"（3 个字符，3 = UTF-8 字节数）。
        // 若反过来先映射再切分，"Ġ" 等映射字符属 \p{L}，piece 边界会与 HF 不一致。
        assert_eq!(v2, vec!["ä¸Ń"]);
        assert_eq!(v2[0].chars().count(), 3, "字节级展开后字符数应等于 UTF-8 字节数");
        // ` ?\p{L}+` 使前导空格并入后继 piece（映射为 Ġ）
        let v3 = pt(&cfg, "hi there");
        assert_eq!(v3, vec!["hi", "Ġthere"]);
    }

    #[test]
    fn byte_level_offsets_point_to_original_bytes() {
        let cfg = PreTokenizerConfig::ByteLevel { add_prefix_space: false, trim_offsets: false, use_regex: true };
        let p = PreTokenizerWrapper::build(&cfg).unwrap();
        let mut ns = NormalizedString::from_str("ab");
        p.pre_tokenize(&mut ns).unwrap();
        let ts = ns.pretokens();
        assert_eq!(ts.len(), 1);
        assert_eq!(ts[0].offsets, (0, 2));
    }

    #[test]
    fn digits_individual() {
        let v = pt(&PreTokenizerConfig::Digits { individual_digits: true }, "a12b");
        assert_eq!(v, vec!["a", "1", "2", "b"]);
    }

    #[test]
    fn metaspace_marks_word_starts() {
        let cfg = PreTokenizerConfig::Metaspace {
            replacement: "\u{2581}".into(),
            add_prefix_space: true,
            prepend_scheme: Some("always".into()),
        };
        let v = pt(&cfg, "hello world");
        assert_eq!(v, vec!["\u{2581}hello", "\u{2581}world"]);
    }

    #[test]
    fn split_with_removed_behavior() {
        let cfg = PreTokenizerConfig::Split {
            pattern: PatternSpec::new("\\s+"),
            behavior: BehaviorSpec::named("removed"),
            invert: false,
        };
        let v = pt(&cfg, "a  b");
        assert_eq!(v, vec!["a", "b"]);
    }

    #[test]
    fn bert_pretokenizer_isolates_each_punctuation_char() {
        // HF 语义：Whitespace(removed) + Punctuation(isolated)——
        // 连续标点必须**逐字符**断开（"?!" → "?" , "!"），空白不产出 piece。
        let v = pt(&PreTokenizerConfig::BertPreTokenizer, "punct?! (x) [y]");
        assert_eq!(v, vec!["punct", "?", "!", "(", "x", ")", "[", "y", "]"]);
    }

    #[test]
    fn whitespace_split_drops_whitespace_pieces() {
        // HF `WhitespaceSplit` 丢弃空白，不产出空白 piece
        let v = pt(&PreTokenizerConfig::WhitespaceSplit, "a  b\tc");
        assert_eq!(v, vec!["a", "b", "c"]);
    }

    #[test]
    fn sequence_is_applied_in_order() {
        let cfg = PreTokenizerConfig::Sequence {
            pretokenizers: vec![
                PreTokenizerConfig::BertPreTokenizer,
                PreTokenizerConfig::Digits { individual_digits: true },
            ],
        };
        let v = pt(&cfg, "ab12");
        assert_eq!(v, vec!["ab", "1", "2"]);
    }
}
