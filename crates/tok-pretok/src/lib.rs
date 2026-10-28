//! # tok-pretok
//!
//! L2 预分词层实现。契约：切分后 `splits` 的拼接必须等于 normalized 全文，
//! 且每个切分片都能映射回原始偏移（`convert_offsets`）。

use std::collections::BTreeMap;

use fancy_regex::Regex as FancyRegex;
use serde::{Deserialize, Serialize};
use tok_core::{
    layer, NormalizedString, PatternSpec, PreTokenizer, Result, TokenizerError,
    GPT2_BYTE_LEVEL_PATTERN,
};
use unicode_general_category::{get_general_category, GeneralCategory};
// 重新导出核心层的分隔符行为，L2 使用者无需直接依赖 `tok-core`。
pub use tok_core::SplitDelimiterBehavior;

/// 分隔符行为：兼容 HF 的 `"behavior": "Isolated"`、小写写法与
/// `{"SplitDelimiterBehavior":"isolated"}` 形式。
///
/// 落盘时统一序列化成 HF 的 PascalCase 枚举名——HF 的 `tokenizers` 用
/// `#[serde(rename_all = "PascalCase")]` 反序列化该枚举，写 `"isolated"` 会导致
/// `tokenizer.json` 直接加载失败（E3 对照时踩到的坑）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BehaviorSpec {
    Plain(String),
    Tagged { behavior: String },
}

impl BehaviorSpec {
    pub fn named(s: &str) -> Self {
        BehaviorSpec::Plain(s.to_string())
    }

    /// 原始写法（可能为小写/下划线形式）。
    pub fn raw(&self) -> &str {
        match self {
            BehaviorSpec::Plain(s) => s.as_str(),
            BehaviorSpec::Tagged { behavior } => behavior.as_str(),
        }
    }

    /// HF 兼容的规范名；未知写法返回 `None`，由 `parse` 决定如何报错。
    pub fn canonical(&self) -> Option<&'static str> {
        canonical_behavior(self.raw())
    }

    pub fn parse(&self) -> Result<SplitDelimiterBehavior> {
        match canonical_behavior(self.raw()) {
            Some("Removed") => Ok(SplitDelimiterBehavior::Removed),
            Some("Isolated") => Ok(SplitDelimiterBehavior::Isolated),
            Some("MergedWithPrevious") => Ok(SplitDelimiterBehavior::MergedWithPrevious),
            Some("MergedWithNext") => Ok(SplitDelimiterBehavior::MergedWithNext),
            Some("Contiguous") => Err(TokenizerError::config(
                "behavior=contiguous 尚未实现（HF 的 Contiguous 变体暂不支持）".to_string(),
            )),
            _ => Err(TokenizerError::config(format!(
                "未知 SplitDelimiterBehavior: {}（可选 removed / isolated / merged_with_previous / merged_with_next）",
                self.raw()
            ))),
        }
    }
}

/// 名称归一化：大小写、连字符、下划线都不敏感。
fn canonical_behavior(raw: &str) -> Option<&'static str> {
    let key: String = raw
        .chars()
        .filter(|c| *c != '_' && *c != '-')
        .flat_map(|c| c.to_lowercase())
        .collect();
    match key.as_str() {
        "removed" => Some("Removed"),
        "isolated" => Some("Isolated"),
        "mergedwithprevious" => Some("MergedWithPrevious"),
        "mergedwithnext" => Some("MergedWithNext"),
        "contiguous" => Some("Contiguous"),
        _ => None,
    }
}

impl Serialize for BehaviorSpec {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.canonical().unwrap_or_else(|| self.raw()))
    }
}

impl<'de> Deserialize<'de> for BehaviorSpec {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Plain(String),
            Keyed(BTreeMap<String, String>),
        }
        match Repr::deserialize(deserializer)? {
            Repr::Plain(s) => Ok(BehaviorSpec::Plain(s)),
            Repr::Keyed(map) => {
                let behavior = map
                    .get("SplitDelimiterBehavior")
                    .or_else(|| map.get("value"))
                    .or_else(|| map.get("behavior"))
                    .cloned()
                    .ok_or_else(|| {
                        serde::de::Error::custom(
                            "behavior 对象缺少 SplitDelimiterBehavior/value 字段",
                        )
                    })?;
                Ok(BehaviorSpec::Tagged { behavior })
            }
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
        /// 原始 pattern：落盘时要还原 `Regex` / `String` 标签
        spec: PatternSpec,
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

/// Split 的正则源文本：
///
/// * `String`（字面量）模式先转义再编译——`.` `\` `*` 等不被当作元字符；
/// * `Regex` 模式原样交给 fancy_regex，保留后向引用/环视等扩展语法。
fn split_pattern_source(pattern: &PatternSpec) -> Result<String> {
    match pattern {
        PatternSpec::Literal(_) => Ok(pattern.compile()?.as_str().to_string()),
        other => Ok(other.as_str().to_string()),
    }
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
            PreTokenizerConfig::ByteLevel {
                add_prefix_space,
                trim_offsets,
                use_regex,
            } => PreTokenizerWrapper::ByteLevel {
                regex: fancy(GPT2_BYTE_LEVEL_PATTERN)?,
                add_prefix_space: *add_prefix_space,
                trim_offsets: *trim_offsets,
                use_regex: *use_regex,
            },
            PreTokenizerConfig::Metaspace {
                replacement,
                add_prefix_space,
                prepend_scheme,
            } => PreTokenizerWrapper::Metaspace {
                replacement: replacement.clone(),
                add_prefix_space: *add_prefix_space,
                prepend_scheme: prepend_scheme.clone().unwrap_or_else(|| {
                    if *add_prefix_space {
                        "always".to_string()
                    } else {
                        "never".to_string()
                    }
                }),
            },
            PreTokenizerConfig::Digits { individual_digits } => PreTokenizerWrapper::Digits {
                individual_digits: *individual_digits,
            },
            PreTokenizerConfig::Punctuation { behavior } => PreTokenizerWrapper::Punctuation {
                behavior: behavior.parse()?,
            },
            PreTokenizerConfig::Split {
                pattern,
                behavior,
                invert,
            } => PreTokenizerWrapper::Split {
                spec: pattern.clone(),
                regex: fancy(&split_pattern_source(pattern)?)?,
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
            PreTokenizerWrapper::ByteLevel {
                add_prefix_space, ..
            } => *add_prefix_space,
            PreTokenizerWrapper::Sequence(v) => v.iter().any(|p| p.needs_prefix_space()),
            _ => false,
        }
    }

    fn run_one(&self, ns: &mut NormalizedString) -> Result<()> {
        match self {
            PreTokenizerWrapper::Whitespace => {
                // HF `Whitespace`：等价正则 `\w+|[^\w\s]+`——连续 word 字符成一段、
                // 连续「非空白且非 word」的符号成一段，空白全部**丢弃**（不产出 piece）。
                // 关键差别：这里会**切开标点**，"world!" → ["world", "!"]。
                let spans = word_or_symbol_spans(ns.normalized());
                ns.split_by_matches(spans);
                Ok(())
            }
            PreTokenizerWrapper::WhitespaceSplit => {
                // HF `WhitespaceSplit`：仅按空白切分且**丢弃**空白段（不产出 piece），
                // **不切标点**——"world!" 仍是单个 piece。这是与 `Whitespace` 的语义差别，
                // 两者不能互相替代：训练侧产物按哪种配置生成，推理侧必须用同一种。
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
            PreTokenizerWrapper::ByteLevel {
                regex,
                use_regex,
                add_prefix_space,
                trim_offsets: _,
            } => {
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
                                return Err(TokenizerError::Regex(format!(
                                    "ByteLevel 匹配失败: {}",
                                    e
                                )))
                            }
                        }
                    }
                    ns.byte_level_map_pieces(hits);
                } else {
                    ns.byte_level_expand();
                }
                Ok(())
            }
            PreTokenizerWrapper::Metaspace {
                replacement,
                add_prefix_space,
                prepend_scheme,
            } => {
                let repl = replacement.clone();
                // HF 语义：仅把 ASCII 空格 0x20 换成 replacement；tab / NBSP / 全角空格等
                // 其它空白字符原样保留，交由模型（byte_fallback 时落 <0x09>）处理。
                ns.map_chars(move |c| {
                    if c == ' ' && c.to_string() != repl {
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
                    // 插入的 ▁ 与首个字符共享 original 区间，保证首个 token offset 与 HF 一致
                    ns.prepend_absorbing_next(replacement);
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
                ns.split_on_pred(is_punctuation, *behavior);
                Ok(())
            }
            PreTokenizerWrapper::Split {
                regex,
                behavior,
                invert,
                ..
            } => {
                let mut delims = Vec::new();
                for m in regex.find_iter(ns.normalized()) {
                    match m {
                        Ok(m) => delims.push((m.start(), m.end())),
                        Err(e) => {
                            return Err(TokenizerError::Regex(format!("Split 匹配失败: {}", e)))
                        }
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

/// 标点判定：与 HF `unicode_categories` 的 `P*` 类别严格对齐。
///
/// 只认 Unicode 标点类别（Pc / Pd / Pe / Pf / Pi / Po / Ps），
/// emoji（So）、数学符号（Sm）、货币符号（Sc）等**不算**标点——
/// 否则 `"😀🎉🧠"` 会被逐字符拆成独立 piece，WordPiece 也就逐字符吐 `[UNK]`，
/// 而 HF 会把它当成一个 word、整词退化为单个 `[UNK]`。
fn is_punctuation(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_punctuation();
    }
    matches!(
        get_general_category(c),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::OtherPunctuation
    )
}

/// `\w` 判定：与 Rust `regex` 的 Unicode `\w` 保持一致。
///
/// `\w` = Alphabetic（L* + Nl + Other_Alphabetic）+ M* + Nd + Pc + Join_Control。
/// 注意 `char::is_numeric()` 覆盖的是全部 N*（含 Nl / No），比 `\w` 更宽，
/// 因此数字位必须单独用 `GeneralCategory::DecimalNumber` 判定。
fn is_word_char(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric() || c == '_';
    }
    c.is_alphabetic()
        || matches!(
            get_general_category(c),
            GeneralCategory::DecimalNumber
                | GeneralCategory::NonspacingMark
                | GeneralCategory::SpacingMark
                | GeneralCategory::EnclosingMark
                | GeneralCategory::ConnectorPunctuation
        )
        || c == '\u{200C}'
        || c == '\u{200D}'
}

/// HF `Whitespace` 预分词的等价切分：正则 `\w+|[^\w\s]+` 的命中区间。
///
/// 连续 word 字符归为一段，连续「非空白且非 word」的符号归为一段，空白字符不入段。
/// 返回 normalized 文本中的**字节**区间（`NormalizedString::split_by_matches` 约定）。
fn word_or_symbol_spans(text: &str) -> Vec<(usize, usize)> {
    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Word,
        Symbol,
    }

    let mut out: Vec<(usize, usize)> = Vec::with_capacity(text.len() / 4 + 1);
    let mut cur: Option<(Kind, usize, usize)> = None;
    for (i, c) in text.char_indices() {
        let end = i + c.len_utf8();
        let kind = if c.is_whitespace() {
            None
        } else if is_word_char(c) {
            Some(Kind::Word)
        } else {
            Some(Kind::Symbol)
        };
        match (kind, cur) {
            (Some(k), Some((ck, s, _))) if ck == k => cur = Some((k, s, end)),
            (Some(k), Some((_, s, e))) => {
                out.push((s, e));
                cur = Some((k, i, end));
            }
            (Some(k), None) => cur = Some((k, i, end)),
            (None, Some((_, s, e))) => {
                out.push((s, e));
                cur = None;
            }
            (None, None) => {}
        }
    }
    if let Some((_, s, e)) = cur {
        out.push((s, e));
    }
    out
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
            PreTokenizerWrapper::WhitespaceSplit => {
                Ok(serde_json::json!({"type": "WhitespaceSplit"}))
            }
            PreTokenizerWrapper::BertPreTokenizer => {
                Ok(serde_json::json!({"type": "BertPreTokenizer"}))
            }
            PreTokenizerWrapper::ByteLevel {
                add_prefix_space,
                trim_offsets,
                use_regex,
                ..
            } => Ok(serde_json::json!({
                "type": "ByteLevel",
                "add_prefix_space": add_prefix_space,
                "trim_offsets": trim_offsets,
                "use_regex": use_regex,
            })),
            PreTokenizerWrapper::Metaspace {
                replacement,
                prepend_scheme,
                ..
            } => Ok(serde_json::json!({
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
            PreTokenizerWrapper::Split {
                spec,
                behavior,
                invert,
                ..
            } => Ok(serde_json::json!({
                "type": "Split",
                "pattern": spec.to_hf_json(),
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

/// 导出名必须用 HF 的 PascalCase 枚举名，否则 `tokenizer.json` 无法被
/// `tokenizers` 反序列化（`unknown variant`）。
fn behavior_name(b: SplitDelimiterBehavior) -> &'static str {
    match b {
        SplitDelimiterBehavior::Removed => "Removed",
        SplitDelimiterBehavior::Isolated => "Isolated",
        SplitDelimiterBehavior::MergedWithPrevious => "MergedWithPrevious",
        SplitDelimiterBehavior::MergedWithNext => "MergedWithNext",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(cfg: &PreTokenizerConfig, text: &str) -> Vec<String> {
        let p = PreTokenizerWrapper::build(cfg).unwrap();
        let mut ns = NormalizedString::new(text);
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
        let cfg = PreTokenizerConfig::ByteLevel {
            add_prefix_space: false,
            trim_offsets: false,
            use_regex: true,
        };
        let v = pt(&cfg, "hi");
        assert_eq!(v, vec!["hi"]);
        let v2 = pt(&cfg, "中");
        // HF 顺序：正则先作用于**原始文本** —— "中" 命中 `\p{L}+` → 单个 piece；
        // 再对该 piece 做字节级映射 → "ä¸Ń"（3 个字符，3 = UTF-8 字节数）。
        // 若反过来先映射再切分，"Ġ" 等映射字符属 \p{L}，piece 边界会与 HF 不一致。
        assert_eq!(v2, vec!["ä¸Ń"]);
        assert_eq!(
            v2[0].chars().count(),
            3,
            "字节级展开后字符数应等于 UTF-8 字节数"
        );
        // ` ?\p{L}+` 使前导空格并入后继 piece（映射为 Ġ）
        let v3 = pt(&cfg, "hi there");
        assert_eq!(v3, vec!["hi", "Ġthere"]);
    }

    #[test]
    fn byte_level_offsets_point_to_original_bytes() {
        let cfg = PreTokenizerConfig::ByteLevel {
            add_prefix_space: false,
            trim_offsets: false,
            use_regex: true,
        };
        let p = PreTokenizerWrapper::build(&cfg).unwrap();
        let mut ns = NormalizedString::new("ab");
        p.pre_tokenize(&mut ns).unwrap();
        let ts = ns.pretokens();
        assert_eq!(ts.len(), 1);
        assert_eq!(ts[0].offsets, (0, 2));
    }

    #[test]
    fn digits_individual() {
        let v = pt(
            &PreTokenizerConfig::Digits {
                individual_digits: true,
            },
            "a12b",
        );
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
    fn whitespace_splits_punctuation_like_hf() {
        // HF `Whitespace` 等价 `\w+|[^\w\s]+`：标点各自成段，空白丢弃
        let v = pt(
            &PreTokenizerConfig::Whitespace,
            "hello, world! (Hi) 中文，标点。",
        );
        assert_eq!(
            v,
            vec!["hello", ",", "world", "!", "(", "Hi", ")", "中文", "，", "标点", "。"]
        );
    }

    #[test]
    fn whitespace_differs_from_whitespace_split_on_punctuation() {
        // 两者的唯一语义差别：Whitespace 切标点，WhitespaceSplit 不切
        assert_eq!(
            pt(&PreTokenizerConfig::Whitespace, "world!"),
            vec!["world", "!"]
        );
        assert_eq!(
            pt(&PreTokenizerConfig::WhitespaceSplit, "world!"),
            vec!["world!"]
        );
    }

    #[test]
    fn whitespace_keeps_word_chars_together() {
        // `\w` 含 Pc / Nd / M*：下划线、数字串、组合记号都不断开；emoji（So）归符号段
        let v = pt(&PreTokenizerConfig::Whitespace, "a_b 3.14 x\u{0301}y 😀😀");
        assert_eq!(v, vec!["a_b", "3", ".", "14", "x\u{0301}y", "😀😀"]);
    }

    #[test]
    fn whitespace_piece_offsets_point_to_original() {
        // split_by_matches 必须保留 normalized → original 的对齐（多字节 + 丢弃空白）
        let p = PreTokenizerWrapper::build(&PreTokenizerConfig::Whitespace).unwrap();
        let mut ns = NormalizedString::new("hi, 世界!");
        p.pre_tokenize(&mut ns).unwrap();
        let got: Vec<(String, (usize, usize))> = ns
            .pretokens()
            .into_iter()
            .map(|t| (t.value, t.offsets))
            .collect();
        assert_eq!(
            got,
            vec![
                ("hi".to_string(), (0, 2)),
                (",".to_string(), (2, 3)),
                ("世界".to_string(), (4, 10)),
                ("!".to_string(), (10, 11)),
            ]
        );
    }

    #[test]
    fn sequence_is_applied_in_order() {
        let cfg = PreTokenizerConfig::Sequence {
            pretokenizers: vec![
                PreTokenizerConfig::BertPreTokenizer,
                PreTokenizerConfig::Digits {
                    individual_digits: true,
                },
            ],
        };
        let v = pt(&cfg, "ab12");
        assert_eq!(v, vec!["ab", "1", "2"]);
    }
}
