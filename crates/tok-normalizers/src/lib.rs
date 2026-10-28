//! # tok-normalizers
//!
//! L1 规范化层实现。约束：只依赖 `tok-core`；每个组件都必须维护 original↔normalized 对齐表。

use regex::Regex;
use serde::{Deserialize, Serialize};
use tok_core::{layer, NormalizedString, PatternSpec, Result, TokenizerError};

/// L1 组件配置（与 HF `normalizer` 节的 `type` 标签一一对应）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NormalizerConfig {
    /// 恒等规范化（导出为 `null`）
    #[serde(rename = "Identity")]
    Identity,
    #[serde(rename = "Lowercase")]
    Lowercase,
    #[serde(rename = "NFC")]
    Nfc,
    #[serde(rename = "NFD")]
    Nfd,
    #[serde(rename = "NFKC")]
    Nfkc,
    #[serde(rename = "NFKD")]
    Nfkd,
    #[serde(rename = "StripAccents")]
    StripAccents,
    #[serde(rename = "Strip")]
    Strip,
    #[serde(rename = "Replace")]
    Replace {
        pattern: PatternSpec,
        content: String,
    },
    #[serde(rename = "Prepend")]
    Prepend { prepend: String },
    #[serde(rename = "Sequence")]
    Sequence { normalizers: Vec<NormalizerConfig> },
}

/// 运行时组件。
pub enum NormalizerWrapper {
    Identity,
    Lowercase,
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    StripAccents,
    Strip,
    Replace(Regex, String),
    Prepend(String),
    Sequence(Vec<NormalizerWrapper>),
    /// 扩展点：第三方自定义规范化器（无法导出为 HF 配置，导出时显式报错）
    Custom(Box<dyn tok_core::Normalizer>),
}

impl std::fmt::Debug for NormalizerWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NormalizerWrapper({})",
            <Self as tok_core::Normalizer>::name(self)
        )
    }
}

impl NormalizerWrapper {
    pub fn build(cfg: &NormalizerConfig) -> Result<Self> {
        Ok(match cfg {
            NormalizerConfig::Identity => NormalizerWrapper::Identity,
            NormalizerConfig::Lowercase => NormalizerWrapper::Lowercase,
            NormalizerConfig::Nfc => NormalizerWrapper::Nfc,
            NormalizerConfig::Nfd => NormalizerWrapper::Nfd,
            NormalizerConfig::Nfkc => NormalizerWrapper::Nfkc,
            NormalizerConfig::Nfkd => NormalizerWrapper::Nfkd,
            NormalizerConfig::StripAccents => NormalizerWrapper::StripAccents,
            NormalizerConfig::Strip => NormalizerWrapper::Strip,
            NormalizerConfig::Replace { pattern, content } => {
                NormalizerWrapper::Replace(pattern.compile()?, content.clone())
            }
            NormalizerConfig::Prepend { prepend } => NormalizerWrapper::Prepend(prepend.clone()),
            NormalizerConfig::Sequence { normalizers } => {
                let mut v = Vec::with_capacity(normalizers.len());
                for n in normalizers {
                    v.push(NormalizerWrapper::build(n)?);
                }
                NormalizerWrapper::Sequence(v)
            }
        })
    }

    /// 是否为恒等（导出时应写 `null` 而非 `{"type":"Identity"}`）
    pub fn is_identity(&self) -> bool {
        matches!(self, NormalizerWrapper::Identity)
    }

    pub fn config_of(cfg: &NormalizerConfig) -> Result<NormalizerWrapper> {
        NormalizerWrapper::build(cfg)
    }
}

impl tok_core::Normalizer for NormalizerWrapper {
    fn normalize(&self, mut ns: NormalizedString) -> NormalizedString {
        match self {
            NormalizerWrapper::Identity => ns,
            NormalizerWrapper::Lowercase => {
                ns.lowercase();
                ns
            }
            NormalizerWrapper::Nfc => {
                ns.nfc();
                ns
            }
            NormalizerWrapper::Nfd => {
                ns.nfd();
                ns
            }
            NormalizerWrapper::Nfkc => {
                ns.nfkc();
                ns
            }
            NormalizerWrapper::Nfkd => {
                ns.nfkd();
                ns
            }
            NormalizerWrapper::StripAccents => {
                ns.strip_accents();
                ns
            }
            NormalizerWrapper::Strip => {
                ns.strip();
                ns
            }
            NormalizerWrapper::Replace(re, content) => {
                ns.replace_all(re, content);
                ns
            }
            NormalizerWrapper::Prepend(s) => {
                ns.prepend(s);
                ns
            }
            NormalizerWrapper::Sequence(list) => {
                for n in list {
                    ns = n.normalize(ns);
                }
                ns
            }
            NormalizerWrapper::Custom(c) => c.normalize(ns),
        }
    }

    fn to_config_json(&self) -> Result<serde_json::Value> {
        match self {
            NormalizerWrapper::Identity => Ok(serde_json::Value::Null),
            NormalizerWrapper::Lowercase => Ok(serde_json::json!({"type": "Lowercase"})),
            NormalizerWrapper::Nfc => Ok(serde_json::json!({"type": "NFC"})),
            NormalizerWrapper::Nfd => Ok(serde_json::json!({"type": "NFD"})),
            NormalizerWrapper::Nfkc => Ok(serde_json::json!({"type": "NFKC"})),
            NormalizerWrapper::Nfkd => Ok(serde_json::json!({"type": "NFKD"})),
            NormalizerWrapper::StripAccents => Ok(serde_json::json!({"type": "StripAccents"})),
            NormalizerWrapper::Strip => Ok(serde_json::json!({"type": "Strip"})),
            NormalizerWrapper::Replace(re, content) => Ok(serde_json::json!({
                "type": "Replace",
                "pattern": {"Regex": re.as_str()},
                "content": content,
            })),
            NormalizerWrapper::Prepend(s) => {
                Ok(serde_json::json!({"type": "Prepend", "prepend": s}))
            }
            NormalizerWrapper::Sequence(list) => {
                let mut items = Vec::with_capacity(list.len());
                for n in list {
                    items.push(n.to_config_json()?);
                }
                Ok(serde_json::json!({"type": "Sequence", "normalizers": items}))
            }
            NormalizerWrapper::Custom(_) => Err(TokenizerError::layer(
                layer::NORMALIZER,
                "Custom normalizer 无法导出为 HF 配置",
            )),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            NormalizerWrapper::Identity => "Identity",
            NormalizerWrapper::Lowercase => "Lowercase",
            NormalizerWrapper::Nfc => "NFC",
            NormalizerWrapper::Nfd => "NFD",
            NormalizerWrapper::Nfkc => "NFKC",
            NormalizerWrapper::Nfkd => "NFKD",
            NormalizerWrapper::StripAccents => "StripAccents",
            NormalizerWrapper::Strip => "Strip",
            NormalizerWrapper::Replace(_, _) => "Replace",
            NormalizerWrapper::Prepend(_) => "Prepend",
            NormalizerWrapper::Sequence(_) => "Sequence",
            NormalizerWrapper::Custom(_) => "Custom",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tok_core::Normalizer;

    #[test]
    fn lowercase_then_strip_accents() {
        let cfg = NormalizerConfig::Sequence {
            normalizers: vec![NormalizerConfig::Lowercase, NormalizerConfig::StripAccents],
        };
        let n = NormalizerWrapper::build(&cfg).unwrap();
        let out = n.normalize(NormalizedString::new("CAFÉ"));
        assert_eq!(out.normalized(), "cafe");
        let o = out.convert_offsets((0, 4)).unwrap();
        assert_eq!((o.start, o.end), (0, 5));
    }

    #[test]
    fn nfkc_folds_fullwidth_and_flags_degraded() {
        let n = NormalizerWrapper::build(&NormalizerConfig::Nfkc).unwrap();
        let out = n.normalize(NormalizedString::new("ＡＢ"));
        assert_eq!(out.normalized(), "AB");
        assert!(out.is_degraded(), "字符数变化必须显式标记退化");
    }

    #[test]
    fn replace_with_regex_pattern_spec() {
        let cfg = NormalizerConfig::Replace {
            pattern: PatternSpec::new("\\s+"),
            content: "_".to_string(),
        };
        let n = NormalizerWrapper::build(&cfg).unwrap();
        let out = n.normalize(NormalizedString::new("a   b"));
        assert_eq!(out.normalized(), "a_b");
    }

    #[test]
    fn prepend_maps_to_original_start() {
        let n = NormalizerWrapper::build(&NormalizerConfig::Prepend {
            prepend: "\u{2581}".into(),
        })
        .unwrap();
        let out = n.normalize(NormalizedString::new("hello"));
        assert_eq!(out.normalized(), "\u{2581}hello");
        let o = out.convert_offsets((3, 8)).unwrap();
        assert_eq!((o.start, o.end), (0, 5));
    }

    #[test]
    fn config_json_matches_hf_shape() {
        let cfg = NormalizerConfig::Replace {
            pattern: PatternSpec::new("x"),
            content: "y".into(),
        };
        let n = NormalizerWrapper::build(&cfg).unwrap();
        let v = n.to_config_json().unwrap();
        assert_eq!(v["type"], "Replace");
        assert_eq!(v["pattern"]["Regex"], "x");
        assert_eq!(v["content"], "y");
    }
}
