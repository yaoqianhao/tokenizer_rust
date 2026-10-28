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
    /// 字节级重映射（GPT-2 风格）：`你好` → `ä½ł` 这类可打印化表示。
    ///
    /// HF 的该 normalizer 无参数，落盘形如 `{"type": "ByteLevel"}`。
    #[serde(rename = "ByteLevel")]
    ByteLevel,
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
    ByteLevel,
    Replace {
        /// 原始 pattern（落盘时必须还原 `Regex` / `String` 标签）
        spec: PatternSpec,
        re: Regex,
        content: String,
    },
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
            NormalizerConfig::ByteLevel => NormalizerWrapper::ByteLevel,
            NormalizerConfig::Replace { pattern, content } => NormalizerWrapper::Replace {
                spec: pattern.clone(),
                re: pattern.compile()?,
                content: content.clone(),
            },
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
            NormalizerWrapper::ByteLevel => {
                ns.byte_level_expand();
                ns
            }
            NormalizerWrapper::Replace { re, content, .. } => {
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
            // HF 的 Strip 配置要求 strip_left / strip_right 两个必填布尔字段，
            // 缺失会导致 tokenizer.json 加载报 `missing field strip_left`。
            // 本实现 `ns.strip()` 同时裁剪两端，故双双为 true。
            NormalizerWrapper::Strip => {
                Ok(serde_json::json!({"type": "Strip", "strip_left": true, "strip_right": true}))
            }
            NormalizerWrapper::ByteLevel => Ok(serde_json::json!({"type": "ByteLevel"})),
            NormalizerWrapper::Replace { spec, content, .. } => Ok(serde_json::json!({
                "type": "Replace",
                "pattern": spec.to_hf_json(),
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
            NormalizerWrapper::ByteLevel => "ByteLevel",
            NormalizerWrapper::Replace { .. } => "Replace",
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

    #[test]
    fn replace_literal_does_not_interpret_metachars() {
        // String 模式：`.` 是普通字符，只命中真正的句点
        let cfg = NormalizerConfig::Replace {
            pattern: PatternSpec::literal("."),
            content: "_".into(),
        };
        let n = NormalizerWrapper::build(&cfg).unwrap();
        assert_eq!(
            n.normalize(NormalizedString::new("a.b")).normalized(),
            "a_b"
        );

        // 对照 Regex 模式：`.` 命中任意字符，四字符全被替换
        let re_cfg = NormalizerConfig::Replace {
            pattern: PatternSpec::new("."),
            content: "_".into(),
        };
        let re_n = NormalizerWrapper::build(&re_cfg).unwrap();
        assert_eq!(
            re_n.normalize(NormalizedString::new("a.b")).normalized(),
            "___"
        );
    }

    #[test]
    fn replace_literal_config_json_uses_string_tag() {
        let cfg = NormalizerConfig::Replace {
            pattern: PatternSpec::literal("\u{a0}"),
            content: " ".into(),
        };
        let v = NormalizerWrapper::build(&cfg)
            .unwrap()
            .to_config_json()
            .unwrap();
        assert_eq!(v["pattern"]["String"], "\u{a0}");
        assert!(v["pattern"].get("Regex").is_none());
    }

    /// 落盘 → 反序列化 往返：`String` 标签不能被重读成正则。
    #[test]
    fn pattern_spec_roundtrip_keeps_literal_tag() {
        let cfg = NormalizerConfig::Replace {
            pattern: PatternSpec::literal("[a-z]+"),
            content: "#".into(),
        };
        let v = NormalizerWrapper::build(&cfg)
            .unwrap()
            .to_config_json()
            .unwrap();
        let back: NormalizerConfig = serde_json::from_value(v).unwrap();
        match &back {
            NormalizerConfig::Replace { pattern, .. } => {
                assert!(pattern.is_literal(), "String 模式重读后必须仍是字面量");
                assert_eq!(pattern.as_str(), "[a-z]+");
            }
            other => panic!("期望 Replace，得到 {:?}", other),
        }
        // 字面量语义：方括号与加号都不解释，只命中原文 "[a-z]+"
        let n = NormalizerWrapper::build(&back).unwrap();
        assert_eq!(
            n.normalize(NormalizedString::new("x [a-z]+ y"))
                .normalized(),
            "x # y"
        );
    }

    #[test]
    fn byte_level_expands_to_gpt2_visible_bytes() {
        let n = NormalizerWrapper::build(&NormalizerConfig::ByteLevel).unwrap();
        // '你' = E4 BD A0 → ä½ł（HF ByteLevel 的可打印映射）
        let out = n.normalize(NormalizedString::new("你"));
        assert_eq!(out.normalized(), "ä½ł");
        // 字节级映射是可逆的一对一变换（1 字节 ↔ 1 映射字符），不属于信息退化
        assert_eq!(out.normalized().chars().count(), "你".len());
        assert!(!out.is_degraded());
        // 映射后可回推原文偏移（偏移量一律是 UTF-8 字节）
        let o = out.convert_offsets((0, out.normalized().len())).unwrap();
        assert_eq!((o.start, o.end), (0, 3));

        let v = n.to_config_json().unwrap();
        assert_eq!(v, serde_json::json!({"type": "ByteLevel"}));
    }
}
