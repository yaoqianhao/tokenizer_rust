//! # tok-decoders
//!
//! L6 解码层：把 token 序列还原为可读文本。
//!
//! `decode_chain` 保持 HF 语义——逐步对 token 列表做变换（可能改变长度），
//! 最终由调用方把结果拼接成字符串（`decode_ids`）。

use serde::{Deserialize, Serialize};
use tok_core::{Decoder, PatternSpec, Result, TokenizerError, Vocabulary, byte_level, layer};

/// 解码器配置（HF `decoder` 节）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DecoderConfig {
    #[serde(rename = "ByteLevel")]
    ByteLevel {
        #[serde(default)]
        add_prefix_space: bool,
        #[serde(default)]
        trim_offsets: bool,
        #[serde(default = "default_true")]
        use_regex: bool,
    },
    #[serde(rename = "WordPiece")]
    WordPiece {
        #[serde(default = "default_wordpiece_prefix")]
        prefix: String,
        #[serde(default = "default_true")]
        cleanup: bool,
    },
    #[serde(rename = "Metaspace")]
    Metaspace {
        #[serde(default = "default_replacement")]
        replacement: String,
        #[serde(default = "default_scheme")]
        prepend_scheme: String,
        #[serde(default = "default_true")]
        split: bool,
    },
    #[serde(rename = "BPEDecoder")]
    BPEDecoder {
        #[serde(default = "default_eow")]
        suffix: String,
    },
    #[serde(rename = "CTC")]
    CTC {
        #[serde(default = "default_pad")]
        pad_token: String,
        #[serde(default = "default_word_delim")]
        word_delimiter_token: String,
        #[serde(default = "default_true")]
        cleanup: bool,
    },
    #[serde(rename = "Replace")]
    Replace { pattern: PatternSpec, content: String },
    #[serde(rename = "ByteFallback")]
    ByteFallback,
    #[serde(rename = "Fuse")]
    Fuse { decoders: Vec<DecoderConfig> },
    #[serde(rename = "Sequence")]
    Sequence { decoders: Vec<DecoderConfig> },
    #[serde(rename = "Strip")]
    Strip {
        content: String,
        #[serde(default)]
        start: usize,
        #[serde(default = "default_strip_stop")]
        stop: usize,
    },
}

fn default_true() -> bool {
    true
}
fn default_wordpiece_prefix() -> String {
    "##".to_string()
}
fn default_replacement() -> String {
    "\u{2581}".to_string()
}
fn default_scheme() -> String {
    "always".to_string()
}
fn default_eow() -> String {
    "</w>".to_string()
}
fn default_pad() -> String {
    "<pad>".to_string()
}
fn default_word_delim() -> String {
    "|".to_string()
}
fn default_strip_stop() -> usize {
    1
}

/// 运行时解码器。
#[derive(Debug, Clone)]
pub struct DecoderWrapper {
    cfg: Option<DecoderConfig>,
}

impl DecoderWrapper {
    /// 无解码器：token 原文直接拼接（HF 中 `decoder` 缺省）。
    pub fn identity() -> Self {
        Self { cfg: None }
    }

    pub fn is_identity(&self) -> bool {
        self.cfg.is_none()
    }

    pub fn config(&self) -> Option<&DecoderConfig> {
        self.cfg.as_ref()
    }

    pub fn build(cfg: &DecoderConfig) -> Result<Self> {
        Ok(Self { cfg: Some(cfg.clone()) })
    }

    /// 常用组合：`ByteLevel` 预分词 + `ByteLevel` 解码（GPT-2 家族）。
    pub fn byte_level() -> Self {
        Self::build(&DecoderConfig::ByteLevel {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        })
        .unwrap()
    }

    /// `ids → 文本`：先 `decode_chain`，再无损拼接。
    pub fn decode_ids(&self, ids: &[u32], vocab: &Vocabulary) -> Result<String> {
        let mut tokens: Vec<String> = Vec::with_capacity(ids.len());
        for id in ids {
            match vocab.id_to_token(*id) {
                Some(t) => tokens.push(t.to_string()),
                None => {
                    return Err(TokenizerError::layer(
                        layer::DECODER,
                        format!("解码失败：词表中不存在 id {}", id),
                    ))
                }
            }
        }
        let chain = self.decode_chain(tokens)?;
        Ok(chain.concat())
    }

    /// `ids → 文本`，跳过特殊 token（`skip_special_tokens = true` 语义）。
    pub fn decode_ids_skip_special(&self, ids: &[u32], vocab: &Vocabulary, specials: &[u32]) -> Result<String> {
        let kept: Vec<u32> = ids.iter().copied().filter(|i| !specials.contains(i)).collect();
        self.decode_ids(&kept, vocab)
    }
}

impl Decoder for DecoderWrapper {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        match &self.cfg {
            None => Ok(tokens),
            Some(DecoderConfig::ByteLevel { .. }) => {
                // HF 语义：先把所有 token 还原为字节、整体拼接，再做**一次** UTF-8 解码。
                // 逐 token 解码会把被 BPE 撕开的多字节字符解成 U+FFFD（如 "中" → 3 个字节级 token）。
                let mut bytes: Vec<u8> = Vec::new();
                for t in &tokens {
                    bytes.extend(byte_level::byte_level_decode(t));
                }
                Ok(vec![String::from_utf8_lossy(&bytes).into_owned()])
            }
            Some(DecoderConfig::WordPiece { prefix, cleanup }) => {
                let mut out: Vec<String> = tokens
                    .into_iter()
                    .map(|t| match t.strip_prefix(prefix.as_str()) {
                        Some(rest) => rest.to_string(),
                        None => t,
                    })
                    .collect();
                if out.is_empty() {
                    return Ok(out);
                }
                if *cleanup {
                    // HF：把段首的空格并入前一段（"##" 前缀反解后合并词内片段）
                    let mut merged: Vec<String> = Vec::with_capacity(out.len());
                    for t in out.drain(..) {
                        match merged.last_mut() {
                            Some(prev) if !prev.is_empty() && !t.starts_with(' ') => prev.push_str(&t),
                            _ => merged.push(t),
                        }
                    }
                    out = merged;
                }
                Ok(out)
            }
            Some(DecoderConfig::Metaspace { replacement, .. }) => Ok(tokens
                .into_iter()
                .map(|t| t.replace(replacement.as_str(), " "))
                .collect()),
            Some(DecoderConfig::BPEDecoder { suffix }) => Ok(tokens
                .into_iter()
                .map(|t| t.strip_suffix(suffix.as_str()).map(|s| s.to_string()).unwrap_or(t))
                .collect()),
            Some(DecoderConfig::CTC { pad_token, word_delimiter_token, cleanup }) => {
                let mut out = Vec::new();
                for t in tokens {
                    if *cleanup && &t == pad_token {
                        continue;
                    }
                    if &t == word_delimiter_token {
                        out.push(" ".to_string());
                    } else {
                        out.push(t.replace(word_delimiter_token.as_str(), " "));
                    }
                }
                Ok(out)
            }
            Some(DecoderConfig::Replace { pattern, content }) => {
                let re = compile(pattern)?;
                Ok(tokens.into_iter().map(|t| re.replace_all(&t, content.as_str()).into_owned()).collect())
            }
            Some(DecoderConfig::ByteFallback) => {
                let mut out = Vec::with_capacity(tokens.len());
                let mut pending: Vec<u8> = Vec::new();
                let flush = |pending: &mut Vec<u8>, out: &mut Vec<String>| {
                    if !pending.is_empty() {
                        out.push(String::from_utf8_lossy(pending).into_owned());
                        pending.clear();
                    }
                };
                for t in tokens {
                    if let Some(hex) = t.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
                        match u8::from_str_radix(hex, 16) {
                            Ok(b) => pending.push(b),
                            Err(_) => {
                                flush(&mut pending, &mut out);
                                out.push(t);
                            }
                        }
                    } else {
                        flush(&mut pending, &mut out);
                        out.push(t);
                    }
                }
                flush(&mut pending, &mut out);
                Ok(out)
            }
            Some(DecoderConfig::Fuse { decoders }) | Some(DecoderConfig::Sequence { decoders }) => {
                let mut cur = tokens;
                for d in decoders {
                    cur = DecoderWrapper::build(d)?.decode_chain(cur)?;
                }
                Ok(cur)
            }
            Some(DecoderConfig::Strip { content, start, stop }) => Ok(tokens
                .into_iter()
                .map(|mut t| {
                    if *start > 0 && t.starts_with(content) {
                        t = t[content.len().min(t.len())..].to_string();
                    }
                    if *stop > 0 && t.ends_with(content) {
                        let cut = t.len().saturating_sub(content.len());
                        t.truncate(cut);
                    }
                    t
                })
                .collect()),
        }
    }

    fn to_config_json(&self) -> Result<serde_json::Value> {
        match &self.cfg {
            None => Ok(serde_json::Value::Null),
            Some(c) => Ok(serde_json::to_value(c)?),
        }
    }

    fn name(&self) -> &'static str {
        match &self.cfg {
            None => "None",
            Some(DecoderConfig::ByteLevel { .. }) => "ByteLevel",
            Some(DecoderConfig::WordPiece { .. }) => "WordPiece",
            Some(DecoderConfig::Metaspace { .. }) => "Metaspace",
            Some(DecoderConfig::BPEDecoder { .. }) => "BPEDecoder",
            Some(DecoderConfig::CTC { .. }) => "CTC",
            Some(DecoderConfig::Replace { .. }) => "Replace",
            Some(DecoderConfig::ByteFallback) => "ByteFallback",
            Some(DecoderConfig::Fuse { .. }) => "Fuse",
            Some(DecoderConfig::Sequence { .. }) => "Sequence",
            Some(DecoderConfig::Strip { .. }) => "Strip",
        }
    }
}

fn compile(pattern: &PatternSpec) -> Result<regex::Regex> {
    regex::Regex::new(pattern.as_str())
        .map_err(|e| TokenizerError::Regex(format!("解码器 Replace 正则非法: {} ({})", pattern.as_str(), e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tok_core::Decoder;

    fn vocab() -> Vocabulary {
        let mut v = Vocabulary::new();
        for (t, id) in [("H", 0u32), ("i", 1), ("Ġ", 2), ("there", 3), ("##ing", 4), ("go", 5)] {
            v.insert(t, id).unwrap();
        }
        v
    }

    #[test]
    fn bytelevel_decodes_gpt2_style() {
        let d = DecoderWrapper::byte_level();
        // "Hi there" 经字节级编码后为 "H", "i", "Ġ", "there"
        let text = d.decode_ids(&[0, 1, 2, 3], &vocab()).unwrap();
        assert_eq!(text, "Hi there");
    }

    #[test]
    fn wordpiece_strips_prefix_and_merges() {
        let d = DecoderWrapper::build(&DecoderConfig::WordPiece { prefix: "##".into(), cleanup: true }).unwrap();
        let chain = d.decode_chain(vec!["go".into(), "##ing".into()]).unwrap();
        assert_eq!(chain.concat(), "going");
    }

    #[test]
    fn metaspace_restores_spaces() {
        let d = DecoderWrapper::build(&DecoderConfig::Metaspace {
            replacement: "\u{2581}".into(),
            prepend_scheme: "always".into(),
            split: true,
        })
        .unwrap();
        let chain = d.decode_chain(vec!["\u{2581}Hello".into(), "\u{2581}world".into()]).unwrap();
        assert_eq!(chain.concat(), " Hello world");
    }

    #[test]
    fn byte_fallback_rebuilds_utf8() {
        let d = DecoderWrapper::build(&DecoderConfig::ByteFallback).unwrap();
        // "中" = E4 B8 AD
        let chain = d.decode_chain(vec!["<0xE4>".into(), "<0xB8>".into(), "<0xAD>".into()]).unwrap();
        assert_eq!(chain.concat(), "中");
    }

    #[test]
    fn identity_and_round_trip_config() {
        let d = DecoderWrapper::identity();
        assert!(d.is_identity());
        assert_eq!(d.to_config_json().unwrap(), serde_json::Value::Null);
        let cfg = DecoderConfig::CTC {
            pad_token: "<pad>".into(),
            word_delimiter_token: "|".into(),
            cleanup: true,
        };
        let w = DecoderWrapper::build(&cfg).unwrap();
        assert_eq!(serde_json::from_value::<DecoderConfig>(w.to_config_json().unwrap()).unwrap(), cfg);
    }

    #[test]
    fn strip_removes_suffix() {
        let d = DecoderWrapper::build(&DecoderConfig::Strip { content: "</w>".into(), start: 0, stop: 1 }).unwrap();
        let chain = d.decode_chain(vec!["low</w>".into(), "est".into()]).unwrap();
        assert_eq!(chain, vec!["low".to_string(), "est".to_string()]);
    }
}
