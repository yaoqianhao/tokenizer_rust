//! HuggingFace fast tokenizer 互操作：`tokenizer.json` / `tokenizer_config.json` /
//! `special_tokens_map.json` 的构造与落盘。
//!
//! 字段对齐 HF `tokenizers` 0.20+ 的 `PreTrainedTokenizerFast` 加载要求：
//! `model.type`、`vocab`、`merges`（`[["a","b"], ...]` 形式）、`normalizer`、`pre_tokenizer`、
//! `post_processor`、`decoder`、`added_tokens`。

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tok_core::{AddedToken, Result, TokenizerError};

/// 文件名常量（HF 目录约定）
pub const TOKENIZER_JSON: &str = "tokenizer.json";
pub const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";
pub const SPECIAL_TOKENS_MAP_JSON: &str = "special_tokens_map.json";
/// `config.json`（`AutoTokenizer.from_pretrained(dir)` 识别模型类型所需）
pub const CONFIG_JSON: &str = "config.json";
/// `tokenizer.json` 的 schema 版本
pub const HF_SCHEMA_VERSION: &str = "1.0";

/// `config.json` 内容：只写 `AutoTokenizer` 识别身份所需的最小集，
/// 不掺入真实模型的超参（hidden_size / num_layers 等），避免与骨干配置混淆。
pub fn model_config(model_type: &str, tokenizer_class: &str) -> Value {
    json!({
        "model_type": model_type,
        "tokenizer_class": tokenizer_class,
    })
}

/// 由模型实现类型推断 HF `model_type` / `tokenizer_class`（无法高置信推断时返回 `None`）。
///
/// 依据：BPE → GPT-2 系（ByteLevel 即 GPT-2 上游）；WordPiece → BERT 系；
/// Unigram → XLM-R 系。WordLevel 等无稳定对应，需调用方显式指定。
pub fn infer_hf_model_type(model_type: &str) -> Option<(&'static str, &'static str)> {
    match model_type {
        "BPE" => Some(("gpt2", "GPT2Tokenizer")),
        "WordPiece" => Some(("bert", "BertTokenizer")),
        "Unigram" => Some(("xlm-roberta", "XLMRobertaTokenizer")),
        _ => None,
    }
}

/// 角色 token 字面量（写入 `tokenizer_config.json` 与 `special_tokens_map.json`）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenRoles {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unk_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bos_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eos_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cls_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sep_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask_token: Option<String>,
    /// 追加的特殊 token（不含角色 token）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_special_tokens: Vec<String>,
}

impl TokenRoles {
    /// 从后加 token 列表做启发式识别（未显式配置时的兜底）。
    pub fn infer(added: &[AddedToken]) -> Self {
        let mut roles = TokenRoles::default();
        for t in added.iter().filter(|t| t.special) {
            let c = t.content.as_str();
            let lower = c.to_ascii_lowercase();
            if roles.unk_token.is_none() && (lower.contains("unk") || c == "<unk>") {
                roles.unk_token = Some(c.to_string());
            } else if roles.bos_token.is_none()
                && (c == "<s>" || lower.contains("bos") || lower.contains("startoftext") || lower.contains("start_of_text"))
            {
                roles.bos_token = Some(c.to_string());
            } else if roles.eos_token.is_none()
                && (c == "</s>" || lower.contains("eos") || lower.contains("endoftext") || lower.contains("end_of_text"))
            {
                roles.eos_token = Some(c.to_string());
            } else if roles.pad_token.is_none() && (lower.contains("pad") || c == "<pad>") {
                roles.pad_token = Some(c.to_string());
            } else if roles.cls_token.is_none() && (lower.contains("cls") || c == "[CLS]") {
                roles.cls_token = Some(c.to_string());
            } else if roles.sep_token.is_none() && (lower.contains("sep") || c == "[SEP]") {
                roles.sep_token = Some(c.to_string());
            } else if roles.mask_token.is_none() && (lower.contains("mask") || c == "[MASK]") {
                roles.mask_token = Some(c.to_string());
            } else if !roles.additional_special_tokens.contains(&c.to_string()) {
                roles.additional_special_tokens.push(c.to_string());
            }
        }
        roles
    }

    /// 把角色 token 补进 `added`（缺失时追加为 special AddedToken）。
    pub fn ensure_in_added(&self, added: &mut Vec<AddedToken>) {
        for c in self.all() {
            if !added.iter().any(|t| t.content == c) {
                added.push(AddedToken::new(c.clone(), true));
            }
        }
    }

    /// 用另一份角色表补全本表**尚未设置**的槽位（已显式配置的角色优先，不被覆盖）。
    pub fn fill_missing(&mut self, other: &TokenRoles) {
        fn take(slot: &mut Option<String>, src: &Option<String>) {
            if slot.is_none() {
                *slot = src.clone();
            }
        }
        take(&mut self.unk_token, &other.unk_token);
        take(&mut self.bos_token, &other.bos_token);
        take(&mut self.eos_token, &other.eos_token);
        take(&mut self.pad_token, &other.pad_token);
        take(&mut self.cls_token, &other.cls_token);
        take(&mut self.sep_token, &other.sep_token);
        take(&mut self.mask_token, &other.mask_token);
        for t in &other.additional_special_tokens {
            if !self.additional_special_tokens.contains(t) {
                self.additional_special_tokens.push(t.clone());
            }
        }
    }

    /// 是否未设置任何角色 token。
    pub fn is_empty(&self) -> bool {
        self.all().is_empty()
    }

    pub fn all(&self) -> Vec<String> {
        let mut v = Vec::new();
        for o in [
            &self.unk_token,
            &self.bos_token,
            &self.eos_token,
            &self.pad_token,
            &self.cls_token,
            &self.sep_token,
            &self.mask_token,
        ]
        .into_iter()
        .flatten()
        {
            if !v.contains(o) {
                v.push(o.clone());
            }
        }
        for t in &self.additional_special_tokens {
            if !v.contains(t) {
                v.push(t.clone());
            }
        }
        v
    }
}

/// `tokenizer.json` 顶层（写入侧用强类型，读取侧用 `Value` 更宽容）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfTokenizerJson {
    pub version: String,
    #[serde(default)]
    pub truncation: Value,
    #[serde(default)]
    pub padding: Value,
    #[serde(default)]
    pub added_tokens: Vec<Value>,
    #[serde(default)]
    pub normalizer: Value,
    #[serde(default)]
    pub pre_tokenizer: Value,
    #[serde(default)]
    pub post_processor: Value,
    #[serde(default)]
    pub decoder: Value,
    pub model: Value,
}

/// `special_tokens_map.json` 内容。
pub fn special_tokens_map(roles: &TokenRoles) -> Value {
    json!({
        "unk_token": roles.unk_token.clone(),
        "bos_token": roles.bos_token.clone(),
        "eos_token": roles.eos_token.clone(),
        "pad_token": roles.pad_token.clone(),
        "cls_token": roles.cls_token.clone(),
        "sep_token": roles.sep_token.clone(),
        "mask_token": roles.mask_token.clone(),
        "additional_special_tokens": roles.additional_special_tokens.clone(),
    })
}

/// `tokenizer_config.json` 内容。
pub fn tokenizer_config(
    roles: &TokenRoles,
    added_with_ids: &[(AddedToken, u32)],
    model_max_length: usize,
    model_type: &str,
) -> Value {
    json!({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": model_max_length,
        "model_type": model_type,
        "clean_up_tokenization_spaces": false,
        "added_tokens_decoder": added_tokens_decoder(added_with_ids),
        "unk_token": roles.unk_token.clone(),
        "bos_token": roles.bos_token.clone(),
        "eos_token": roles.eos_token.clone(),
        "pad_token": roles.pad_token.clone(),
        "cls_token": roles.cls_token.clone(),
        "sep_token": roles.sep_token.clone(),
        "mask_token": roles.mask_token.clone(),
        "additional_special_tokens": roles.additional_special_tokens.clone(),
    })
}

/// `added_tokens_decoder`：`{"<id>": {content, single_word, lstrip, rstrip, normalized, special}}`
pub fn added_tokens_decoder(added: &[(AddedToken, u32)]) -> Value {
    let mut map = serde_json::Map::new();
    for (t, id) in added {
        map.insert(id.to_string(), serde_json::to_value(t).unwrap_or(Value::Null));
    }
    Value::Object(map)
}

/// 写入 JSON 文件（2 空格缩进 + 末尾换行），返回路径与行数。
pub fn write_json(path: &Path, value: &Value) -> Result<(PathBuf, usize)> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(value)?;
    let text = format!("{}\n", text);
    let lines = text.lines().count();
    std::fs::write(path, text)?;
    Ok((path.to_path_buf(), lines))
}

/// 校验写出的 `tokenizer.json` 是否满足"HF fast 必需字段"（缺一即报错，不静默）。
pub fn validate_tokenizer_json(value: &Value) -> Result<()> {
    for key in ["version", "model", "added_tokens"] {
        if value.get(key).is_none() {
            return Err(TokenizerError::config(format!("tokenizer.json 缺少必需字段 `{}`", key)));
        }
    }
    let model = value.get("model").ok_or_else(|| TokenizerError::config("tokenizer.json 缺少 model 节"))?;
    let ty = model.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
        TokenizerError::config("tokenizer.json 的 model 节缺少 `type`（HF 要求 BPE/WordPiece/WordLevel/Unigram）")
    })?;
    // `merges` 形状校验与模型类型**无关**：只要该字段出现就必须是合法形状，
    // 否则（如 WordPiece 里塞了残缺 merges）会被静默带过。
    if model.get("merges").is_some() && !model["merges"].is_null() {
        let merges = model
            .get("merges")
            .and_then(|m| m.as_array())
            .ok_or_else(|| TokenizerError::config("`merges` 必须为数组"))?;
        for m in merges {
            let ok = m.as_array().map(|p| p.len() == 2 && p.iter().all(|x| x.is_string())).unwrap_or(false)
                || m.is_string();
            if !ok {
                return Err(TokenizerError::config(
                    "merges 元素必须为 [\"a\",\"b\"] 二元组（或 HF 旧式 \"a b\" 字符串）",
                ));
            }
        }
    }
    match ty {
        "BPE" => {
            if model.get("vocab").is_none() || model.get("merges").is_none() {
                return Err(TokenizerError::config("BPE 模型必须同时含 `vocab` 与 `merges`"));
            }
        }
        "WordPiece" | "WordLevel" => {
            if model.get("vocab").is_none() {
                return Err(TokenizerError::config(format!("{} 模型必须含 `vocab`", ty)));
            }
        }
        "Unigram" => {
            let vocab = model.get("vocab").and_then(|v| v.as_array()).ok_or_else(|| {
                TokenizerError::config("Unigram 的 `vocab` 必须为 [[piece, score], ...] 数组")
            })?;
            for e in vocab {
                let ok = e.as_array().map(|p| p.len() == 2 && p[0].is_string() && p[1].is_number()).unwrap_or(false);
                if !ok {
                    return Err(TokenizerError::config("Unigram vocab 元素必须为 [piece, score]"));
                }
            }
        }
        other => {
            return Err(TokenizerError::config(format!(
                "unknown model type: {}（HF 仅接受 BPE/WordPiece/WordLevel/Unigram）",
                other
            )))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_inferred_from_special_tokens() {
        let added = vec![
            AddedToken::new("<unk>", true),
            AddedToken::new("<s>", true),
            AddedToken::new("</s>", true),
            AddedToken::new("<|endoftext|>", true),
        ];
        let roles = TokenRoles::infer(&added);
        assert_eq!(roles.unk_token.as_deref(), Some("<unk>"));
        assert_eq!(roles.bos_token.as_deref(), Some("<s>"));
        assert_eq!(roles.eos_token.as_deref(), Some("</s>"));
        let map = special_tokens_map(&roles);
        assert_eq!(map["unk_token"], json!("<unk>"));
        assert!(map["pad_token"].is_null());
    }

    #[test]
    fn validate_rejects_bad_merges() {
        let bad = json!({
            "version": "1.0",
            "added_tokens": [],
            "model": {"type": "BPE", "vocab": {"a": 0}, "merges": [[1, 2]]}
        });
        assert!(validate_tokenizer_json(&bad).is_err());
        let good = json!({
            "version": "1.0",
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {"type": "BPE", "vocab": {"a": 0}, "merges": [["a", "b"]]}
        });
        assert!(validate_tokenizer_json(&good).is_ok());
    }

    #[test]
    fn added_tokens_decoder_maps_ids() {
        let v = added_tokens_decoder(&[(AddedToken::new("<s>", true), 3)]);
        assert_eq!(v["3"]["content"], json!("<s>"));
        assert_eq!(v["3"]["special"], json!(true));
    }
}
