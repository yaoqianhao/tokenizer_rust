//! # tok-compat
//!
//! **HuggingFace 互操作层**（L0 的旁路，不在六层内核里）：
//!
//! * 直接复用 [`tok_pipeline::hf`] 的 fast tokenizer 三件套读写能力（避免双实现漂移）；
//! * 额外兼容 **legacy slow tokenizer 资产**：`vocab.json` + `merges.txt` → [`ModelConfig`]；
//! * 提供目录级体检（判定一个目录里有哪些 HF 产物、是否可用于加载）。
//!
//! 依赖方向：本 crate 位于流水线之上，只允许被 CLI / 外部工具依赖，**不得被内核层引用**。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tok_core::{Result, TokenizerError};
use tok_models::{MergeSpec, ModelConfig};

pub use tok_pipeline::hf::{
    HF_SCHEMA_VERSION, HfTokenizerJson, SPECIAL_TOKENS_MAP_JSON, TOKENIZER_CONFIG_JSON, TOKENIZER_JSON, TokenRoles,
    added_tokens_decoder, special_tokens_map, tokenizer_config, validate_tokenizer_json, write_json,
};

/// 构造 IO 类错误（`TokenizerError` 只暴露了部分便捷构造器）。
fn io_err(message: impl Into<String>) -> TokenizerError {
    TokenizerError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, message.into()))
}

/// 一个目录里的 HF tokenizer 产物清单。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArtifactInventory {
    /// `tokenizer.json` 是否存在（fast tokenizer 加载必需的入口）
    pub tokenizer_json: bool,
    /// `tokenizer_config.json` 是否存在
    pub tokenizer_config_json: bool,
    /// `special_tokens_map.json` 是否存在
    pub special_tokens_map_json: bool,
    /// 目录中出现的其它 HF 相关文件（如 `vocab.json` / `merges.txt` / `vocab.txt`）
    pub others: Vec<PathBuf>,
}

impl ArtifactInventory {
    /// 是否满足 fast tokenizer 的最小加载条件。
    pub fn is_loadable_fast(&self) -> bool {
        self.tokenizer_json
    }

    /// 是否三件套齐全（可被 transformers 完整识别，含 special tokens 元信息）。
    pub fn is_complete(&self) -> bool {
        self.tokenizer_json && self.tokenizer_config_json && self.special_tokens_map_json
    }
}

/// 体检目录：列出 HF tokenizer 相关产物。
pub fn inventory(dir: &Path) -> Result<ArtifactInventory> {
    if !dir.is_dir() {
        return Err(io_err(format!("目录不存在: {}", dir.display())));
    }
    let mut inv = ArtifactInventory::default();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        let path = entry.path();
        match name.as_str() {
            TOKENIZER_JSON => inv.tokenizer_json = true,
            TOKENIZER_CONFIG_JSON => inv.tokenizer_config_json = true,
            SPECIAL_TOKENS_MAP_JSON => inv.special_tokens_map_json = true,
            "vocab.json" | "merges.txt" | "vocab.txt" | "added_tokens.json" => inv.others.push(path),
            _ => {}
        }
    }
    inv.others.sort();
    Ok(inv)
}

// ------------------------------------------------------------------ legacy 资产

/// 读取 legacy slow tokenizer 的 `vocab.json`（token → id）。
pub fn load_vocab_json(path: &Path) -> Result<BTreeMap<String, u32>> {
    let text = std::fs::read_to_string(path)?;
    let value: Value = serde_json::from_str(&text)?;
    let obj = value
        .as_object()
        .ok_or_else(|| TokenizerError::config("vocab.json 顶层必须是对象"))?;
    let mut out = BTreeMap::new();
    for (token, id) in obj {
        let id = id
            .as_u64()
            .ok_or_else(|| TokenizerError::config(format!("vocab.json 中 '{}' 的 id 不是整数", token)))?;
        out.insert(token.clone(), id as u32);
    }
    Ok(out)
}

/// 读取 legacy `merges.txt`（`#version` 头与空行自动跳过）。
pub fn load_merges_txt(path: &Path) -> Result<Vec<MergeSpec>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r').trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let a = it.next();
        let b = it.next();
        match (a, b, it.next()) {
            (Some(a), Some(b), None) => out.push(MergeSpec::from_pair(a, b)),
            _ => {
                return Err(TokenizerError::config(format!(
                    "merges.txt 第 {} 行不是一对 token: '{}'",
                    lineno + 1,
                    line
                )));
            }
        }
    }
    Ok(out)
}

/// 由 legacy 资产构造 BPE 模型配置（`vocab.json` + `merges.txt`）。
pub fn load_legacy_bpe_model(vocab_json: &Path, merges_txt: &Path) -> Result<ModelConfig> {
    let vocab = load_vocab_json(vocab_json)?;
    let merges = load_merges_txt(merges_txt)?;
    let cfg = tok_models::byte_level_bpe_config(vocab, merges);
    cfg.validate()?;
    Ok(cfg)
}

/// 从目录里的 `vocab.json` / `merges.txt` 反装（legacy 路径）。
pub fn legacy_model_config(dir: &Path) -> Result<ModelConfig> {
    let vocab = dir.join("vocab.json");
    let merges = dir.join("merges.txt");
    if !vocab.is_file() {
        return Err(io_err(format!("缺 vocab.json: {}", vocab.display())));
    }
    if !merges.is_file() {
        return Err(io_err(format!("缺 merges.txt: {}", merges.display())));
    }
    load_legacy_bpe_model(&vocab, &merges)
}

// ------------------------------------------------------------------ 一致性比对

/// 逐 id 比对结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdDiff {
    pub text: String,
    pub left: Vec<u32>,
    pub right: Vec<u32>,
}

impl IdDiff {
    pub fn is_match(&self) -> bool {
        self.left == self.right
    }
}

/// 对一组文本逐条比对两个 tokenizer 的编码 id（CP-2 的实现侧辅助）。
pub fn diff_ids(
    left: &tok_pipeline::Tokenizer,
    right: &tok_pipeline::Tokenizer,
    texts: &[String],
) -> Result<Vec<IdDiff>> {
    let mut out = Vec::new();
    for t in texts {
        out.push(IdDiff {
            text: t.clone(),
            left: left.encode(t, true)?.ids,
            right: right.encode(t, true)?.ids,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tok_pipeline::Tokenizer;

    fn write_legacy(dir: &Path) {
        std::fs::write(dir.join("vocab.json"), r#"{"a":0,"b":1,"ab":2,"<unk>":3}"#).unwrap();
        std::fs::write(dir.join("merges.txt"), "#version: 0.2\na b\n").unwrap();
    }

    #[test]
    fn legacy_assets_load_into_bpe_config() {
        let dir = std::env::temp_dir().join(format!("tok_compat_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_legacy(&dir);

        let cfg = legacy_model_config(&dir).unwrap();
        match cfg {
            ModelConfig::Bpe { vocab, merges, .. } => {
                assert_eq!(vocab.len(), 4);
                assert_eq!(merges.len(), 1);
            }
            other => panic!("应为 BPE，实际 {:?}", other),
        }

        let inv = inventory(&dir).unwrap();
        assert!(!inv.is_loadable_fast(), "只有 legacy 资产时不算 fast 可加载");
        assert_eq!(inv.others.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_artifacts_detected_after_save_pretrained() {
        let dir = std::env::temp_dir().join(format!("tok_compat_hf_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let vocab: BTreeMap<String, u32> = [("a".to_string(), 0u32), ("b".to_string(), 1), ("<unk>".to_string(), 2)]
            .into_iter()
            .collect();
        let roles = TokenRoles { unk_token: Some("<unk>".to_string()), ..TokenRoles::default() };
        let cfg = tok_pipeline::byte_level_bpe(vocab, Vec::new(), roles);
        let tk = Tokenizer::from_config(&cfg).unwrap();
        let out = dir.join("hf");
        tk.save_pretrained(&out).unwrap();

        let inv = inventory(&out).unwrap();
        assert!(inv.is_complete(), "三件套应齐全: {:?}", inv);

        // 回读并比对自身（同一实现内自洽）
        let tk2 = tok_pipeline::load_tokenizer_json(&out.join(TOKENIZER_JSON)).unwrap();
        let diffs = diff_ids(&tk, &tk2, &["ab".to_string(), "a b".to_string()]).unwrap();
        assert!(diffs.iter().all(|d| d.is_match()), "回读后编码应逐 id 一致: {:?}", diffs);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
