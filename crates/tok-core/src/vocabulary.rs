//! 词表层：ID 布局、后加 token 登记与前置切分、词表与指纹。
//!
//! INV-3：`token_to_id` 与 `id_to_token` 必须双向可还原；重复 ID / 空洞在 `validate` 中暴露。

use std::collections::HashMap;

use ahash::AHashMap;

use crate::config::{AddedToken, AddedTokenWithId};
use crate::error::{Result, TokenizerError};

/// ID 布局（对应架构方案 §4.4）：
/// `[base 常规区][byte_fallback 常驻区][added 后加区][reserved 预留区]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IdLayout {
    pub base_start: u32,
    pub base_end: u32,
    pub byte_fallback_start: Option<u32>,
    pub byte_fallback_end: Option<u32>,
    pub added_start: u32,
    pub added_end: u32,
    pub reserved_start: u32,
    pub reserved_end: u32,
}

/// 角色 token 名称（用于从 `TokenizerConfig` 中取回关键角色的字面量）。
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TokenName {
    Unk,
    Bos,
    Eos,
    Pad,
    Cls,
    Sep,
    Mask,
    Other(String),
}

/// 前置切分单元：要么是普通正文（`id = None`），要么是命中的后加 token。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitUnit {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub id: Option<u32>,
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// 后加 token 的运行时视图：负责"最长左匹配"式前置切分（S1）。
#[derive(Debug, Clone, Default)]
pub struct AddedVocabulary {
    tokens: Vec<(AddedToken, u32)>,
}

impl AddedVocabulary {
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    pub fn add(&mut self, token: AddedToken, id: u32) {
        self.tokens.push((token, id));
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn tokens(&self) -> &[(AddedToken, u32)] {
        &self.tokens
    }

    pub fn token_to_id(&self, content: &str) -> Option<u32> {
        self.tokens
            .iter()
            .find(|(t, _)| t.content == content)
            .map(|(_, id)| *id)
    }

    pub fn with_ids(&self) -> Vec<AddedTokenWithId> {
        self.tokens
            .iter()
            .map(|(t, id)| AddedTokenWithId::from_token(t, *id))
            .collect()
    }

    /// S1：把输入切成 [正文, 后加token, 正文, ...]，保持字节下标连续。
    ///
    /// 说明：`normalized = true` 的后加 token 严格来说应在 L1 之后匹配
    /// （HF 的行为），V0.4 统一在 L1 之前匹配并置 token 为 special（默认 normalized=false，
    /// 与主流模型一致）；该差异已在 README "已知边界" 中登记。
    pub fn split(&self, text: &str) -> Vec<SplitUnit> {
        if self.tokens.is_empty() {
            return vec![SplitUnit {
                text: text.to_string(),
                start: 0,
                end: text.len(),
                id: None,
            }];
        }
        let mut pats: Vec<(&str, u32, &AddedToken)> = self
            .tokens
            .iter()
            .map(|(t, id)| (t.content.as_str(), *id, t))
            .collect();
        pats.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        let bytes = text.as_bytes();
        let mut out: Vec<SplitUnit> = Vec::new();
        let mut cursor = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            let mut matched: Option<(usize, u32, &AddedToken)> = None;
            for (content, id, tok) in pats.iter() {
                let cb = content.as_bytes();
                if bytes.len() >= i + cb.len() && &bytes[i..i + cb.len()] == cb {
                    if tok.single_word {
                        let prev_ok = i == 0
                            || !text[..i]
                                .chars()
                                .next_back()
                                .map(is_word_char)
                                .unwrap_or(false);
                        let next = i + cb.len();
                        let next_ok = next >= bytes.len()
                            || !text[next..]
                                .chars()
                                .next()
                                .map(is_word_char)
                                .unwrap_or(false);
                        if !(prev_ok && next_ok) {
                            continue;
                        }
                    }
                    matched = Some((cb.len(), *id, tok));
                    break;
                }
            }
            match matched {
                Some((len, id, tok)) => {
                    let mut start = i;
                    let mut end = i + len;
                    if tok.lstrip {
                        while start > cursor {
                            let c = text[..start].chars().next_back().unwrap();
                            if c.is_whitespace() {
                                start -= c.len_utf8();
                            } else {
                                break;
                            }
                        }
                    }
                    if tok.rstrip {
                        while end < bytes.len() {
                            let c = text[end..].chars().next().unwrap();
                            if c.is_whitespace() {
                                end += c.len_utf8();
                            } else {
                                break;
                            }
                        }
                    }
                    if start > cursor {
                        out.push(SplitUnit {
                            text: text[cursor..start].to_string(),
                            start: cursor,
                            end: start,
                            id: None,
                        });
                    }
                    out.push(SplitUnit {
                        text: text[start..end].to_string(),
                        start,
                        end,
                        id: Some(id),
                    });
                    cursor = end;
                    i = end;
                }
                None => {
                    let c = text[i..].chars().next().unwrap();
                    i += c.len_utf8();
                }
            }
        }
        if cursor < bytes.len() {
            out.push(SplitUnit {
                text: text[cursor..].to_string(),
                start: cursor,
                end: bytes.len(),
                id: None,
            });
        }
        out
    }
}

/// 词表：ID 双向映射 + 后加 token 视图 + 布局信息 + 指纹。
#[derive(Debug, Clone, Default)]
pub struct Vocabulary {
    token_to_id: AHashMap<String, u32>,
    id_to_token: Vec<Option<String>>,
    added: AddedVocabulary,
    layout: IdLayout,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self::default()
    }

    /// 插入常规 token；ID 冲突或 token 冲突直接报错（不覆盖、不静默）。
    pub fn insert(&mut self, token: &str, id: u32) -> Result<()> {
        if let Some(existing) = self.token_to_id.get(token) {
            if *existing != id {
                return Err(TokenizerError::vocab(format!(
                    "token '{}' 已存在（id={}），不能重复登记为 id={}",
                    token, existing, id
                )));
            }
            return Ok(());
        }
        let idx = id as usize;
        if self.id_to_token.len() <= idx {
            self.id_to_token.resize(idx + 1, None);
        }
        if let Some(other) = &self.id_to_token[idx] {
            if other != token {
                return Err(TokenizerError::vocab(format!(
                    "id {} 已被 '{}' 占用，无法再登记 '{}'",
                    id, other, token
                )));
            }
        }
        self.id_to_token[idx] = Some(token.to_string());
        self.token_to_id.insert(token.to_string(), id);
        Ok(())
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token.get(id as usize).and_then(|v| v.clone())
    }

    pub fn id_to_token_ref(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).and_then(|v| v.as_deref())
    }

    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    pub fn slots(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn max_id(&self) -> u32 {
        self.id_to_token
            .iter()
            .rposition(|v| v.is_some())
            .map(|i| i as u32)
            .unwrap_or(0)
    }

    pub fn added(&self) -> &AddedVocabulary {
        &self.added
    }

    pub fn layout(&self) -> IdLayout {
        self.layout
    }

    pub fn set_layout(&mut self, layout: IdLayout) {
        self.layout = layout;
    }

    /// 登记后加/特殊 token（同时写入 model 的 vocab，与 HF BpeTrainer 行为一致）。
    pub fn add_special(&mut self, token: AddedToken, id: u32) -> Result<()> {
        self.insert(&token.content, id)?;
        self.added.add(token, id);
        Ok(())
    }

    /// 把**已存在于 base 词表**的 token 登记进后加视图（不改 id、不重写 base 映射）。
    ///
    /// 场景：训练器已把 special token 写进 base 词表。HF 语义下这类 token 既在 `vocab`
    /// 也在 `added_tokens` 中；`added_tokens` 导出、`special_tokens_mask`、
    /// `skip_special_tokens` 都读后加视图，因此必须补登记，否则会被当成普通 token。
    pub fn register_added_view(&mut self, token: AddedToken, id: u32) -> Result<()> {
        if self.token_to_id.get(&token.content).copied() != Some(id) {
            return Err(TokenizerError::vocab(format!(
                "token `{}` 的既有 id 与登记 id {} 不一致",
                token.content, id
            )));
        }
        if self.added.token_to_id(&token.content).is_none() {
            self.added.add(token, id);
        }
        Ok(())
    }

    /// 把已在后加视图中的 token **就地**升级为 special（id 不变）。
    ///
    /// 场景（E6）：用户先用 `--special` 把一个 token 当普通后加 token 加进来，
    /// 随后又声明它为 `cls`/`sep` 等角色。HF 中角色 token 必然是 special，
    /// 因此需要在**不动 id** 的前提下翻标记，而不能重加一个同内容的 token。
    /// 返回是否命中既有后加 token。
    pub fn promote_added_special(&mut self, content: &str) -> bool {
        if let Some((token, _)) = self
            .added
            .tokens
            .iter_mut()
            .find(|(t, _)| t.content == content)
        {
            token.special = true;
            return true;
        }
        false
    }

    pub fn added_tokens_with_ids(&self) -> Vec<AddedTokenWithId> {
        self.added.with_ids()
    }

    /// 供 HF 导出使用的 token→id 映射（BTreeMap 保证输出稳定）。
    pub fn to_id_map(&self) -> std::collections::BTreeMap<String, u32> {
        self.token_to_id
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// 按 ID 升序的 (token, id) 列表（训练产物 / merges 导出使用）。
    pub fn sorted_entries(&self) -> Vec<(String, u32)> {
        let mut v: Vec<(String, u32)> = self
            .id_to_token
            .iter()
            .enumerate()
            .filter_map(|(i, t)| t.as_ref().map(|t| (t.clone(), i as u32)))
            .collect();
        v.sort_by_key(|(_, id)| *id);
        v
    }

    /// 词表指纹：token/ID 布局变化的敏感摘要（CP-3 使用）。
    pub fn fingerprint(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mix = |s: &str, hash: &mut u64| {
            for b in s.as_bytes() {
                *hash ^= *b as u64;
                *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for (t, id) in self.sorted_entries() {
            mix(&t, &mut hash);
            mix(&format!("#{}", id), &mut hash);
        }
        for (tk, id) in self.added.tokens() {
            mix(&tk.content, &mut hash);
            mix(
                &format!("@{}{}{}", id, tk.special, tk.normalized),
                &mut hash,
            );
        }
        mix(
            &format!(
                "{:?}/{:?}/{:?}/{}",
                self.layout.base_start,
                self.layout.byte_fallback_start,
                self.layout.added_start,
                self.layout.reserved_end
            ),
            &mut hash,
        );
        hash
    }

    /// INV-3 校验 + ID 空洞检查。
    pub fn validate(&self) -> Result<()> {
        if self.token_to_id.len() != self.id_to_token.iter().filter(|v| v.is_some()).count() {
            return Err(TokenizerError::vocab("token→id 与 id→token 表长度不一致"));
        }
        for (t, id) in self.token_to_id.iter() {
            match self.id_to_token_ref(*id) {
                Some(v) if v == t => {}
                Some(v) => {
                    return Err(TokenizerError::vocab(format!(
                        "INV-3 失败：'{}'→{}，但 {}→'{}'",
                        t, id, id, v
                    )))
                }
                None => {
                    return Err(TokenizerError::vocab(format!(
                        "INV-3 失败：'{}'→{}，但 id→token 表中该槽位为空",
                        t, id
                    )))
                }
            }
        }
        if let Some(end) = self.layout.added_end.checked_sub(self.layout.added_start) {
            let n = self.added.len() as u32;
            if end != 0 && n != end {
                return Err(TokenizerError::vocab(format!(
                    "added 区声明 {} 个 ID，实际登记 {} 个",
                    end, n
                )));
            }
        }
        Ok(())
    }

    /// 导出为普通 HashMap（调试/对比使用）。
    pub fn to_hash_map(&self) -> HashMap<String, u32> {
        self.token_to_id
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_conflicting_ids() {
        let mut v = Vocabulary::new();
        v.insert("a", 0).unwrap();
        assert!(v.insert("b", 0).is_err());
        assert!(v.insert("a", 1).is_err());
        v.insert("a", 0).unwrap();
        v.validate().unwrap();
    }

    #[test]
    fn special_token_split_is_positional() {
        let mut added = AddedVocabulary::new();
        added.add(AddedToken::new("<unk>", true), 3);
        let parts = added.split("hi<unk>there");
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].text, "hi");
        assert_eq!(parts[1].id, Some(3));
        assert_eq!((parts[1].start, parts[1].end), (2, 7));
        assert_eq!(parts[2].text, "there");
    }

    #[test]
    fn rstrip_consumes_following_whitespace() {
        let mut added = AddedVocabulary::new();
        let mut t = AddedToken::new("<s>", true);
        t.rstrip = true;
        added.add(t, 0);
        let parts = added.split("<s>  hello");
        assert_eq!(parts.len(), 2);
        assert_eq!((parts[0].start, parts[0].end), (0, 5));
        assert_eq!(parts[1].text, "hello");
    }
}
