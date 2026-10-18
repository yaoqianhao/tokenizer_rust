//! L3 BPE 模型与训练器。
//!
//! 关键设计（架构方案 §4.3 / §4.4）：
//! - **字节级 BPE 不是独立算法**，而是"initial_alphabet = 256 个字节字符 + ByteLevel 预分词"的配置组合；
//! - ID 布局：`[排序字母表 0..A][字节保留区][merge 顺序][special 追加在末尾]`；
//! - 训练**不重编号已有 ID**（permutation 不变）：`apply_artifact` 只做整体替换；
//! - unk / byte-fallback / special 常驻，防剪枝丢失。

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use tok_core::{
    AddedToken, Encoding, Model, Result, Token, TokenizerError, Trainer, TrainerArtifact, TrainerParams,
    Vocabulary, layer,
};

/// HF `merges` 的两种写法：`["a","b"]` 与 `"a b"`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MergeSpec {
    Pair(Vec<String>),
    Text(String),
}

impl MergeSpec {
    pub fn to_pair(&self) -> Result<(String, String)> {
        match self {
            MergeSpec::Pair(v) if v.len() == 2 => Ok((v[0].clone(), v[1].clone())),
            MergeSpec::Pair(v) => Err(TokenizerError::config(format!(
                "merges 元素必须是二元组，实际 {} 元",
                v.len()
            ))),
            MergeSpec::Text(s) => {
                let mut it = s.splitn(2, ' ');
                match (it.next(), it.next()) {
                    (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => {
                        Ok((a.to_string(), b.to_string()))
                    }
                    _ => Err(TokenizerError::config(format!("无法解析 merges 元素: '{}'", s))),
                }
            }
        }
    }

    pub fn from_pair(a: &str, b: &str) -> Self {
        MergeSpec::Pair(vec![a.to_string(), b.to_string()])
    }
}

/// 训练/推理用的符号（含相对 sequence 的字节区间）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// BPE 模型。
pub struct BpeModel {
    vocab: Vocabulary,
    merges: Vec<(String, String)>,
    merge_ranks: AHashMap<(String, String), u32>,
    pub dropout: Option<f32>,
    pub unk_token: Option<String>,
    pub continuing_subword_prefix: Option<String>,
    pub end_of_word_suffix: Option<String>,
    pub fuse_unk: bool,
    pub byte_fallback: bool,
    pub ignore_merges: bool,
    rng_state: std::sync::atomic::AtomicU64,
}

impl BpeModel {
    pub fn new(vocab: Vocabulary, merges: Vec<(String, String)>) -> Self {
        let mut merge_ranks = AHashMap::new();
        for (i, (a, b)) in merges.iter().enumerate() {
            merge_ranks.entry((a.clone(), b.clone())).or_insert(i as u32);
        }
        Self {
            vocab,
            merges,
            merge_ranks,
            dropout: None,
            unk_token: None,
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            fuse_unk: false,
            byte_fallback: false,
            ignore_merges: false,
            rng_state: std::sync::atomic::AtomicU64::new(0x9E37_79B9_7F4A_7C15),
        }
    }

    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }

    pub fn unk_id(&self) -> Option<u32> {
        self.unk_token.as_ref().and_then(|t| self.vocab.token_to_id(t))
    }

    fn next_rand(&self) -> f64 {
        // xorshift64*：避免引入 rand 依赖，用于 dropout 采样
        let mut x = self.rng_state.load(std::sync::atomic::Ordering::Relaxed);
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state.store(x, std::sync::atomic::Ordering::Relaxed);
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }

    /// 对符号序列执行 BPE 合并（按 merge rank 贪心）。
    fn merge_symbols(&self, mut symbols: Vec<Symbol>) -> Vec<Symbol> {
        loop {
            if symbols.len() < 2 {
                break;
            }
            let mut best: Option<(u32, usize)> = None;
            for i in 0..symbols.len() - 1 {
                let key = (symbols[i].text.clone(), symbols[i + 1].text.clone());
                if let Some(rank) = self.merge_ranks.get(&key) {
                    if best.map_or(true, |(br, _)| *rank < br) {
                        best = Some((*rank, i));
                    }
                }
            }
            match best {
                None => break,
                Some((_, i)) => {
                    // dropout：以概率 p 跳过本次合并（HF 语义：训练期正则化，推理请置 None）
                    if let Some(p) = self.dropout {
                        if p > 0.0 && self.next_rand() < p as f64 {
                            // 跳过该 pair：临时屏蔽时直接退出，避免死循环
                            break;
                        }
                    }
                    let merged = Symbol {
                        text: format!("{}{}", symbols[i].text, symbols[i + 1].text),
                        start: symbols[i].start,
                        end: symbols[i + 1].end,
                    };
                    symbols.splice(i..i + 2, std::iter::once(merged));
                }
            }
        }
        symbols
    }

    /// 单个符号 → Token（含 byte-fallback 与 unk 处理）。
    fn symbol_to_tokens(&self, sym: &Symbol) -> Result<Vec<Token>> {
        if let Some(id) = self.vocab.token_to_id(&sym.text) {
            return Ok(vec![Token { id, value: sym.text.clone(), offsets: (sym.start, sym.end) }]);
        }
        if self.byte_fallback {
            let mut out = Vec::with_capacity(sym.text.len());
            let mut ok = true;
            for (k, b) in sym.text.as_bytes().iter().enumerate() {
                let name = format!("<0x{:02X}>", b);
                match self.vocab.token_to_id(&name) {
                    Some(id) => out.push(Token { id, value: name, offsets: (sym.start + k, sym.start + k + 1) }),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && !out.is_empty() {
                return Ok(out);
            }
        }
        match self.unk_id() {
            Some(id) => Ok(vec![Token { id, value: sym.text.clone(), offsets: (sym.start, sym.end) }]),
            None => Err(TokenizerError::layer(
                layer::MODEL,
                format!("token '{}' 未登录且未配置 unk_token / byte_fallback", sym.text),
            )),
        }
    }

    fn apply_affixes(&self, syms: &mut Vec<Symbol>) {
        if syms.is_empty() {
            return;
        }
        if let Some(p) = &self.continuing_subword_prefix {
            for s in syms.iter_mut().skip(1) {
                s.text = format!("{}{}", p, s.text);
            }
        }
        if let Some(suf) = &self.end_of_word_suffix {
            let last = syms.len() - 1;
            syms[last].text = format!("{}{}", syms[last].text, suf);
        }
    }

    fn fuse_unks(&self, tokens: Vec<Token>) -> Vec<Token> {
        if !self.fuse_unk {
            return tokens;
        }
        let Some(unk) = self.unk_id() else {
            return tokens;
        };
        let mut out: Vec<Token> = Vec::with_capacity(tokens.len());
        for t in tokens {
            if t.id == unk {
                if let Some(last) = out.last_mut() {
                    if last.id == unk {
                        last.end_merge(&t);
                        continue;
                    }
                }
            }
            out.push(t);
        }
        out
    }

    /// HF `model` 节
    pub fn to_config_json(&self) -> serde_json::Value {
        let merges: Vec<serde_json::Value> = self
            .merges
            .iter()
            .map(|(a, b)| serde_json::json!([a, b]))
            .collect();
        serde_json::json!({
            "type": "BPE",
            "dropout": self.dropout,
            "unk_token": self.unk_token,
            "continuing_subword_prefix": self.continuing_subword_prefix,
            "end_of_word_suffix": self.end_of_word_suffix,
            "fuse_unk": self.fuse_unk,
            "byte_fallback": self.byte_fallback,
            "ignore_merges": self.ignore_merges,
            "vocab": self.vocab.to_id_map(),
            "merges": merges,
        })
    }
}

/// Token 的偏移合并（fuse_unk 用）。
trait EndMerge {
    fn end_merge(&mut self, other: &Token);
}

impl EndMerge for Token {
    fn end_merge(&mut self, other: &Token) {
        self.offsets.1 = other.offsets.1;
        self.value.push_str(&other.value);
    }
}

impl Model for BpeModel {
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(Vec::new());
        }
        if self.ignore_merges && self.vocab.contains(sequence) {
            let id = self.vocab.token_to_id(sequence).unwrap();
            return Ok(vec![Token { id, value: sequence.to_string(), offsets: (0, sequence.len()) }]);
        }
        let mut syms: Vec<Symbol> = sequence
            .char_indices()
            .map(|(i, c)| Symbol { text: c.to_string(), start: i, end: i + c.len_utf8() })
            .collect();
        self.apply_affixes(&mut syms);
        let merged = self.merge_symbols(syms);
        let mut out = Vec::new();
        for s in &merged {
            out.extend(self.symbol_to_tokens(s)?);
        }
        Ok(self.fuse_unks(out))
    }

    fn get_vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    fn get_vocab_mut(&mut self) -> &mut Vocabulary {
        &mut self.vocab
    }

    fn to_hf_json(&self) -> Result<serde_json::Value> {
        Ok(self.to_config_json())
    }

    fn save(&self, dir: &std::path::Path) -> Result<Vec<String>> {
        std::fs::create_dir_all(dir)?;
        let vocab = serde_json::to_string_pretty(&self.vocab.to_id_map())?;
        std::fs::write(dir.join("vocab.json"), vocab)?;
        let merges: String = self
            .merges
            .iter()
            .map(|(a, b)| format!("{} {}\n", a, b))
            .collect();
        std::fs::write(dir.join("merges.txt"), merges)?;
        Ok(vec!["vocab.json".to_string(), "merges.txt".to_string()])
    }

    fn get_trainer(&self, params: &TrainerParams) -> Option<Box<dyn Trainer>> {
        let mut t = BpeTrainer::new(params.clone());
        t.byte_fallback = self.byte_fallback || params.byte_fallback;
        t.unk_token = self.unk_token.clone();
        t.continuing_subword_prefix = self.continuing_subword_prefix.clone().or(params.continuing_subword_prefix.clone());
        t.end_of_word_suffix = self.end_of_word_suffix.clone().or(params.end_of_word_suffix.clone());
        Some(Box::new(t))
    }

    fn apply_artifact(&mut self, artifact: &TrainerArtifact) -> Result<()> {
        match artifact {
            TrainerArtifact::Bpe { vocab, merges } => {
                let mut v = Vocabulary::new();
                for (t, id) in vocab {
                    v.insert(t, *id)?;
                }
                self.vocab = v;
                self.merges = merges.clone();
                self.merge_ranks.clear();
                for (i, (a, b)) in self.merges.iter().enumerate() {
                    self.merge_ranks.entry((a.clone(), b.clone())).or_insert(i as u32);
                }
                Ok(())
            }
            other => Err(TokenizerError::layer(
                layer::MODEL,
                format!("BpeModel 无法消费 artifact kind={:?}", std::mem::discriminant(other)),
            )),
        }
    }
}

/// BPE 训练器。
///
/// 训练流程（V0.4）：
/// 1. 统计词频（输入已是 L1/L2 处理后的预分词单元）；
/// 2. `limit_alphabet` 过滤低频字符（映射为 unk）；
/// 3. 迭代取全局最高频 pair（tie → 字典序最小，保证可复现）合并；
/// 4. 组装词表：字母表 → 字节保留区 → merges → special（末尾）。
pub struct BpeTrainer {
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub special_tokens: Vec<AddedToken>,
    pub initial_alphabet: Vec<char>,
    pub limit_alphabet: Option<usize>,
    pub max_token_length: Option<usize>,
    pub byte_fallback: bool,
    pub unk_token: Option<String>,
    pub continuing_subword_prefix: Option<String>,
    pub end_of_word_suffix: Option<String>,
    word_counts: AHashMap<Vec<String>, u64>,
    fed: u64,
}

impl BpeTrainer {
    pub fn new(params: TrainerParams) -> Self {
        let special_tokens = params
            .special_tokens
            .iter()
            .map(|s| AddedToken::new(s.clone(), true))
            .collect();
        Self {
            vocab_size: params.vocab_size,
            min_frequency: params.min_frequency,
            special_tokens,
            initial_alphabet: params.initial_alphabet,
            limit_alphabet: params.limit_alphabet,
            max_token_length: params.max_token_length,
            byte_fallback: params.byte_fallback,
            unk_token: None,
            continuing_subword_prefix: params.continuing_subword_prefix,
            end_of_word_suffix: params.end_of_word_suffix,
            word_counts: AHashMap::new(),
            fed: 0,
        }
    }

    pub fn word_to_symbols(&self, word: &str) -> Vec<String> {
        let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if syms.is_empty() {
            return syms;
        }
        if let Some(p) = &self.continuing_subword_prefix {
            for s in syms.iter_mut().skip(1) {
                *s = format!("{}{}", p, s);
            }
        }
        if let Some(suf) = &self.end_of_word_suffix {
            let last = syms.len() - 1;
            syms[last] = format!("{}{}", syms[last], suf);
        }
        if let Some(max) = self.max_token_length {
            if syms.len() > max {
                let unk = self.unk_token.clone().unwrap_or_else(|| "[UNK]".to_string());
                return vec![unk];
            }
        }
        syms
    }

    fn char_frequencies(&self) -> AHashMap<String, u64> {
        let mut freq: AHashMap<String, u64> = AHashMap::new();
        for (word, count) in self.word_counts.iter() {
            for sym in word {
                *freq.entry(sym.clone()).or_insert(0) += *count;
            }
        }
        for c in &self.initial_alphabet {
            freq.entry(c.to_string()).or_insert(0);
        }
        freq
    }

    /// 组装字母表：initial_alphabet ∪ 语料字符，按字符序排序（保证可复现）。
    fn build_alphabet(&self) -> (Vec<String>, std::collections::BTreeSet<String>) {
        let freq = self.char_frequencies();
        let mut alphabet: Vec<String> = freq.keys().cloned().collect();
        let mut dropped: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        if let Some(limit) = self.limit_alphabet {
            if alphabet.len() > limit {
                let mut items: Vec<(String, u64)> = freq.iter().map(|(k, v)| (k.clone(), *v)).collect();
                // 频次降序；频次相同按字符序，保证确定性
                items.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                let kept: std::collections::BTreeSet<String> =
                    items.into_iter().take(limit).map(|(k, _)| k).collect();
                for a in &alphabet {
                    if !kept.contains(a) {
                        dropped.insert(a.clone());
                    }
                }
                alphabet.retain(|a| kept.contains(a));
            }
        }
        alphabet.sort();
        (alphabet, dropped)
    }

}

impl Trainer for BpeTrainer {
    fn feed_groups(&mut self, groups: &[Vec<String>]) -> Result<()> {
        for group in groups {
            for word in group {
                if word.is_empty() {
                    continue;
                }
                let syms = self.word_to_symbols(word);
                if syms.is_empty() {
                    continue;
                }
                *self.word_counts.entry(syms).or_insert(0) += 1;
                self.fed += 1;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<TrainerArtifact> {
        if self.fed == 0 {
            return Err(TokenizerError::layer(layer::MODEL, "BPE 训练：语料为空，无法训练"));
        }

        // 1. limit_alphabet 后需要把被丢弃字符映射为 unk（防止训练/推理不一致）
        let (alphabet, dropped) = self.build_alphabet();
        if !dropped.is_empty() {
            let unk = self
                .unk_token
                .clone()
                .ok_or_else(|| TokenizerError::layer(layer::MODEL, "limit_alphabet 需要配套 unk_token"))?;
            let mut new_counts: AHashMap<Vec<String>, u64> = AHashMap::new();
            for (word, count) in self.word_counts.iter() {
                let mapped: Vec<String> = word
                    .iter()
                    .map(|s| {
                        if dropped.contains(s) {
                            unk.clone()
                        } else {
                            s.clone()
                        }
                    })
                    .collect();
                *new_counts.entry(mapped).or_insert(0) += *count;
            }
            self.word_counts = new_counts;
        }

        // 2. 逐个挑选最高频 pair
        let reserved = self.special_tokens.len()
            + if self.byte_fallback { 256 } else { 0 };
        let max_tokens = self.vocab_size.saturating_sub(reserved);
        let mut merges: Vec<(String, String)> = Vec::new();
        let mut merged_symbols: Vec<String> = Vec::new(); // merge 顺序对应的新 token
        let mut current_words: AHashMap<Vec<String>, u64> = self.word_counts.clone();
        let mut vocab_len = alphabet.len();

        while vocab_len < max_tokens {
            let counts = {
                let mut counts: AHashMap<(String, String), u64> = AHashMap::new();
                for (word, freq) in current_words.iter() {
                    for w in word.windows(2) {
                        *counts.entry((w[0].clone(), w[1].clone())).or_insert(0) += *freq;
                    }
                }
                counts
            };
            let best = counts
                .iter()
                .filter(|(_, c)| **c >= self.min_frequency)
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .map(|(k, v)| (k.clone(), *v));
            let Some(((a, b), _)) = best else { break };

            // 合并
            let merged = format!("{}{}", a, b);
            let mut next_words: AHashMap<Vec<String>, u64> = AHashMap::new();
            for (word, freq) in current_words.iter() {
                let mut out: Vec<String> = Vec::with_capacity(word.len());
                let mut i = 0usize;
                while i < word.len() {
                    if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                        out.push(merged.clone());
                        i += 2;
                    } else {
                        out.push(word[i].clone());
                        i += 1;
                    }
                }
                *next_words.entry(out).or_insert(0) += *freq;
            }
            current_words = next_words;
            merges.push((a, b));
            merged_symbols.push(merged);
            vocab_len += 1;
        }

        // 3. 组装词表
        let mut vocab: Vec<(String, u32)> = Vec::new();
        let mut next_id = 0u32;
        for ch in &alphabet {
            vocab.push((ch.clone(), next_id));
            next_id += 1;
        }
        if self.byte_fallback {
            for b in 0u8..=255u8 {
                vocab.push((format!("<0x{:02X}>", b), next_id));
                next_id += 1;
            }
        }
        for s in &merged_symbols {
            vocab.push((s.clone(), next_id));
            next_id += 1;
        }
        for t in &self.special_tokens {
            if vocab.iter().any(|(s, _)| s == &t.content) {
                continue;
            }
            vocab.push((t.content.clone(), next_id));
            next_id += 1;
        }

        Ok(TrainerArtifact::Bpe { vocab, merges })
    }

    fn name(&self) -> &'static str {
        "BpeTrainer"
    }
}

/// 由 artifact 组装可用的 BPE 模型（含 unk / affix / byte_fallback 配置）。
pub fn bpe_from_artifact(
    artifact: &TrainerArtifact,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    byte_fallback: bool,
) -> Result<BpeModel> {
    match artifact {
        TrainerArtifact::Bpe { vocab, merges } => {
            let mut v = Vocabulary::new();
            for (t, id) in vocab {
                v.insert(t, *id)?;
            }
            let mut m = BpeModel::new(v, merges.clone());
            m.unk_token = unk_token;
            m.continuing_subword_prefix = continuing_subword_prefix;
            m.end_of_word_suffix = end_of_word_suffix;
            m.byte_fallback = byte_fallback;
            Ok(m)
        }
        _ => Err(TokenizerError::layer(layer::MODEL, "artifact 不是 BPE 类型")),
    }
}

/// 便捷编码：把 BPE 结果打包成 Encoding（供单元测试与门面复用）。
pub fn encode_words(model: &BpeModel, words: &[String]) -> Result<Encoding> {
    let mut enc = Encoding::default();
    let mut cursor = 0usize;
    for w in words {
        let toks = model.tokenize(w)?;
        for t in toks {
            enc.ids.push(t.id);
            enc.type_ids.push(0);
            enc.tokens.push(t.value);
            enc.offsets.push((cursor + t.offsets.0, cursor + t.offsets.1));
            enc.attention_mask.push(1);
            let is_special = model
                .get_vocab()
                .added()
                .tokens()
                .iter()
                .any(|(tk, id)| *id == t.id && tk.special);
            enc.special_tokens_mask.push(if is_special { 1 } else { 0 });
        }
        cursor += w.len();
    }
    enc.validate()?;
    Ok(enc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn train(words: &[&str], vocab_size: usize) -> BpeModel {
        let params = TrainerParams { vocab_size, min_frequency: 2, ..Default::default() };
        let mut t = BpeTrainer::new(params);
        let group: Vec<String> = words.iter().map(|w| w.to_string()).collect();
        t.feed_groups(&[group]).unwrap();
        let art = t.finish().unwrap();
        let mut m = bpe_from_artifact(&art, Some("<unk>".into()), None, None, false).unwrap();
        // unk 追加到末尾（训练流程由 pipeline 的 special_tokens 完成，这里显式补）
        let next = m.get_vocab().max_id() + 1;
        m.get_vocab_mut().add_special(AddedToken::new("<unk>", true), next).unwrap();
        m.fuse_unk = true;
        m
    }

    #[test]
    fn merges_are_learned_in_frequency_order() {
        let m = train(&["low", "low", "lower", "lowest"], 20);
        // "l o w" 至少应学到 (l,o) 或 (o,w)
        assert!(!m.merges().is_empty());
        let toks = m.tokenize("low").unwrap();
        assert_eq!(toks.len(), 1, "高频词应合并为单 token：{:?}", toks);
        assert_eq!(toks[0].offsets, (0, 3));
    }

    #[test]
    fn unknown_falls_back_to_unk() {
        let m = train(&["aab", "aab"], 6);
        let toks = m.tokenize("zzz").unwrap();
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].value, "zzz");
        assert_eq!(toks[0].id, m.unk_id().unwrap());
    }

    #[test]
    fn byte_fallback_uses_byte_tokens() {
        let params = TrainerParams { vocab_size: 400, min_frequency: 1, byte_fallback: true, ..Default::default() };
        let mut t = BpeTrainer::new(params);
        t.feed_groups(&[vec!["ab".to_string()]]).unwrap();
        let art = t.finish().unwrap();
        let m = bpe_from_artifact(&art, Some("<unk>".into()), None, None, true).unwrap();
        // 字节保留区存在
        assert!(m.get_vocab().contains("<0x61>"));
        let toks = m.tokenize("z").unwrap();
        // 'z' 未被合并 → 走 byte fallback
        assert_eq!(toks.len(), 1);
        assert!(toks[0].value.starts_with("<0x"));
    }

    #[test]
    fn ignore_merges_short_circuits_whole_word() {
        let mut m = train(&["ab", "ab", "ab"], 10);
        m.ignore_merges = true;
        if m.get_vocab().contains("ab") {
            let toks = m.tokenize("ab").unwrap();
            assert_eq!(toks.len(), 1);
        }
    }

    #[test]
    fn vocab_layout_is_stable_and_deterministic() {
        let params = TrainerParams { vocab_size: 12, min_frequency: 1, ..Default::default() };
        let mut t1 = BpeTrainer::new(params.clone());
        let mut t2 = BpeTrainer::new(params);
        t1.feed_groups(&[vec!["abc".into(), "abd".into()]]).unwrap();
        t2.feed_groups(&[vec!["abc".into(), "abd".into()]]).unwrap();
        let a1 = t1.finish().unwrap();
        let a2 = t2.finish().unwrap();
        assert_eq!(a1, a2, "相同语料必须产出逐位相同的词表");
    }

    #[test]
    fn special_tokens_appended_at_end() {
        let params = TrainerParams {
            vocab_size: 10,
            min_frequency: 1,
            special_tokens: vec!["<unk>".into(), "<s>".into(), "</s>".into()],
            ..Default::default()
        };
        let mut t = BpeTrainer::new(params);
        t.feed_groups(&[vec!["abc".into()]]).unwrap();
        let art = t.finish().unwrap();
        let TrainerArtifact::Bpe { vocab, .. } = art else { panic!() };
        for s in ["<unk>", "<s>", "</s>"] {
            let id = vocab.iter().find(|(t, _)| t == s).unwrap().1;
            assert!(id as usize >= vocab.len() - 3, "special 必须落在末尾：{} -> {}", s, id);
        }
    }
}
