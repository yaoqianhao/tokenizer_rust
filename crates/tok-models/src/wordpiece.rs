//! L3 WordPiece 模型与训练器（贪心最长优先匹配）。
//!
//! 与 BPE 的差别：切分是"自左向右、最长优先"，非连续片段带 `##` 前缀；
//! 任何一个字符无法匹配时整词退化为 `unk`（HF 语义）。

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use tok_core::{
    layer, AddedToken, Model, Result, Token, TokenizerError, Trainer, TrainerArtifact,
    TrainerParams, Vocabulary,
};

/// WordPiece 模型。
pub struct WordPieceModel {
    vocab: Vocabulary,
    pub unk_token: String,
    pub continuing_subword_prefix: String,
    pub max_input_chars_per_word: usize,
}

impl WordPieceModel {
    pub fn new(vocab: Vocabulary, unk_token: impl Into<String>) -> Self {
        Self {
            vocab,
            unk_token: unk_token.into(),
            continuing_subword_prefix: "##".to_string(),
            max_input_chars_per_word: 100,
        }
    }

    pub fn unk_id(&self) -> Option<u32> {
        self.vocab.token_to_id(&self.unk_token)
    }

    /// HF `model` 节
    pub fn to_config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "WordPiece",
            "unk_token": self.unk_token,
            "continuing_subword_prefix": self.continuing_subword_prefix,
            "max_input_chars_per_word": self.max_input_chars_per_word,
            "vocab": self.vocab.to_id_map(),
        })
    }

    pub fn set_prefix(&mut self, prefix: impl Into<String>) {
        self.continuing_subword_prefix = prefix.into();
    }
}

impl Model for WordPieceModel {
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(Vec::new());
        }
        let chars: Vec<(usize, char)> = sequence.char_indices().collect();
        let unk = |s: &str| -> Result<Vec<Token>> {
            match self.unk_id() {
                // HF 语义：未登录词输出 unk_token 字面值（如 "[UNK]"），偏移覆盖整词
                Some(id) => Ok(vec![Token {
                    id,
                    value: self.unk_token.clone(),
                    offsets: (0, s.len()),
                }]),
                None => Err(TokenizerError::layer(
                    layer::MODEL,
                    format!("WordPiece：未登录词 '{}' 且缺少 unk_token", s),
                )),
            }
        };
        if chars.len() > self.max_input_chars_per_word {
            return unk(sequence);
        }

        let mut out: Vec<Token> = Vec::new();
        let mut start = 0usize;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found: Option<(String, usize)> = None;
            while start < end {
                let substr: String = chars[start..end].iter().map(|(_, c)| *c).collect();
                let candidate = if start > 0 {
                    format!("{}{}", self.continuing_subword_prefix, substr)
                } else {
                    substr.clone()
                };
                if self.vocab.contains(&candidate) {
                    found = Some((candidate, end));
                    break;
                }
                end -= 1;
            }
            match found {
                Some((token, next)) => {
                    let id = self.vocab.token_to_id(&token).unwrap();
                    let b_start = chars[start].0;
                    let b_end = if next < chars.len() {
                        chars[next].0
                    } else {
                        sequence.len()
                    };
                    // HF 语义：续接片段在 `tokens` 中原样保留 `##` 前缀（去前缀由解码层负责）
                    out.push(Token {
                        id,
                        value: token,
                        offsets: (b_start, b_end),
                    });
                    start = next;
                }
                None => return unk(sequence),
            }
        }
        Ok(out)
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
        std::fs::write(dir.join("vocab.txt"), self.vocab_txt())?;
        Ok(vec!["vocab.txt".to_string()])
    }

    fn get_trainer(&self, params: &TrainerParams) -> Option<Box<dyn Trainer>> {
        Some(Box::new(WordPieceTrainer::new(
            params.clone(),
            self.unk_token.clone(),
        )))
    }

    fn apply_artifact(&mut self, artifact: &TrainerArtifact) -> Result<()> {
        match artifact {
            TrainerArtifact::WordPiece { vocab } => {
                let mut v = Vocabulary::new();
                for (t, id) in vocab {
                    v.insert(t, *id)?;
                }
                self.vocab = v;
                Ok(())
            }
            _ => Err(TokenizerError::layer(
                layer::MODEL,
                "artifact 不是 WordPiece 类型",
            )),
        }
    }
}

impl WordPieceModel {
    /// 按 ID 顺序输出 vocab.txt（BERT 约定的纯文本词表）。
    pub fn vocab_txt(&self) -> String {
        let mut pairs = self.vocab.sorted_entries();
        pairs.sort_by_key(|(_, id)| *id);
        pairs.into_iter().map(|(t, _)| format!("{}\n", t)).collect()
    }
}

/// WordPiece 训练器（V0.4 简化版：整词优先 + 续接片段补齐）。
///
/// 说明：完整 WordPiece 训练是"迭代挑选使似然增益最大的片段"（Greedy 子词），
/// 本版本先用整词建立常规区，再按频次补齐 `##` 续接片段；完整算法列入演进路线。
pub struct WordPieceTrainer {
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub special_tokens: Vec<AddedToken>,
    pub max_token_length: Option<usize>,
    pub continuing_subword_prefix: String,
    unk_token: String,
    word_counts: AHashMap<String, u64>,
    fed: u64,
}

impl WordPieceTrainer {
    pub fn new(params: TrainerParams, unk_token: impl Into<String>) -> Self {
        Self {
            vocab_size: params.vocab_size,
            min_frequency: params.min_frequency,
            special_tokens: params
                .special_tokens
                .iter()
                .map(|s| AddedToken::new(s.clone(), true))
                .collect(),
            max_token_length: params.max_token_length,
            continuing_subword_prefix: params
                .continuing_subword_prefix
                .unwrap_or_else(|| "##".to_string()),
            unk_token: unk_token.into(),
            word_counts: AHashMap::new(),
            fed: 0,
        }
    }
}

impl Trainer for WordPieceTrainer {
    fn feed_groups(&mut self, groups: &[Vec<String>]) -> Result<()> {
        for g in groups {
            for w in g {
                if w.is_empty() {
                    continue;
                }
                *self.word_counts.entry(w.clone()).or_insert(0) += 1;
                self.fed += 1;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<TrainerArtifact> {
        if self.fed == 0 {
            return Err(TokenizerError::layer(
                layer::MODEL,
                "WordPiece 训练：语料为空",
            ));
        }
        // 1) 字母表（单字符）
        let mut chars: Vec<char> = self
            .word_counts
            .keys()
            .flat_map(|w| w.chars())
            .filter(|c| !c.is_whitespace())
            .collect();
        chars.sort();
        chars.dedup();

        // 2) 整词（频次降序，同频按字典序）
        let min_freq = self.min_frequency;
        let mut words: Vec<(String, u64)> = self
            .word_counts
            .iter()
            .filter(|(_, c)| **c >= min_freq)
            .map(|(w, c)| (w.clone(), *c))
            .collect();
        words.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let reserved = self.special_tokens.len();
        let budget = self.vocab_size.saturating_sub(reserved);

        let mut vocab: Vec<(String, u32)> = Vec::new();
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut next_id = 0u32;
        let mut push = |tok: String, vocab: &mut Vec<(String, u32)>, id: &mut u32| {
            if seen.insert(tok.clone()) {
                vocab.push((tok, *id));
                *id += 1;
                true
            } else {
                false
            }
        };
        for c in &chars {
            push(c.to_string(), &mut vocab, &mut next_id);
        }
        for (w, _) in &words {
            if vocab.len() >= budget {
                break;
            }
            if self
                .max_token_length
                .map(|m| w.chars().count() > m)
                .unwrap_or(false)
            {
                continue;
            }
            push(w.clone(), &mut vocab, &mut next_id);
        }
        // 3) 续接片段 `##suffix`
        let mut cont: AHashMap<String, u64> = AHashMap::new();
        for (w, c) in self.word_counts.iter() {
            let cs: Vec<char> = w.chars().collect();
            if cs.len() < 2 {
                continue;
            }
            for i in 1..cs.len() {
                let piece: String = cs[i..].iter().collect();
                let cand = format!("{}{}", self.continuing_subword_prefix, piece);
                *cont.entry(cand).or_insert(0) += *c;
            }
        }
        let mut cont_vec: Vec<(String, u64)> = cont.into_iter().collect();
        cont_vec.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (p, _) in cont_vec {
            if vocab.len() >= budget {
                break;
            }
            push(p, &mut vocab, &mut next_id);
        }
        // 4) special 追加末尾
        for t in &self.special_tokens {
            push(t.content.clone(), &mut vocab, &mut next_id);
        }
        // 5) unk 兜底：必须在词表内，否则推理会直接报错
        // 5) unk 兜底：必须在词表内，否则推理会直接报错（push 自带去重）
        let _ = push(self.unk_token.clone(), &mut vocab, &mut next_id);

        Ok(TrainerArtifact::WordPiece { vocab })
    }

    fn name(&self) -> &'static str {
        "WordPieceTrainer"
    }
}

/// HF `tokenizer_config.json` 里的 `WordPiece` 旧式字段（兼容用）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordPieceLegacyConfig {
    pub unk_token: String,
    pub continuing_subword_prefix: String,
    pub max_input_chars_per_word: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> WordPieceModel {
        let mut v = Vocabulary::new();
        for (t, id) in [
            ("[UNK]", 0u32),
            ("un", 1),
            ("##want", 2),
            ("##ed", 3),
            ("he", 4),
            ("##llo", 5),
        ] {
            v.insert(t, id).unwrap();
        }
        WordPieceModel::new(v, "[UNK]")
    }

    #[test]
    fn greedy_longest_match() {
        let m = model();
        let toks = m.tokenize("unwanted").unwrap();
        let vals: Vec<String> = toks.iter().map(|t| t.value.clone()).collect();
        assert_eq!(vals, vec!["un", "##want", "##ed"]);
        // 续接片段保留 ## 前缀（HF 语义），偏移仍连续覆盖
        assert_eq!(toks[0].offsets, (0, 2));
        assert_eq!(toks[1].offsets, (2, 6));
        assert_eq!(toks[2].offsets, (6, 8));
    }

    #[test]
    fn second_piece_keeps_continuing_prefix() {
        let m = model();
        let toks = m.tokenize("hello").unwrap();
        let vals: Vec<String> = toks.iter().map(|t| t.value.clone()).collect();
        assert_eq!(vals, vec!["he", "##llo"]);
    }

    #[test]
    fn unmatched_becomes_unk() {
        let m = model();
        let toks = m.tokenize("zzz").unwrap();
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].value, "[UNK]");
        assert_eq!(toks[0].offsets, (0, 3));
    }

    #[test]
    fn trainer_keeps_unk_in_vocab() {
        let params = TrainerParams {
            vocab_size: 30,
            min_frequency: 1,
            ..Default::default()
        };
        let mut t = WordPieceTrainer::new(params, "[UNK]");
        t.feed_groups(&[vec!["hello".into(), "world".into()]])
            .unwrap();
        let TrainerArtifact::WordPiece { vocab } = t.finish().unwrap() else {
            panic!()
        };
        assert!(vocab.iter().any(|(s, _)| s == "[UNK]"));
        assert!(vocab.iter().any(|(s, _)| s == "##llo" || s == "##lo"));
    }
}
