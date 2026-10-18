//! L3 Unigram 模型与训练器（V0.4 骨架版：Viterbi 最优切分 + 字符/片段打分）。
//!
//! 说明：完整 Unigram 训练（EM + 剪枝，HF `UnigramTrainer`）列入演进路线；
//! 本版本实现的是**推理侧完整**的 Viterbi 切分与分数导出，训练侧用频次-长度启发式打分，
//! 保证 `to_hf_json` / `apply_artifact` 可用、可跨实现加载。

use ahash::AHashMap;
use tok_core::{
    AddedToken, Model, Result, Token, TokenizerError, Trainer, TrainerArtifact, TrainerParams, Vocabulary,
    layer,
};

/// Unigram 模型：词表为 `(piece, log_prob)`，切分走 Viterbi（最大对数概率路径）。
pub struct UnigramModel {
    vocab: Vocabulary,
    scores: AHashMap<String, f64>,
    /// `id → (piece, score)`，导出与解码使用
    entries: Vec<(String, f64)>,
    pub unk_token: String,
}

impl UnigramModel {
    pub fn new(unk_token: impl Into<String>) -> Self {
        Self {
            vocab: Vocabulary::new(),
            scores: AHashMap::new(),
            entries: Vec::new(),
            unk_token: unk_token.into(),
        }
    }

    /// 从 `(piece, score)` 列表构造（ID 按给定顺序分配）。
    pub fn from_pieces(pieces: Vec<(String, f64)>, unk_token: impl Into<String>) -> Result<Self> {
        let mut m = Self::new(unk_token);
        for (i, (p, s)) in pieces.into_iter().enumerate() {
            m.push_piece(p, s, i as u32)?;
        }
        Ok(m)
    }

    fn push_piece(&mut self, piece: String, score: f64, id: u32) -> Result<()> {
        self.vocab.insert(&piece, id)?;
        self.scores.insert(piece.clone(), score);
        let idx = id as usize;
        if self.entries.len() <= idx {
            self.entries.resize(idx + 1, (String::new(), f64::NEG_INFINITY));
        }
        self.entries[idx] = (piece, score);
        Ok(())
    }

    pub fn score_of(&self, piece: &str) -> Option<f64> {
        self.scores.get(piece).copied()
    }

    /// HF `model` 节：Unigram 的 `vocab` 是 `[[piece, score], ...]`（按 ID 升序）。
    pub fn to_config_json(&self) -> serde_json::Value {
        let vocab: Vec<serde_json::Value> = self
            .entries
            .iter()
            .filter(|(p, _)| !p.is_empty())
            .map(|(p, s)| serde_json::json!([p, s]))
            .collect();
        serde_json::json!({
            "type": "Unigram",
            "unk_id": self.unk_id(),
            "vocab": vocab,
            "byte_fallback": false,
        })
    }

    pub fn unk_id(&self) -> Option<u32> {
        self.vocab.token_to_id(&self.unk_token)
    }

    /// Viterbi 最优切分：返回 `(piece, byte_start, byte_end, score)` 序列。
    fn viterbi(&self, s: &str) -> Vec<(String, usize, usize, f64)> {
        let n = s.len();
        let mut best = vec![f64::NEG_INFINITY; n + 1];
        let mut back: Vec<Option<(usize, String)>> = vec![None; n + 1];
        best[0] = 0.0;
        let unk_penalty = -100.0;
        for i in 0..n {
            if best[i].is_infinite() && best[i] < 0.0 {
                continue;
            }
            // 按字符边界扩展
            let mut j = i;
            for c in s[i..].chars() {
                j += c.len_utf8();
                let piece = &s[i..j];
                let score = self.scores.get(piece).copied();
                let (cand_score, cand_piece) = match score {
                    Some(sc) => (best[i] + sc, piece.to_string()),
                    // 单字符兜底：未登录时允许按字符走 unk（V0.4 近似，Unigram 的 unk 语义见 README）
                    None if piece.chars().count() == 1 => {
                        (best[i] + unk_penalty, self.unk_token.clone())
                    }
                    None => continue,
                };
                if cand_score > best[j] {
                    best[j] = cand_score;
                    back[j] = Some((i, cand_piece));
                }
            }
        }
        let mut out: Vec<(String, usize, usize, f64)> = Vec::new();
        let mut cur = n;
        while cur > 0 {
            match back[cur].take() {
                Some((prev, piece)) => {
                    let sc = best[cur] - best[prev];
                    out.push((piece, prev, cur, sc));
                    cur = prev;
                }
                None => return Vec::new(), // 不可达（理论上每个单字符都有兜底）
            }
        }
        out.reverse();
        out
    }
}

impl Model for UnigramModel {
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(Vec::new());
        }
        let path = self.viterbi(sequence);
        if path.is_empty() {
            return match self.unk_id() {
                Some(id) => Ok(vec![Token { id, value: sequence.to_string(), offsets: (0, sequence.len()) }]),
                None => Err(TokenizerError::layer(
                    layer::MODEL,
                    format!("Unigram：'{}' 无法切分且缺少 unk_token", sequence),
                )),
            };
        }
        let mut out = Vec::with_capacity(path.len());
        for (piece, s, e, _) in path {
            let id = match self.vocab.token_to_id(&piece) {
                Some(id) => id,
                None => self.unk_id().ok_or_else(|| {
                    TokenizerError::layer(layer::MODEL, format!("Unigram：'{}' 未登录且缺少 unk_token", piece))
                })?,
            };
            out.push(Token { id, value: piece, offsets: (s, e) });
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
        let j = serde_json::to_string_pretty(&self.to_config_json())?;
        std::fs::write(dir.join("unigram.json"), j)?;
        Ok(vec!["unigram.json".to_string()])
    }

    fn get_trainer(&self, params: &TrainerParams) -> Option<Box<dyn Trainer>> {
        Some(Box::new(UnigramTrainer::new(params.clone(), self.unk_token.clone())))
    }

    fn apply_artifact(&mut self, artifact: &TrainerArtifact) -> Result<()> {
        match artifact {
            TrainerArtifact::Unigram { vocab } => {
                let mut m = UnigramModel::new(self.unk_token.clone());
                for (i, (p, s)) in vocab.iter().enumerate() {
                    m.push_piece(p.clone(), *s, i as u32)?;
                }
                *self = m;
                Ok(())
            }
            _ => Err(TokenizerError::layer(layer::MODEL, "artifact 不是 Unigram 类型")),
        }
    }
}

/// Unigram 训练器（V0.4：频次-长度启发式打分 + 词表裁剪）。
pub struct UnigramTrainer {
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub special_tokens: Vec<AddedToken>,
    unk_token: String,
    counts: AHashMap<String, u64>,
    total: u64,
}

impl UnigramTrainer {
    pub fn new(params: TrainerParams, unk_token: impl Into<String>) -> Self {
        Self {
            vocab_size: params.vocab_size,
            min_frequency: params.min_frequency,
            special_tokens: params
                .special_tokens
                .iter()
                .map(|s| AddedToken::new(s.clone(), true))
                .collect(),
            unk_token: unk_token.into(),
            counts: AHashMap::new(),
            total: 0,
        }
    }
}

impl Trainer for UnigramTrainer {
    fn feed_groups(&mut self, groups: &[Vec<String>]) -> Result<()> {
        for g in groups {
            for w in g {
                if w.is_empty() {
                    continue;
                }
                self.total += 1;
                // 统计所有长度 ≥ 1 的前缀片段（近似 EM 的候选集）
                let chars: Vec<(usize, char)> = w.char_indices().collect();
                for i in 0..chars.len() {
                    for j in (i + 1)..=chars.len() {
                        let piece = &w[chars[i].0..(if j < chars.len() { chars[j].0 } else { w.len() })];
                        *self.counts.entry(piece.to_string()).or_insert(0) += 1;
                    }
                }
                *self.counts.entry(w.clone()).or_insert(0) += 1;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<TrainerArtifact> {
        if self.total == 0 {
            return Err(TokenizerError::layer(layer::MODEL, "Unigram 训练：语料为空"));
        }
        let reserved = self.special_tokens.len();
        let budget = self.vocab_size.saturating_sub(reserved);
        // 打分：log(频次) - 0.1 * 字符数（长度惩罚，抑制超长片段）
        let mut cand: Vec<(String, f64, u64)> = self
            .counts
            .iter()
            .filter(|(_, c)| **c >= self.min_frequency)
            .map(|(p, c)| {
                let len = p.chars().count() as f64;
                let sc = (*c as f64).ln() - 0.1 * len;
                (p.clone(), sc, *c)
            })
            .collect();
        cand.sort_by(|a, b| b.2.cmp(&a.2).then(b.0.cmp(&a.0)));

        let mut vocab: Vec<(String, f64)> = Vec::new();
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for (p, sc, _) in cand.into_iter() {
            if vocab.len() >= budget {
                break;
            }
            if seen.insert(p.clone()) {
                vocab.push((p, sc));
            }
        }
        for t in &self.special_tokens {
            if seen.insert(t.content.clone()) {
                vocab.push((t.content.clone(), 0.0));
            }
        }
        if seen.insert(self.unk_token.clone()) {
            vocab.push((self.unk_token.clone(), 0.0));
        }
        Ok(TrainerArtifact::Unigram { vocab })
    }

    fn name(&self) -> &'static str {
        "UnigramTrainer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> UnigramModel {
        // "ab" 的分数高于 "a"+"b" → Viterbi 应选 "ab"
        UnigramModel::from_pieces(
            vec![
                ("<unk>".to_string(), 0.0),
                ("a".to_string(), -3.0),
                ("b".to_string(), -3.0),
                ("ab".to_string(), -1.0),
                ("c".to_string(), -3.0),
            ],
            "<unk>",
        )
        .unwrap()
    }

    #[test]
    fn viterbi_prefers_high_score_piece() {
        let m = model();
        let toks = m.tokenize("ab").unwrap();
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].value, "ab");
        assert_eq!(toks[0].offsets, (0, 2));
    }

    #[test]
    fn falls_back_to_single_chars() {
        let m = model();
        let toks = m.tokenize("ac").unwrap();
        assert_eq!(toks.len(), 2);
        assert_eq!(toks[0].value, "a");
        assert_eq!(toks[1].value, "c");
    }

    #[test]
    fn hf_json_uses_pair_list() {
        let m = model();
        let j = m.to_hf_json().unwrap();
        assert_eq!(j["type"], "Unigram");
        let arr = j["vocab"].as_array().unwrap();
        assert_eq!(arr[0][0], "<unk>");
        assert!(arr[0][1].is_number());
    }

    #[test]
    fn trainer_assigns_unk_and_specials() {
        let params = TrainerParams { vocab_size: 16, min_frequency: 1, ..Default::default() };
        let mut t = UnigramTrainer::new(params, "<unk>");
        t.feed_groups(&[vec!["ab".into(), "ab".into()]]).unwrap();
        let TrainerArtifact::Unigram { vocab } = t.finish().unwrap() else { panic!() };
        assert!(vocab.iter().any(|(p, _)| p == "<unk>"));
    }
}
