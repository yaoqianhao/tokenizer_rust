//! L3 WordLevel 模型（整词查表 + unk 兜底）与训练器。

use ahash::AHashMap;
use tok_core::{
    AddedToken, Model, Result, Token, TokenizerError, Trainer, TrainerArtifact, TrainerParams, Vocabulary,
    layer,
};

/// WordLevel 模型：不做子词切分，直接查表。
pub struct WordLevelModel {
    vocab: Vocabulary,
    pub unk_token: String,
    pub max_input_chars_per_word: usize,
}

impl WordLevelModel {
    pub fn new(vocab: Vocabulary, unk_token: impl Into<String>) -> Self {
        Self { vocab, unk_token: unk_token.into(), max_input_chars_per_word: 100 }
    }

    pub fn to_config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "WordLevel",
            "unk_token": self.unk_token,
            "vocab": self.vocab.to_id_map(),
        })
    }
}

impl Model for WordLevelModel {
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(id) = self.vocab.token_to_id(sequence) {
            return Ok(vec![Token { id, value: sequence.to_string(), offsets: (0, sequence.len()) }]);
        }
        match self.vocab.token_to_id(&self.unk_token) {
            // HF 语义：未登录词的 token 文本即 unk_token 本身（offsets 仍覆盖原片段）
            Some(id) => Ok(vec![Token { id, value: self.unk_token.clone(), offsets: (0, sequence.len()) }]),
            None => Err(TokenizerError::layer(
                layer::MODEL,
                format!("WordLevel：'{}' 未登录且缺少 unk_token", sequence),
            )),
        }
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
        let mut entries = self.vocab.sorted_entries();
        entries.sort_by_key(|(_, id)| *id);
        let txt: String = entries.into_iter().map(|(t, _)| format!("{}\n", t)).collect();
        std::fs::write(dir.join("vocab.txt"), txt)?;
        Ok(vec!["vocab.txt".to_string()])
    }

    fn get_trainer(&self, params: &TrainerParams) -> Option<Box<dyn Trainer>> {
        Some(Box::new(WordLevelTrainer::new(params.clone(), self.unk_token.clone())))
    }

    fn apply_artifact(&mut self, artifact: &TrainerArtifact) -> Result<()> {
        match artifact {
            TrainerArtifact::WordLevel { vocab } => {
                let mut v = Vocabulary::new();
                for (t, id) in vocab {
                    v.insert(t, *id)?;
                }
                self.vocab = v;
                Ok(())
            }
            _ => Err(TokenizerError::layer(layer::MODEL, "artifact 不是 WordLevel 类型")),
        }
    }
}

/// WordLevel 训练器：按频次降序登记整词，special 追加在末尾。
pub struct WordLevelTrainer {
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub special_tokens: Vec<AddedToken>,
    pub max_token_length: Option<usize>,
    unk_token: String,
    word_counts: AHashMap<String, u64>,
    fed: u64,
}

impl WordLevelTrainer {
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
            unk_token: unk_token.into(),
            word_counts: AHashMap::new(),
            fed: 0,
        }
    }
}

impl Trainer for WordLevelTrainer {
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
            return Err(TokenizerError::layer(layer::MODEL, "WordLevel 训练：语料为空"));
        }
        let reserved = self.special_tokens.len();
        let budget = self.vocab_size.saturating_sub(reserved);
        let mut entries: Vec<(String, u64)> = self
            .word_counts
            .iter()
            .filter(|(_, c)| **c >= self.min_frequency)
            .map(|(w, c)| (w.clone(), *c))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let mut vocab: Vec<(String, u32)> = Vec::new();
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut next_id = 0u32;
        for (w, _) in &entries {
            if vocab.len() >= budget {
                break;
            }
            if self.max_token_length.map(|m| w.chars().count() > m).unwrap_or(false) {
                continue;
            }
            if seen.insert(w.clone()) {
                vocab.push((w.clone(), next_id));
                next_id += 1;
            }
        }
        for t in &self.special_tokens {
            if seen.insert(t.content.clone()) {
                vocab.push((t.content.clone(), next_id));
                next_id += 1;
            }
        }
        if seen.insert(self.unk_token.clone()) {
            vocab.push((self.unk_token.clone(), next_id));
        }
        Ok(TrainerArtifact::WordLevel { vocab })
    }

    fn name(&self) -> &'static str {
        "WordLevelTrainer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_and_unk() {
        let mut v = Vocabulary::new();
        v.insert("hello", 0).unwrap();
        v.insert("[UNK]", 1).unwrap();
        let m = WordLevelModel::new(v, "[UNK]");
        assert_eq!(m.tokenize("hello").unwrap()[0].id, 0);
        assert_eq!(m.tokenize("nope").unwrap()[0].id, 1);
        assert_eq!(m.tokenize("nope").unwrap()[0].offsets, (0, 4));
        // HF 语义：未登录词的 token 文本为 unk_token 本身，而非原始词形
        assert_eq!(m.tokenize("nope").unwrap()[0].value, "[UNK]");
    }

    #[test]
    fn trainer_is_frequency_ordered() {
        let params = TrainerParams { vocab_size: 8, min_frequency: 1, ..Default::default() };
        let mut t = WordLevelTrainer::new(params, "[UNK]");
        t.feed_groups(&[vec!["b".into(), "b".into(), "a".into()]]).unwrap();
        let TrainerArtifact::WordLevel { vocab } = t.finish().unwrap() else { panic!() };
        assert_eq!(vocab[0].0, "b");
        assert_eq!(vocab[1].0, "a");
    }
}
