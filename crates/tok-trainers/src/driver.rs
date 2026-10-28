//! 训练驱动：语料 → L1 规范化 → L2 预分词 → L3 训练器 → 产物回填模型。

use std::time::Instant;

use tok_core::error::TokenizerError;
use tok_core::{
    layer, AddedToken, Model, NormalizedString, Normalizer, PreTokenizer, Result, Trainer,
    TrainerArtifact, TrainerParams,
};

use crate::TrainerReport;

/// 训练选项：把"预处理链 + 训练参数"打包，便于 CLI 与测试共用。
pub struct TrainOptions<'a> {
    pub params: TrainerParams,
    /// L1：训练时对语料做的规范化（必须与推理时一致）
    pub normalizer: Option<&'a dyn Normalizer>,
    /// L2：训练时的预分词（必须与推理时一致）
    pub pre_tokenizer: Option<&'a dyn PreTokenizer>,
    /// 每批喂给训练器的行数（控制内存峰值）
    pub batch_lines: usize,
}

impl<'a> TrainOptions<'a> {
    pub fn new(params: TrainerParams) -> Self {
        Self {
            params,
            normalizer: None,
            pre_tokenizer: None,
            batch_lines: 10_000,
        }
    }

    pub fn with_normalizer(mut self, n: &'a dyn Normalizer) -> Self {
        self.normalizer = Some(n);
        self
    }

    pub fn with_pre_tokenizer(mut self, p: &'a dyn PreTokenizer) -> Self {
        self.pre_tokenizer = Some(p);
        self
    }

    pub fn with_batch_lines(mut self, n: usize) -> Self {
        self.batch_lines = n.max(1);
        self
    }
}

/// 单行预处理：L1 → L2 → 取预分词单元字面值。
/// 返回 `(words, degraded)`，`degraded = true` 表示该行发生了对齐退化（不静默）。
pub fn prepare_words(
    line: &str,
    normalizer: Option<&dyn Normalizer>,
    pre_tokenizer: Option<&dyn PreTokenizer>,
) -> Result<(Vec<String>, bool)> {
    let mut ns = NormalizedString::new(line);
    if let Some(n) = normalizer {
        ns = n.normalize(ns);
    }
    if let Some(p) = pre_tokenizer {
        p.pre_tokenize(&mut ns)?;
    }
    let degraded = ns.is_degraded();
    let words: Vec<String> = ns
        .pretokens()
        .into_iter()
        .map(|t| t.value)
        .filter(|v| !v.is_empty())
        .collect();
    Ok((words, degraded))
}

/// 训练主流程。
///
/// 1. 从模型取训练器（`Model::get_trainer`）；
/// 2. 分批喂入预处理后的语料（`Trainer::feed_groups`）；
/// 3. 收尾得到 `TrainerArtifact`，回填模型（`Model::apply_artifact`）；
/// 4. 补齐 `TrainerParams.special_tokens`（训练器已含则跳过），返回报告。
pub fn train_model(
    model: &mut dyn Model,
    lines: &[String],
    opts: &TrainOptions<'_>,
) -> Result<TrainerReport> {
    let started = Instant::now();
    let model_type = model_type_of(model);

    let mut trainer: Box<dyn Trainer> = model.get_trainer(&opts.params).ok_or_else(|| {
        TokenizerError::layer(layer::MODEL, format!("模型 {} 未提供训练器", model_type))
    })?;

    let mut words_total = 0usize;
    let mut degraded_lines = 0usize;
    let batch = opts.batch_lines.max(1);
    let mut buf: Vec<Vec<String>> = Vec::with_capacity(batch.min(lines.len()));

    for line in lines {
        let (words, degraded) = prepare_words(line, opts.normalizer, opts.pre_tokenizer)?;
        if degraded {
            degraded_lines += 1;
        }
        if words.is_empty() {
            continue;
        }
        words_total += words.len();
        buf.push(words);
        if buf.len() >= batch {
            trainer.feed_groups(&buf)?;
            buf.clear();
        }
    }
    if !buf.is_empty() {
        trainer.feed_groups(&buf)?;
    }

    let artifact = trainer.finish()?;
    let merges = match &artifact {
        TrainerArtifact::Bpe { merges, .. } => merges.len(),
        _ => 0,
    };
    model.apply_artifact(&artifact)?;

    // 补齐特殊 token（训练器通常已把 specials 放在词表末尾）
    let mut specials_added = 0usize;
    for s in &opts.params.special_tokens {
        if model.get_vocab().token_to_id(s).is_none() {
            let id = model.get_vocab().max_id() + 1;
            model
                .get_vocab_mut()
                .add_special(AddedToken::new(s.clone(), true), id)?;
            specials_added += 1;
        }
    }

    let byte_fallback_reserved = if opts.params.byte_fallback {
        (0..=255u32).all(|b| {
            model
                .get_vocab()
                .token_to_id(&format!("<0x{:02X}>", b))
                .is_some()
        })
    } else {
        true
    };

    Ok(TrainerReport {
        model_type: model_type_of(model),
        vocab_size: model.get_vocab().len() as u32,
        merges,
        lines: lines.len(),
        words: words_total,
        specials_added,
        degraded_lines,
        byte_fallback_reserved,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

/// 从模型自述的 HF JSON 中读取算法类型（不依赖具体实现 crate）。
fn model_type_of(model: &dyn Model) -> String {
    model
        .to_hf_json()
        .ok()
        .and_then(|v| {
            v.get("type")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "Unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tok_core::{TrainerArtifact, Vocabulary};

    /// 测试替身：固定返回一个 BPE artifact 的最小模型。
    struct StubModel {
        vocab: Vocabulary,
    }

    impl Model for StubModel {
        fn tokenize(&self, sequence: &str) -> Result<Vec<tok_core::Token>> {
            Ok(vec![tok_core::Token {
                id: 0,
                value: sequence.to_string(),
                offsets: (0, sequence.len()),
            }])
        }
        fn get_vocab(&self) -> &Vocabulary {
            &self.vocab
        }
        fn get_vocab_mut(&mut self) -> &mut Vocabulary {
            &mut self.vocab
        }
        fn to_hf_json(&self) -> Result<serde_json::Value> {
            Ok(serde_json::json!({"type": "BPE"}))
        }
        fn save(&self, _dir: &std::path::Path) -> Result<Vec<String>> {
            Ok(vec![])
        }
        fn get_trainer(&self, _params: &TrainerParams) -> Option<Box<dyn Trainer>> {
            Some(Box::new(StubTrainer))
        }
        fn apply_artifact(&mut self, artifact: &TrainerArtifact) -> Result<()> {
            if let TrainerArtifact::Bpe { vocab, .. } = artifact {
                self.vocab = Vocabulary::new();
                for (t, id) in vocab {
                    self.vocab.insert(t, *id)?;
                }
            }
            Ok(())
        }
    }

    struct StubTrainer;

    impl Trainer for StubTrainer {
        fn feed_groups(&mut self, groups: &[Vec<String>]) -> Result<()> {
            assert!(!groups.is_empty());
            Ok(())
        }
        fn finish(&mut self) -> Result<TrainerArtifact> {
            Ok(TrainerArtifact::Bpe {
                vocab: vec![("a".into(), 0), ("b".into(), 1)],
                merges: vec![("a".into(), "b".into())],
            })
        }
        fn name(&self) -> &'static str {
            "StubTrainer"
        }
    }

    #[test]
    fn train_driver_fills_vocab_and_specials() {
        let mut model = StubModel {
            vocab: Vocabulary::new(),
        };
        let params = TrainerParams {
            vocab_size: 16,
            min_frequency: 1,
            special_tokens: vec!["<unk>".into(), "<s>".into()],
            ..Default::default()
        };
        let opts = TrainOptions::new(params).with_batch_lines(1);
        let lines = vec!["hello".to_string(), "world".to_string()];
        let report = train_model(&mut model, &lines, &opts).unwrap();
        assert_eq!(report.merges, 1);
        assert_eq!(report.specials_added, 2);
        assert_eq!(model.get_vocab().token_to_id("<unk>"), Some(2));
        assert_eq!(model.get_vocab().token_to_id("<s>"), Some(3));
        assert!(report.byte_fallback_reserved);
    }

    #[test]
    fn prepare_words_without_layers_returns_identity() {
        let (w, degraded) = prepare_words("hello", None, None).unwrap();
        assert_eq!(w, vec!["hello".to_string()]);
        assert!(!degraded);
    }
}
