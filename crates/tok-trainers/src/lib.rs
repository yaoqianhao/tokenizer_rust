//! # tok-trainers
//!
//! 训练编排层：语料读取（`CorpusReader`）+ 训练驱动（`train_model`）。
//!
//! 依赖红线：本 crate **只依赖 `tok-core`**，通过 `Model` / `Trainer` / `Normalizer` /
//! `PreTokenizer` 四个 trait 与被训练对象交互，不感知具体算法实现。
//! 因此"先把语料喂给 L1+L2 再进 L3"的预处理链也在这里以 trait 对象形式完成。

pub mod corpus;
pub mod driver;

pub use corpus::{CorpusData, CorpusReader};
pub use driver::{TrainOptions, prepare_words, train_model};

use serde::{Deserialize, Serialize};

/// 训练结果报告：同时作为 CLI 输出与测试断言依据。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerReport {
    /// 模型类型（BPE / WordPiece / WordLevel / Unigram）
    pub model_type: String,
    /// 训练完成后词表大小
    pub vocab_size: u32,
    /// merge 数（仅 BPE 非零）
    pub merges: usize,
    /// 参与训练的语料行数
    pub lines: usize,
    /// 经 L1+L2 后送进 L3 的预分词单元数
    pub words: usize,
    /// 词表中新追加的特殊 token 数（训练器已含则不计）
    pub specials_added: usize,
    /// 对齐退化的行数（>0 表示存在近似偏移，工程上需关注）
    pub degraded_lines: usize,
    /// 抗退化断言：训练后词表必须包含 unk / byte-fallback 常驻区
    pub byte_fallback_reserved: bool,
    /// 耗时（毫秒）
    pub elapsed_ms: u128,
}

impl TrainerReport {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    pub fn summary_line(&self) -> String {
        format!(
            "{} | vocab={} merges={} | lines={} words={} | specials_added={} degraded_lines={} | {} ms",
            self.model_type,
            self.vocab_size,
            self.merges,
            self.lines,
            self.words,
            self.specials_added,
            self.degraded_lines,
            self.elapsed_ms
        )
    }
}
