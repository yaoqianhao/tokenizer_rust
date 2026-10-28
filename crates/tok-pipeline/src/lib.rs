//! # tok-pipeline
//!
//! **L0 门面与编排层**：把 L1 normalizer / L2 pre_tokenizer / L3 model / L4 vocabulary /
//! L5 post_processor / L6 decoder 装配成一条可跑的流水线，并负责：
//!
//! * 端到端 `encode` / `decode`（含后加 token 前置切分、截断、填充、pair 模板）；
//! * 训练编排（`Tokenizer::train`，委托 `tok-trainers` 的 driver）；
//! * HuggingFace fast tokenizer 落盘（`tokenizer.json` / `tokenizer_config.json` /
//!   `special_tokens_map.json`）。
//!
//! 依赖红线：本 crate 依赖六个实现 crate；**实现 crate 之间互不依赖、且不得反向依赖本 crate**。

pub mod hf;
pub mod tokenizer;

#[cfg(test)]
mod tests;

pub use hf::{
    added_tokens_decoder, special_tokens_map, tokenizer_config, validate_tokenizer_json,
    write_json, HfTokenizerJson, TokenRoles, HF_SCHEMA_VERSION, SPECIAL_TOKENS_MAP_JSON,
    TOKENIZER_CONFIG_JSON, TOKENIZER_JSON,
};
pub use tokenizer::{
    bert_like, byte_level_bpe, load_tokenizer_json, specials, PipelineConfig, SavedFiles, Tokenizer,
};

// 便捷再导出：上层（CLI / 外部工具）无需逐个依赖六层实现 crate；
// 注意这是**单向**便利，不构成实现 crate 之间的依赖。
pub use tok_core as core;
pub use tok_decoders as decoders;
pub use tok_models as models;
pub use tok_normalizers as normalizers;
pub use tok_post as post;
pub use tok_pretok as pretok;
pub use tok_trainers as trainers;

/// 本层在六层流水线中的层号（L0 = 门面）。
pub const LAYER: u8 = tok_core::layer::FACADE;
