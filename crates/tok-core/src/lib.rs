//! # tok-core
//!
//! Tokenizer 流水线的公共契约层（L0 门面之下的核心）。
//!
//! 本 crate 只包含：数据结构（`NormalizedString` / `PreToken` / `Token` / `Encoding`）、
//! 六层 trait 契约（L1..L6）、统一错误类型、共享配置与词表。
//!
//! 依赖红线（架构方案 §3.2）：本 crate **不依赖**工作区内任何其它 crate；
//! 六个实现 crate 仅依赖本 crate；`tok-pipeline` 负责装配。

pub mod byte_level;
pub mod config;
pub mod error;
pub mod traits;
pub mod types;
pub mod vocabulary;

pub use byte_level::{
    byte_alphabet, byte_level_decode, byte_level_encode, byte_to_char, char_to_byte,
    GPT2_BYTE_LEVEL_PATTERN,
};
pub use config::{
    AddedToken, AddedTokenWithId, ConfigVersion, PaddingDirection, PaddingParams, PaddingStrategy,
    PatternSpec, RegexSpec, TrainerArtifact, TrainerParams, TruncationDirection, TruncationParams,
    TruncationStrategy,
};
pub use error::{Result, TokenizerError};
pub use traits::{Decoder, Funnel, Model, ModelFunnel, Normalizer, PostProcessor, PreTokenizer, Trainer};
pub use types::{Alignment, Encoding, NormalizedString, Offset, PreToken, SplitDelimiterBehavior, Token};
pub use vocabulary::{AddedVocabulary, IdLayout, SplitUnit, TokenName, Vocabulary};

/// 流水线层号常量，用于错误定位（0 = 门面层）。
pub mod layer {
    pub const FACADE: u8 = 0;
    pub const NORMALIZER: u8 = 1;
    pub const PRE_TOKENIZER: u8 = 2;
    pub const MODEL: u8 = 3;
    pub const VOCABULARY: u8 = 4;
    pub const POST_PROCESSOR: u8 = 5;
    pub const DECODER: u8 = 6;
}

/// crate 版本，写入 HF `tokenizer.json` 的元信息。
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
