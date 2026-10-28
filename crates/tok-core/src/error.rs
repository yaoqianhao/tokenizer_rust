//! 统一错误类型（AD-5：禁止 panic 与静默降级）。

use std::fmt;

/// 流水线统一错误类型。错误携带层标识，便于定位失败发生在哪一层。
#[derive(Debug)]
pub enum TokenizerError {
    /// 某一层内部的失败：layer 取 1..=6，0 表示门面层
    Layer { layer: u8, message: String },
    /// 输入非法（如非法 UTF-8、空输入等）
    InvalidInput(String),
    /// 词表相关问题（重复 ID、ID 空洞、未知 token）
    Vocabulary(String),
    /// 缺失关键角色 token（如缺 unk/bos 且配置未提供）
    MissingSpecialToken(String),
    /// 配置问题（未知字段、版本不兼容、字段组合非法）
    Config(String),
    /// 正则构造或匹配失败
    Regex(String),
    /// 该能力尚未支持
    Unsupported(String),
    /// 组件本身不具备该能力（如模型不支持训练）
    NotSupported(String),
    /// IO 错误
    Io(std::io::Error),
    /// JSON 解析/序列化错误
    Json(serde_json::Error),
}

impl TokenizerError {
    pub fn layer(layer: u8, message: impl Into<String>) -> Self {
        TokenizerError::Layer {
            layer,
            message: message.into(),
        }
    }
    pub fn config(message: impl Into<String>) -> Self {
        TokenizerError::Config(message.into())
    }
    pub fn vocab(message: impl Into<String>) -> Self {
        TokenizerError::Vocabulary(message.into())
    }
    pub fn unsupported(message: impl Into<String>) -> Self {
        TokenizerError::Unsupported(message.into())
    }
    pub fn not_supported(message: impl Into<String>) -> Self {
        TokenizerError::NotSupported(message.into())
    }
    pub fn invalid_input(message: impl Into<String>) -> Self {
        TokenizerError::InvalidInput(message.into())
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Layer { layer, message } => write!(f, "[L{}] {}", layer, message),
            TokenizerError::InvalidInput(m) => write!(f, "invalid input: {}", m),
            TokenizerError::Vocabulary(m) => write!(f, "vocabulary error: {}", m),
            TokenizerError::MissingSpecialToken(m) => write!(f, "missing special token: {}", m),
            TokenizerError::Config(m) => write!(f, "config error: {}", m),
            TokenizerError::Regex(m) => write!(f, "regex error: {}", m),
            TokenizerError::Unsupported(m) => write!(f, "unsupported: {}", m),
            TokenizerError::NotSupported(m) => write!(f, "not supported: {}", m),
            TokenizerError::Io(e) => write!(f, "io error: {}", e),
            TokenizerError::Json(e) => write!(f, "json error: {}", e),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self {
        TokenizerError::Io(e)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(e: serde_json::Error) -> Self {
        TokenizerError::Json(e)
    }
}

pub type Result<T> = std::result::Result<T, TokenizerError>;
