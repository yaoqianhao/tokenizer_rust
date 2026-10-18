//! 六层 trait 契约（L1..L6）+ 训练器 + 漏斗基座 `Funnel`。
//!
//! 依赖红线：实现 crate 只依赖 `tok-core`，彼此互不依赖，也不反向依赖 `tok-pipeline`。

use crate::config::{AddedToken, TrainerArtifact, TrainerParams};
use crate::error::Result;
use crate::types::{Encoding, NormalizedString, PreToken, Token};
use crate::vocabulary::Vocabulary;

/// L1 规范化层：仅对 normalized 视图做变换，必须维护对齐表。
pub trait Normalizer: Send + Sync {
    fn normalize(&self, ns: NormalizedString) -> NormalizedString;
    /// 该层的配置（用于 HF 互操作导出）
    fn to_config_json(&self) -> Result<serde_json::Value>;
    fn name(&self) -> &'static str;
}

/// L2 预分词层：决定 PreToken 边界，必须保持 `values == normalized` 的切分拼接。
pub trait PreTokenizer: Send + Sync {
    fn pre_tokenize(&self, ns: &mut NormalizedString) -> Result<()>;
    /// 输出是否为"字节级映射串"（决定偏移换算走 PerByte 而非 PerChar）
    fn is_byte_level(&self) -> bool {
        false
    }
    fn to_config_json(&self) -> Result<serde_json::Value>;
    fn name(&self) -> &'static str;
}

/// L3 子词模型层：唯一存在算法分歧的层。
pub trait Model: Send + Sync {
    /// 单个预分词单元 → 子词序列（offsets 相对该单元的 value 字节区间）
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>>;
    fn get_vocab(&self) -> &Vocabulary;
    fn get_vocab_mut(&mut self) -> &mut Vocabulary;
    /// HF `tokenizer.json` 的 `model` 节
    fn to_hf_json(&self) -> Result<serde_json::Value>;
    /// 落盘模型侧文件（BPE → vocab.json + merges.txt；返回实际写出的文件名）
    fn save(&self, dir: &std::path::Path) -> Result<Vec<String>> {
        let _ = dir;
        Ok(Vec::new())
    }
    /// 该模型是否支持训练；不支持时返回 None（`get_trainer` 的空实现）
    fn get_trainer(&self, _params: &TrainerParams) -> Option<Box<dyn Trainer>> {
        None
    }
    /// 用训练产物重建（`apply_artifact` 后必须 permutation 不变：不重编号既有 ID）
    fn apply_artifact(&mut self, artifact: &TrainerArtifact) -> Result<()>;
    /// 保留区登记（byte-fallback / unk / special 常驻，防剪枝丢失）
    fn reserve_tokens(&mut self, tokens: &[AddedToken]) -> Result<()> {
        let mut next = self.get_vocab().max_id() + 1;
        for t in tokens {
            // 已在词表中：不重编号，但必须保证后加视图登记一致（导出 added_tokens /
            // special_tokens_mask / skip_special_tokens 均依赖该视图）。
            if let Some(existing) = self.get_vocab().token_to_id(&t.content) {
                if self.get_vocab().added().token_to_id(&t.content).is_none() {
                    self.get_vocab_mut().register_added_view(t.clone(), existing)?;
                }
                continue;
            }
            while self.get_vocab().id_to_token(next).is_some() {
                next += 1;
            }
            self.get_vocab_mut().add_special(t.clone(), next)?;
            next += 1;
        }
        Ok(())
    }
}

/// L5 后处理层：模板装配（type_ids / special_tokens_mask / sequence_ranges）。
pub trait PostProcessor: Send + Sync {
    fn process(
        &self,
        encoding: Encoding,
        pair: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding>;
    /// 模板会新增的 token 数（截断预算计算用）
    fn added_tokens(&self, is_pair: bool) -> usize;
    fn to_config_json(&self) -> Result<serde_json::Value>;
    fn name(&self) -> &'static str;
}

/// L6 解码层：`decode_chain` 的 stride 语义（tokens 保留 `##` / `Ġ` 等标记）。
pub trait Decoder: Send + Sync {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>>;
    fn to_config_json(&self) -> Result<serde_json::Value>;
    fn name(&self) -> &'static str;
}

/// 训练器：消费语料迭代器，产出 `TrainerArtifact`。
pub trait Trainer: Send + Sync {
    /// 语料分组：外层为"行"分组边界（对应 HF `feed` 的行语义），内层为预处理后的单元。
    fn feed_groups(&mut self, groups: &[Vec<String>]) -> Result<()>;
    fn finish(&mut self) -> Result<TrainerArtifact>;
    fn name(&self) -> &'static str;
}

/// 漏斗基座：所有层共享的"输入 → 输出"接口，便于统一编排与测试。
pub trait Funnel<In, Out>: Send + Sync {
    fn run(&self, input: In) -> Result<Out>;
}

impl<In, Out, T: Fn(In) -> Result<Out> + Send + Sync> Funnel<In, Out> for T {
    fn run(&self, input: In) -> Result<Out> {
        self(input)
    }
}

/// 便捷别名：预分词单元 → 子词序列的漏斗
pub type ModelFunnel = dyn Funnel<PreToken, Result<Vec<Token>>>;
