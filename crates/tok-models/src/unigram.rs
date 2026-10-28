//! L3 Unigram 模型与训练器（SentencePiece 同源算法）。
//!
//! 推理侧与 HF `models::Unigram::encode_optimized` **逐 token 等价**：
//! - 按字符边界做 Viterbi，候选只来自词表中实际存在的 piece（前缀枚举上限 = 最长 piece 字节数）；
//! - 当前位置若无"单字符 piece"，则插入 unk 节点，`unk_score = min_score - K_UNK_PENALTY`；
//! - 回溯时 `fuse_unk`（与 HF 一致恒为 true）把连续 unk 段合并；
//! - `byte_fallback` 时未登录串拆成 `<0xXX>`，每个字节 token 共享整段 offset（HF 同语义）。
//!
//! 训练侧走 SentencePiece 的三段式：
//! 1. **种子词表**：对预分词单元枚举所有字符级子串，按频次取 `vocab_size × seed_multiplier` 个候选，
//!    并强制保留全部单字符（保证任意串可切分，EM 的 `Z` 不为 0）；
//! 2. **EM**：每轮用当前词表对每个词做 log 域 forward-backward，得每条边的期望使用次数，
//!    M 步用期望计数重估 `log P(piece)`；
//! 3. **剪枝**：按"移除该 piece 造成的期望似然损失"升序剪掉尾部（每轮 ≤ 25%），
//!    迭代到词表规模收敛；单字符 / special / 字节 token 为受保护区。

use ahash::{AHashMap, AHashSet};
use tok_core::{
    layer, AddedToken, Model, Result, Token, TokenizerError, Trainer, TrainerArtifact,
    TrainerParams, Vocabulary,
};

/// 未登录词相对最低分的额外惩罚（HF `K_UNK_PENALTY`）
const K_UNK_PENALTY: f64 = 10.0;
/// 候选 piece 的最大字符数（SentencePiece `max_piece_length` 默认 16）
const DEFAULT_MAX_PIECE_LENGTH: usize = 16;
/// 每轮剪枝保留比例（SentencePiece `shrinking_factor` 默认 0.75）
const DEFAULT_SHRINKING_FACTOR: f64 = 0.75;
/// 每个剪枝阶段内部的 EM 迭代次数（SentencePiece `n_sub_iterations` 默认 2）
const DEFAULT_N_SUB_ITERATIONS: usize = 2;
/// 种子词表相对目标词表的倍数
const SEED_MULTIPLIER: usize = 8;
/// 候选表规模上限（防止语料过大时内存失控）
const MAX_CANDIDATES: usize = 4_000_000;

fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

fn neg_inf() -> f64 {
    f64::NEG_INFINITY
}

// ==================================================================== 模型

/// Unigram 模型：词表是 `(piece, log_prob)`，切分取整句对数概率最大的路径。
pub struct UnigramModel {
    vocab: Vocabulary,
    /// `id → (piece, score)`，导出与解码使用（可能含空洞，导出时过滤）
    entries: Vec<(String, f64)>,
    scores: AHashMap<String, f64>,
    /// unk 的**文本**：训练产物回填时用它在新词表里重新定位 id
    unk_token: String,
    unk_id: Option<u32>,
    byte_fallback: bool,
    min_score: f64,
    /// 词表里最长 piece 的字节数：Viterbi 前缀枚举的硬上界（超出必无命中）
    max_piece_bytes: usize,
}

impl UnigramModel {
    fn empty(unk_token: impl Into<String>, byte_fallback: bool) -> Self {
        Self {
            vocab: Vocabulary::new(),
            entries: Vec::new(),
            scores: AHashMap::new(),
            unk_token: unk_token.into(),
            unk_id: None,
            byte_fallback,
            min_score: f64::INFINITY,
            max_piece_bytes: 4,
        }
    }

    /// 从 `(piece, score)` 列表构造；`unk_id` 指向词表中的 unk piece。
    pub fn from_pieces(
        pieces: Vec<(String, f64)>,
        unk_id: Option<u32>,
        byte_fallback: bool,
    ) -> Result<Self> {
        let unk_token = match unk_id {
            Some(id) => pieces
                .get(id as usize)
                .map(|(p, _)| p.clone())
                .ok_or_else(|| {
                    TokenizerError::layer(
                        layer::MODEL,
                        format!(
                            "Unigram: unk_id {} 超出词表范围（共 {} 项）",
                            id,
                            pieces.len()
                        ),
                    )
                })?,
            None => String::new(),
        };
        let mut m = Self::empty(unk_token, byte_fallback);
        for (i, (p, s)) in pieces.into_iter().enumerate() {
            m.push_piece(p, s, i as u32)?;
        }
        if let Some(id) = unk_id {
            if m.vocab.id_to_token(id).is_none() {
                return Err(TokenizerError::layer(
                    layer::MODEL,
                    "Unigram: unk_id 指向空槽位",
                ));
            }
            m.unk_id = Some(id);
        }
        m.seal();
        Ok(m)
    }

    /// 以文本形式指定 unk（HF `tokenizer.json` 里 unk 由 `unk_id` 表达，等价换算）。
    pub fn with_unk_text(
        pieces: Vec<(String, f64)>,
        unk_token: &str,
        byte_fallback: bool,
    ) -> Result<Self> {
        let unk_id = pieces
            .iter()
            .position(|(p, _)| p == unk_token)
            .map(|i| i as u32);
        Self::from_pieces(pieces, unk_id, byte_fallback)
    }

    fn push_piece(&mut self, piece: String, score: f64, id: u32) -> Result<()> {
        if piece.is_empty() {
            return Err(TokenizerError::layer(
                layer::MODEL,
                "Unigram: 不允许空 piece",
            ));
        }
        self.vocab.insert(&piece, id)?;
        self.scores.insert(piece.clone(), score);
        let idx = id as usize;
        if self.entries.len() <= idx {
            self.entries
                .resize(idx + 1, (String::new(), f64::NEG_INFINITY));
        }
        self.entries[idx] = (piece, score);
        Ok(())
    }

    /// 收尾：算 min_score 与最大 piece 字节数（Viterbi 的上界）。
    fn seal(&mut self) {
        self.min_score = self
            .entries
            .iter()
            .filter(|(p, _)| !p.is_empty())
            .map(|(_, s)| *s)
            .fold(f64::INFINITY, f64::min);
        if !self.min_score.is_finite() {
            self.min_score = 0.0;
        }
        self.max_piece_bytes = self
            .entries
            .iter()
            .filter(|(p, _)| !p.is_empty())
            .map(|(p, _)| p.len())
            .max()
            .unwrap_or(4)
            .max(4);
        self.unk_id = if self.unk_token.is_empty() {
            None
        } else {
            self.vocab.token_to_id(&self.unk_token)
        };
    }

    pub fn score_of(&self, piece: &str) -> Option<f64> {
        self.scores.get(piece).copied()
    }

    pub fn unk_id(&self) -> Option<u32> {
        self.unk_id
    }

    pub fn byte_fallback(&self) -> bool {
        self.byte_fallback
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
            "unk_id": self.unk_id,
            "vocab": vocab,
            "byte_fallback": self.byte_fallback,
        })
    }

    /// 核心切分：返回 piece 字面值序列（与 HF `Unigram::encode` 等价，含 fuse_unk）。
    pub fn encode_str(&self, sentence: &str) -> Result<Vec<String>> {
        if sentence.is_empty() {
            return Ok(Vec::new());
        }
        let unk_id = match self.unk_id {
            Some(id) => Some(id as usize),
            None => None,
        };
        let unk_score = self.min_score - K_UNK_PENALTY;
        let size = sentence.len();

        let mut best = vec![neg_inf(); size + 1];
        let mut starts_at: Vec<Option<usize>> = vec![None; size + 1];
        let mut node_id: Vec<u32> = vec![0; size + 1];
        best[0] = 0.0;

        let mut pos = 0usize;
        while pos < size {
            if !best[pos].is_finite() {
                // 理论上不可达（单字符兜底保证连通），保险起见跳过
                pos += sentence[pos..]
                    .chars()
                    .next()
                    .map(|c| c.len_utf8())
                    .unwrap_or(1);
                continue;
            }
            let base = best[pos];
            let mblen = sentence[pos..]
                .chars()
                .next()
                .map(|c| c.len_utf8())
                .unwrap_or(1);
            let mut has_single_node = false;

            for (off, c) in sentence[pos..].char_indices() {
                let end = pos + off + c.len_utf8();
                if end - pos > self.max_piece_bytes {
                    break;
                }
                let piece = &sentence[pos..end];
                if let Some(&score) = self.scores.get(piece) {
                    let cand = base + score;
                    if starts_at[end].is_none() || cand > best[end] {
                        best[end] = cand;
                        starts_at[end] = Some(pos);
                        node_id[end] = self.vocab.token_to_id(piece).unwrap_or(0);
                    }
                    if end - pos == mblen {
                        has_single_node = true;
                    }
                }
            }

            if !has_single_node {
                let end = pos + mblen;
                let cand = base + unk_score;
                if starts_at[end].is_none() || cand > best[end] {
                    best[end] = cand;
                    starts_at[end] = Some(pos);
                    node_id[end] = unk_id.map(|u| u as u32).ok_or_else(|| {
                        TokenizerError::layer(layer::MODEL, "Unigram: 存在未登录字符但缺少 unk_id")
                    })?;
                }
            }
            pos += mblen;
        }

        // 回溯 + fuse_unk（HF 恒为 true）
        let mut results: Vec<String> = Vec::new();
        let mut acc: Vec<String> = Vec::new();
        let mut ends = size;
        while ends > 0 {
            let start = starts_at[ends].ok_or_else(|| {
                TokenizerError::layer(
                    layer::MODEL,
                    format!("Unigram: '{}' 无可行切分路径", sentence),
                )
            })?;
            let is_unk = unk_id.is_some() && node_id[ends] as usize == unk_id.unwrap();
            if is_unk {
                acc.push(sentence[start..ends].to_string());
            } else {
                if !acc.is_empty() {
                    acc.reverse();
                    results.push(acc.concat());
                    acc.clear();
                }
                results.push(sentence[start..ends].to_string());
            }
            ends = start;
        }
        if !acc.is_empty() {
            acc.reverse();
            results.push(acc.concat());
        }
        results.reverse();
        Ok(results)
    }

    fn missing_unk_error(&self, piece: &str) -> TokenizerError {
        TokenizerError::layer(
            layer::MODEL,
            format!("Unigram: '{}' 不在词表中，且模型未配置 unk_token", piece),
        )
    }
}

impl Model for UnigramModel {
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        let pieces = self.encode_str(sequence)?;
        let mut offset = 0usize;
        let mut out: Vec<Token> = Vec::with_capacity(pieces.len());
        for piece in pieces {
            let len = piece.len();
            let offsets = (offset, offset + len);
            match self.vocab.token_to_id(&piece) {
                Some(id) => out.push(Token {
                    id,
                    value: piece,
                    offsets,
                }),
                None => {
                    // 与 HF 一致：先尝试 byte_fallback，再退 unk
                    if self.byte_fallback {
                        let bytes: Option<Vec<Token>> = piece
                            .bytes()
                            .map(|b| {
                                let tok = format!("<0x{:02X}>", b);
                                self.vocab.token_to_id(&tok).map(|id| Token {
                                    id,
                                    value: tok,
                                    offsets: (offset, offset + len),
                                })
                            })
                            .collect();
                        if let Some(byte_tokens) = bytes {
                            out.extend(byte_tokens);
                            offset += len;
                            continue;
                        }
                    }
                    let unk = self.unk_id.ok_or_else(|| self.missing_unk_error(&piece))?;
                    out.push(Token {
                        id: unk,
                        value: piece,
                        offsets,
                    });
                }
            }
            offset += len;
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
        Some(Box::new(UnigramTrainer::new(
            params.clone(),
            self.unk_token.clone(),
        )))
    }

    fn apply_artifact(&mut self, artifact: &TrainerArtifact) -> Result<()> {
        match artifact {
            TrainerArtifact::Unigram { vocab } => {
                let unk_token = if self.unk_token.is_empty() {
                    "<unk>".to_string()
                } else {
                    self.unk_token.clone()
                };
                let rebuilt =
                    UnigramModel::with_unk_text(vocab.clone(), &unk_token, self.byte_fallback)?;
                *self = rebuilt;
                Ok(())
            }
            _ => Err(TokenizerError::layer(
                layer::MODEL,
                "artifact 不是 Unigram 类型",
            )),
        }
    }
}

// ================================================================ 训练器

/// 一个预分词单元：字符边界表 + 频次（EM 内层循环反复使用，预先算好）
struct WordLattice {
    text: String,
    /// 字符边界（含 0 与 len），长度 = 字符数 + 1
    bounds: Vec<usize>,
    count: u64,
}

/// Unigram 训练器：子串频次种子 → EM → 剪枝，直到目标词表规模。
pub struct UnigramTrainer {
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub special_tokens: Vec<AddedToken>,
    pub unk_token: String,
    pub max_piece_length: usize,
    pub shrinking_factor: f64,
    pub n_sub_iterations: usize,
    pub byte_fallback: bool,
    /// 预分词单元频次
    counts: AHashMap<String, u64>,
    words: u64,
}

impl UnigramTrainer {
    pub fn new(params: TrainerParams, unk_token: impl Into<String>) -> Self {
        Self {
            vocab_size: params.vocab_size.max(1),
            min_frequency: params.min_frequency.max(1),
            special_tokens: params
                .special_tokens
                .iter()
                .map(|s| AddedToken::new(s.clone(), true))
                .collect(),
            unk_token: unk_token.into(),
            max_piece_length: params
                .max_piece_length
                .unwrap_or(DEFAULT_MAX_PIECE_LENGTH)
                .max(1),
            shrinking_factor: params
                .shrinking_factor
                .unwrap_or(DEFAULT_SHRINKING_FACTOR)
                .clamp(0.1, 0.999),
            n_sub_iterations: params
                .n_sub_iterations
                .unwrap_or(DEFAULT_N_SUB_ITERATIONS)
                .max(1),
            byte_fallback: params.byte_fallback,
            counts: AHashMap::new(),
            words: 0,
        }
    }

    /// special 列表（含 unk），保持声明顺序、去重。
    fn special_list(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut seen: AHashSet<String> = AHashSet::new();
        let unk = if self.unk_token.is_empty() {
            "<unk>".to_string()
        } else {
            self.unk_token.clone()
        };
        if seen.insert(unk.clone()) {
            out.push(unk);
        }
        for t in &self.special_tokens {
            if !t.content.is_empty() && seen.insert(t.content.clone()) {
                out.push(t.content.clone());
            }
        }
        out
    }

    /// 种子候选：所有预分词单元的字符级子串（长度 ≤ max_piece_length，整词不限）。
    fn seed_candidates(&self) -> AHashMap<String, u64> {
        let mut cand: AHashMap<String, u64> = AHashMap::new();
        for (w, c) in &self.counts {
            let bounds: Vec<usize> = w
                .char_indices()
                .map(|(i, _)| i)
                .chain(std::iter::once(w.len()))
                .collect();
            let nchars = bounds.len() - 1;
            for i in 0..nchars {
                for j in (i + 1)..=nchars {
                    let span = j - i;
                    if span > self.max_piece_length && span != nchars {
                        break;
                    }
                    let piece = &w[bounds[i]..bounds[j]];
                    *cand.entry(piece.to_string()).or_insert(0) += *c;
                }
            }
            if cand.len() > MAX_CANDIDATES {
                // 超限时先砍低频候选，保留高频与全部单字符
                let mut items: Vec<(String, u64)> = cand.drain().collect();
                items.sort_by(|a, b| b.1.cmp(&a.1));
                let keep = MAX_CANDIDATES / 2;
                for (p, n) in items.into_iter().take(keep) {
                    cand.insert(p, n);
                }
            }
        }
        cand
    }

    /// 初始 log 概率：`ln(count) - ln(total)`。
    fn init_log_probs(pieces: &[String], cand: &AHashMap<String, u64>) -> AHashMap<String, f64> {
        let total: f64 = pieces
            .iter()
            .map(|p| *cand.get(p).unwrap_or(&1) as f64)
            .sum::<f64>()
            .max(1.0);
        let ln_total = total.ln();
        pieces
            .iter()
            .map(|p| {
                (
                    p.clone(),
                    (*cand.get(p).unwrap_or(&1) as f64).ln() - ln_total,
                )
            })
            .collect()
    }

    /// 单轮 EM：返回新的 log 概率与期望计数。
    fn em_step(
        words: &[WordLattice],
        logp: &AHashMap<String, f64>,
        max_piece_length: usize,
    ) -> (AHashMap<String, f64>, AHashMap<String, f64>) {
        let mut expected: AHashMap<String, f64> = AHashMap::new();
        for w in words {
            let n = w.bounds.len() - 1;
            let mut alpha = vec![neg_inf(); n + 1];
            alpha[0] = 0.0;
            for i in 0..n {
                if !alpha[i].is_finite() {
                    continue;
                }
                let start = w.bounds[i];
                for j in (i + 1)..=n {
                    let span = j - i;
                    if span > max_piece_length && span != n {
                        break;
                    }
                    let piece = &w.text[start..w.bounds[j]];
                    if let Some(&lp) = logp.get(piece) {
                        let cand = alpha[i] + lp;
                        alpha[j] = log_sum_exp(alpha[j], cand);
                    }
                }
            }
            let log_z = alpha[n];
            if !log_z.is_finite() {
                continue;
            }
            let mut beta = vec![neg_inf(); n + 1];
            beta[n] = 0.0;
            for i in (0..n).rev() {
                let start = w.bounds[i];
                let mut acc = neg_inf();
                for j in (i + 1)..=n {
                    let span = j - i;
                    if span > max_piece_length && span != n {
                        break;
                    }
                    let piece = &w.text[start..w.bounds[j]];
                    if let Some(&lp) = logp.get(piece) {
                        acc = log_sum_exp(acc, lp + beta[j]);
                    }
                }
                beta[i] = acc;
            }
            let scale = w.count as f64;
            for i in 0..n {
                if !alpha[i].is_finite() {
                    continue;
                }
                let start = w.bounds[i];
                for j in (i + 1)..=n {
                    let span = j - i;
                    if span > max_piece_length && span != n {
                        break;
                    }
                    let piece = &w.text[start..w.bounds[j]];
                    if let Some(&lp) = logp.get(piece) {
                        if !beta[j].is_finite() {
                            continue;
                        }
                        let log_post = alpha[i] + lp + beta[j] - log_z;
                        let count = scale * log_post.exp();
                        if count > 0.0 {
                            *expected.entry(piece.to_string()).or_insert(0.0) += count;
                        }
                    }
                }
            }
        }

        // M 步：重新归一化期望计数
        let total: f64 = expected.values().sum();
        let mut new_logp: AHashMap<String, f64> = AHashMap::new();
        if total > 0.0 && total.is_finite() {
            // 用 log 域相减代替 ln(c/total)：c/total 在极低期望计数下会下溢为 0，
            // ln(0) = -inf 会让该 piece 的分数变成非有限值，最终污染 HF 的 Unigram vocab。
            let log_total = total.ln();
            for (p, c) in &expected {
                if *c > 0.0 && c.is_finite() {
                    new_logp.insert(p.clone(), c.ln() - log_total);
                }
            }
        }
        (new_logp, expected)
    }

    /// 剪枝：按"移除后期望似然损失"升序丢弃，回到 `target` 规模。
    fn prune(
        pieces: Vec<String>,
        logp: &AHashMap<String, f64>,
        expected: &AHashMap<String, f64>,
        protected: &AHashSet<String>,
        target: usize,
    ) -> Vec<String> {
        if pieces.len() <= target {
            return pieces;
        }
        let log_ref = -((pieces.len() as f64).ln());
        let mut removable: Vec<(String, f64)> = pieces
            .iter()
            .filter(|p| !protected.contains(*p))
            .map(|p| {
                let freq = expected.get(p).copied().unwrap_or(0.0);
                let lp = logp.get(p).copied().unwrap_or(log_ref - 1.0);
                (p.clone(), freq * (lp - log_ref))
            })
            .collect();
        let need = pieces.len() - target;
        removable.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let drop: AHashSet<String> = removable
            .iter()
            .take(need)
            .map(|(p, _)| p.clone())
            .collect();
        pieces.into_iter().filter(|p| !drop.contains(p)).collect()
    }
}

impl Trainer for UnigramTrainer {
    fn feed_groups(&mut self, groups: &[Vec<String>]) -> Result<()> {
        for g in groups {
            for w in g {
                if w.is_empty() {
                    continue;
                }
                self.words += 1;
                *self.counts.entry(w.clone()).or_insert(0) += 1;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<TrainerArtifact> {
        if self.words == 0 || self.counts.is_empty() {
            return Err(TokenizerError::layer(
                layer::MODEL,
                "Unigram 训练：语料为空",
            ));
        }

        let specials = self.special_list();
        let byte_tokens: Vec<String> = if self.byte_fallback {
            (0..=255u32).map(|b| format!("<0x{:02X}>", b)).collect()
        } else {
            Vec::new()
        };
        let reserved = specials.len() + byte_tokens.len();
        let budget = self.vocab_size.saturating_sub(reserved).max(1);

        // ---- 1. 种子词表
        let cand = self.seed_candidates();
        let mut ranked: Vec<(String, u64)> = cand.iter().map(|(p, c)| (p.clone(), *c)).collect();
        ranked.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then(a.0.len().cmp(&b.0.len()))
                .then(a.0.cmp(&b.0))
        });
        let seed_target = (budget * SEED_MULTIPLIER).clamp(budget, MAX_CANDIDATES);

        let mut seen: AHashSet<String> = AHashSet::new();
        let mut pieces: Vec<String> = Vec::with_capacity(seed_target.min(ranked.len()) + 16);
        // 单字符必保（受保护区）
        let mut chars: Vec<String> = Vec::new();
        for w in self.counts.keys() {
            for c in w.chars() {
                let s = c.to_string();
                if !chars.contains(&s) {
                    chars.push(s.clone());
                }
            }
        }
        chars.sort();
        for c in &chars {
            if seen.insert(c.clone()) {
                pieces.push(c.clone());
            }
        }
        for (p, _) in &ranked {
            if pieces.len() >= seed_target {
                break;
            }
            if cand.get(p).copied().unwrap_or(0) < self.min_frequency {
                continue;
            }
            if seen.insert(p.clone()) {
                pieces.push(p.clone());
            }
        }

        let protected: AHashSet<String> = chars
            .iter()
            .cloned()
            .chain(specials.iter().cloned())
            .collect();
        let mut logp = Self::init_log_probs(&pieces, &cand);

        // ---- 2. EM + 3. 剪枝
        let words: Vec<WordLattice> = self
            .counts
            .iter()
            .map(|(w, c)| WordLattice {
                text: w.clone(),
                bounds: w
                    .char_indices()
                    .map(|(i, _)| i)
                    .chain(std::iter::once(w.len()))
                    .collect(),
                count: *c,
            })
            .collect();

        let mut rounds = 0usize;
        loop {
            let mut expected: AHashMap<String, f64> = AHashMap::new();
            for _ in 0..self.n_sub_iterations {
                let (new_logp, exp) = Self::em_step(&words, &logp, self.max_piece_length);
                if !new_logp.is_empty() {
                    logp = new_logp;
                    expected = exp;
                }
            }
            if pieces.len() <= budget {
                break;
            }
            let target =
                (((pieces.len() as f64) * self.shrinking_factor).floor() as usize).max(budget);
            let target = if target >= pieces.len() {
                pieces.len() - 1
            } else {
                target
            };
            pieces = Self::prune(pieces, &logp, &expected, &protected, target);
            // 剪枝后按剩余集合重估（缺失的给极低分，保证 EM 可继续）
            let mut total = 0.0f64;
            for p in &pieces {
                total += expected.get(p).copied().unwrap_or(0.0);
            }
            if total > 0.0 {
                let log_total = total.ln();
                let mut next: AHashMap<String, f64> = AHashMap::with_capacity(pieces.len());
                for p in &pieces {
                    let c = expected.get(p).copied().unwrap_or(0.0);
                    // 同样走 log 域相减，避免 (c+eps)/total 下溢后取 ln 得到 -inf
                    next.insert(p.clone(), (c + 1e-12).ln() - log_total);
                }
                logp = next;
            }
            rounds += 1;
            eprintln!(
                "[unigram] 剪枝轮次 {}：词表 {} -> {}（目标 {}）",
                rounds,
                pieces.len() + (pieces.len() as f64 / self.shrinking_factor) as usize
                    - pieces.len(),
                pieces.len(),
                budget
            );
            if rounds > 40 {
                break;
            }
        }

        // ---- 4. 组装词表：special → piece（按概率降序）→ 字节 token
        let mut vocab: Vec<(String, f64)> =
            Vec::with_capacity(specials.len() + pieces.len() + byte_tokens.len());
        let mut used: AHashSet<String> = AHashSet::new();
        for s in &specials {
            if used.insert(s.clone()) {
                vocab.push((s.clone(), 0.0));
            }
        }
        let raw: Vec<(&String, f64)> = pieces
            .iter()
            .filter(|p| !used.contains(*p))
            .map(|p| (p, logp.get(p).copied().unwrap_or(-30.0)))
            .collect();
        // HF 的 Unigram `vocab` 只接受有限 score（非有限值经 serde_json 会变 null，
        // 导出后 PreTrainedTokenizerFast 直接加载失败）。这里对极端分数兜底到当前最低分。
        let finite_floor = raw
            .iter()
            .filter(|(_, s)| s.is_finite())
            .map(|(_, s)| *s)
            .fold(f64::INFINITY, f64::min);
        let fallback = if finite_floor.is_finite() {
            finite_floor
        } else {
            -30.0
        };
        let mut ordered: Vec<(&String, f64)> = raw
            .into_iter()
            .map(|(p, s)| (p, if s.is_finite() { s } else { fallback }))
            .collect();
        ordered.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(b.0))
        });
        for (p, s) in ordered {
            if used.insert(p.clone()) {
                vocab.push((p.clone(), s));
            }
        }
        let piece_floor = vocab
            .iter()
            .filter(|(p, _)| !specials.contains(p))
            .map(|(_, s)| *s)
            .fold(f64::INFINITY, f64::min);
        let byte_score = if piece_floor.is_finite() {
            piece_floor
        } else {
            -20.0
        };
        for b in byte_tokens {
            if used.insert(b.clone()) {
                vocab.push((b, byte_score));
            }
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
        UnigramModel::with_unk_text(
            vec![
                ("<unk>".to_string(), 0.0),
                ("a".to_string(), -3.0),
                ("b".to_string(), -3.0),
                ("ab".to_string(), -1.0),
                ("c".to_string(), -3.0),
            ],
            "<unk>",
            false,
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
    fn unk_nodes_are_fused() {
        let m = model();
        // 'x' 未登录 → 单字符 unk 节点，连续两个 unk 应被 fuse 成一个整段
        let toks = m.tokenize("x".repeat(2).as_str()).unwrap();
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].value, "xx");
        assert!(toks[0].offsets == (0, 2));
    }

    #[test]
    fn byte_fallback_splits_unknown_bytes() {
        let mut pieces = vec![("<unk>".to_string(), 0.0)];
        for b in 0..=255u32 {
            pieces.push((format!("<0x{:02X}>", b), -20.0));
        }
        let m = UnigramModel::from_pieces(pieces, Some(0), true).unwrap();
        let toks = m.tokenize("é").unwrap();
        assert_eq!(toks.len(), 2);
        assert_eq!(toks[0].value, "<0xC3>");
        assert_eq!(toks[1].value, "<0xA9>");
    }

    #[test]
    fn hf_json_uses_pair_list() {
        let m = model();
        let j = m.to_hf_json().unwrap();
        assert_eq!(j["type"], "Unigram");
        assert_eq!(j["unk_id"], 0);
        let arr = j["vocab"].as_array().unwrap();
        assert_eq!(arr[0][0], "<unk>");
        assert!(arr[0][1].is_number());
    }

    #[test]
    fn trainer_assigns_unk_and_specials() {
        let params = TrainerParams {
            vocab_size: 24,
            min_frequency: 1,
            ..Default::default()
        };
        let mut t = UnigramTrainer::new(params, "<unk>");
        t.feed_groups(&[vec!["ab".into(), "ab".into(), "abc".into()]])
            .unwrap();
        let TrainerArtifact::Unigram { vocab } = t.finish().unwrap() else {
            panic!()
        };
        assert!(vocab.iter().any(|(p, _)| p == "<unk>"));
        assert!(vocab.iter().any(|(p, _)| p == "ab"));
    }

    #[test]
    fn trainer_em_beats_raw_frequency() {
        // "ab" 常在 "abc"/"abd" 中出现，EM 后应作为整体进入词表且分数高于其组成字符之和
        let params = TrainerParams {
            vocab_size: 12,
            min_frequency: 1,
            ..Default::default()
        };
        let mut t = UnigramTrainer::new(params, "<unk>");
        let mut groups = Vec::new();
        for _ in 0..50 {
            groups.push(vec!["abc".to_string(), "abd".to_string(), "ab".to_string()]);
        }
        t.feed_groups(&groups).unwrap();
        let TrainerArtifact::Unigram { vocab } = t.finish().unwrap() else {
            panic!()
        };
        let m = UnigramModel::with_unk_text(vocab, "<unk>", false).unwrap();
        let toks = m.tokenize("ab").unwrap();
        assert_eq!(toks.len(), 1, "EM 后 'ab' 应被整体切出");
    }

    #[test]
    fn trainer_scores_stay_finite_for_hf_export() {
        // 剪枝后的期望计数可能出现极低值：若用 ln(c/total) 会下溢成 -inf，
        // 序列化到 tokenizer.json 时变成 null，HF 直接加载失败。
        let params = TrainerParams {
            vocab_size: 96,
            min_frequency: 1,
            ..Default::default()
        };
        let mut t = UnigramTrainer::new(params, "<unk>");
        let text = "the quick brown fox jumps over the lazy dog while another quick brown fox \
                    writes tokenizer tests with unicode héllo 中文 mixed and punctuation, like this: yes!";
        let words: Vec<String> = text.split_whitespace().map(|w| w.to_string()).collect();
        let mut groups = Vec::new();
        for _ in 0..40 {
            groups.push(words.clone());
        }
        t.feed_groups(&groups).unwrap();
        let TrainerArtifact::Unigram { vocab } = t.finish().unwrap() else {
            panic!()
        };
        assert!(
            vocab.iter().all(|(_, s)| s.is_finite()),
            "训练产物出现非有限 score，会导致 HF vocab 里出现 null"
        );
        let m = UnigramModel::with_unk_text(vocab, "<unk>", false).unwrap();
        let arr = m.to_hf_json().unwrap()["vocab"].as_array().unwrap().clone();
        assert!(!arr.is_empty());
        for entry in arr {
            let pair = entry.as_array().unwrap();
            assert_eq!(pair.len(), 2);
            assert!(pair[0].is_string());
            assert!(
                pair[1].is_number(),
                "Unigram score 必须是数字，实际 {}",
                pair[1]
            );
        }
    }

    #[test]
    fn byte_fallback_vocab_is_complete() {
        let params = TrainerParams {
            vocab_size: 300,
            min_frequency: 1,
            byte_fallback: true,
            ..Default::default()
        };
        let mut t = UnigramTrainer::new(params, "<unk>");
        t.feed_groups(&[vec!["héllo".into()]]).unwrap();
        let TrainerArtifact::Unigram { vocab } = t.finish().unwrap() else {
            panic!()
        };
        let m = UnigramModel::with_unk_text(vocab, "<unk>", true).unwrap();
        assert!(m.score_of("<0xE9>").is_some());
        // 语料中出现过的字符已作为单字符进词表（受保护区），此处用未登录字符验证 byte_fallback
        assert_eq!(m.tokenize("ü").unwrap().len(), 2);
    }
}
