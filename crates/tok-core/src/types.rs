//! 核心数据结构：NormalizedString（带对齐表）、PreToken、Token、Encoding。
//!
//! 关键不变量（对应架构方案 §4.2）：
//! - INV-1 `Encoding.ids.len() == tokens.len() == offsets.len() == attention_mask.len() == special_tokens_mask.len()`
//! - INV-2 `NormalizedString.alignments` 对 normalized 全覆盖（字节区间，单片不跨段）
//! - INV-3 词表 ID 双向可还原
//! - INV-4 退化路径不静默：对齐只能近似时置 `degraded = true`

use serde::{Deserialize, Serialize};

use crate::config::{PaddingDirection, PaddingParams, TruncationParams, TruncationStrategy};
use crate::error::{Result, TokenizerError};

/// 字节区间（相对原始输入的 UTF-8 字节偏移）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Offset {
    pub start: usize,
    pub end: usize,
}

/// 对齐单元：normalized 区间 ↔ original 区间（均为字节区间，且落在字符边界上）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Alignment {
    pub n_start: usize,
    pub n_end: usize,
    pub o_start: usize,
    pub o_end: usize,
}

/// 切分时对分隔符的处理方式（HF `SplitDelimiterBehavior` 同构）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitDelimiterBehavior {
    /// 丢弃分隔符
    Removed,
    /// 分隔符独立成段
    Isolated,
    /// 分隔符并入前一段
    MergedWithPrevious,
    /// 分隔符并入后一段
    MergedWithNext,
}

fn char_spans(s: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (i, c) in s.char_indices() {
        out.push((i, i + c.len_utf8()));
    }
    out
}

/// 带对齐信息的可变字符串。所有变换（规范化、预分词切分）都保持 original↔normalized 的映射。
#[derive(Debug, Clone)]
pub struct NormalizedString {
    original: String,
    normalized: String,
    alignments: Vec<Alignment>,
    splits: Option<Vec<(usize, usize)>>,
    degraded: bool,
}

impl NormalizedString {
    pub fn new(s: &str) -> Self {
        let mut alignments = Vec::new();
        for (s0, e0) in char_spans(s) {
            alignments.push(Alignment {
                n_start: s0,
                n_end: e0,
                o_start: s0,
                o_end: e0,
            });
        }
        Self {
            original: s.to_string(),
            normalized: s.to_string(),
            alignments,
            splits: None,
            degraded: false,
        }
    }

    pub fn original(&self) -> &str {
        &self.original
    }

    pub fn normalized(&self) -> &str {
        &self.normalized
    }

    pub fn len(&self) -> usize {
        self.normalized.len()
    }

    pub fn is_empty(&self) -> bool {
        self.normalized.is_empty()
    }

    pub fn is_degraded(&self) -> bool {
        self.degraded
    }

    pub fn alignments(&self) -> &[Alignment] {
        &self.alignments
    }

    pub fn splits(&self) -> Option<&[(usize, usize)]> {
        self.splits.as_deref()
    }

    pub fn clear_splits(&mut self) {
        self.splits = None;
    }

    /// 标记退化（对齐不精确），调用方负责向上传播到 Encoding。
    pub fn mark_degraded(&mut self) {
        self.degraded = true;
    }

    /// 对 normalized 做任意变换，并用"配对键 + 旧对齐表复合"重建对齐表。
    ///
    /// 说明：架构方案 §4.2 规划了 NW 动态规划（阈值 1e6 字符），V0.4 实现为
    /// 单调贪心 + 窗口前瞻，字符数或字节长度发生变化（如 NFKC 全角折叠）时置
    /// `degraded`，保证"不静默"而非"完美对齐"；NW 升级点已在 README 中登记。
    pub fn transform<F: Fn(&str) -> String>(&mut self, f: F) {
        let new = f(&self.normalized);
        if new == self.normalized {
            return;
        }
        self.realign(new);
    }

    fn realign(&mut self, new: String) {
        if new == self.normalized {
            return;
        }
        let old = std::mem::replace(&mut self.normalized, new);
        let prev = std::mem::take(&mut self.alignments);
        let (old_keys, old_key_spans) = keys_with_spans(&old);
        let (new_keys, new_spans) = keys_with_spans(&self.normalized);
        // 先按"配对键"做单调贪心对齐，再把旧串字符跨度经上一轮对齐表复合回 original，
        // 从而保证多级规范化串联后偏移仍指向原始输入（INV-3）。
        let map = greedy_align(&old_keys, &new_keys);
        let mut alignments: Vec<Alignment> = Vec::with_capacity(new_keys.len());
        for (idx, oi) in map.iter().enumerate() {
            let (ns, ne) = new_spans[idx];
            let (ks, ke) = if old_key_spans.is_empty() {
                (0usize, 0usize)
            } else {
                old_key_spans[(*oi).min(old_key_spans.len() - 1)]
            };
            let (os, oe) = map_span_to_original(&prev, ks, ke);
            if let Some(last) = alignments.last_mut() {
                if last.o_start == os && last.o_end == oe && last.n_end == ns {
                    last.n_end = ne;
                    continue;
                }
            }
            alignments.push(Alignment {
                n_start: ns,
                n_end: ne,
                o_start: os,
                o_end: oe,
            });
        }
        let changed_arity = old_keys.len() != new_keys.len();
        let changed_len = old.len() != self.normalized.len();
        self.alignments = alignments;
        self.splits = None;
        if changed_arity || changed_len {
            self.degraded = true;
        }
    }

    /// 逐字符变换：保持 1 字符 ↔ 1 个 original 字节区间，避免"整串 realign"引入次生模糊对齐。
    fn map_chars_keep_arity<F: FnMut(char) -> String>(&mut self, mut f: F) {
        let mut out = String::with_capacity(self.normalized.len());
        let mut alignments: Vec<Alignment> = Vec::with_capacity(self.alignments.len());
        for a in &self.alignments {
            let seg = self.normalized[a.n_start..a.n_end].to_string();
            for c in seg.chars() {
                let new = f(c);
                if new.is_empty() {
                    continue;
                }
                let n_start = out.len();
                out.push_str(&new);
                alignments.push(Alignment {
                    n_start,
                    n_end: out.len(),
                    o_start: a.o_start,
                    o_end: a.o_end,
                });
            }
        }
        if out.chars().count() != self.normalized.chars().count() {
            self.degraded = true;
        }
        self.normalized = out;
        self.alignments = alignments;
        self.splits = None;
    }

    pub fn lowercase(&mut self) {
        // 大小写转换天然 1 字符 ↔ 1 字符（`İ` 等特殊映射除外，由 arity 判定兜底），
        // 逐字符处理可避免整串 realign 时因字符不同而落入"窗口前瞻失败"的模糊分支。
        self.map_chars_keep_arity(|c| c.to_lowercase().collect());
    }

    pub fn nfc(&mut self) {
        use unicode_normalization::UnicodeNormalization;
        let s: String = self.normalized.nfc().collect();
        self.realign(s);
    }

    pub fn nfd(&mut self) {
        use unicode_normalization::UnicodeNormalization;
        let s: String = self.normalized.nfd().collect();
        self.realign(s);
    }

    pub fn nfkc(&mut self) {
        use unicode_normalization::UnicodeNormalization;
        let s: String = self.normalized.nfkc().collect();
        self.realign(s);
    }

    pub fn nfkd(&mut self) {
        use unicode_normalization::UnicodeNormalization;
        let s: String = self.normalized.nfkd().collect();
        self.realign(s);
    }

    pub fn strip_accents(&mut self) {
        use unicode_normalization::UnicodeNormalization;
        let s: String = self
            .normalized
            .nfd()
            .filter(|c| !is_combining_mark(*c))
            .collect();
        self.realign(s);
    }

    /// 字符串首尾去空白（等价 HF `Strip`）。
    pub fn strip(&mut self) {
        let trimmed = self.normalized.trim();
        if trimmed.len() == self.normalized.len() {
            return;
        }
        let start = trimmed.as_ptr() as usize - self.normalized.as_ptr() as usize;
        let end = start + trimmed.len();
        let seg = trimmed.to_string();
        let mut new_align = Vec::new();
        for a in &self.alignments {
            if a.n_start >= start && a.n_end <= end {
                new_align.push(Alignment {
                    n_start: a.n_start - start,
                    n_end: a.n_end - start,
                    o_start: a.o_start,
                    o_end: a.o_end,
                });
            }
        }
        self.normalized = seg;
        self.alignments = new_align;
        self.splits = None;
    }

    /// 正则替换（对应 HF `Replace`）。
    pub fn replace_all(&mut self, re: &regex::Regex, content: &str) {
        if !re.is_match(&self.normalized) {
            return;
        }
        let replaced = re.replace_all(&self.normalized, content).to_string();
        self.realign(replaced);
    }

    /// 前置拼接并把插入片段的 original 区间吸到紧随其后的首个字符上。
    ///
    /// 用于 Metaspace 的 `▁` 前缀：HF 对句首插入的 `▁` 不给独立空区间，
    /// 而是与首个字符共享同一 original 区间（`"hello"` → token `▁hello` offset `(0,5)`）。
    pub fn prepend_absorbing_next(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        let (o_start, o_end) = self
            .alignments
            .first()
            .map(|a| (a.o_start, a.o_end))
            .unwrap_or((0, 0));
        let mut new_norm = String::with_capacity(self.normalized.len() + s.len());
        new_norm.push_str(s);
        new_norm.push_str(&self.normalized);
        let mut alignments = Vec::with_capacity(self.alignments.len() + 1);
        alignments.push(Alignment {
            n_start: 0,
            n_end: s.len(),
            o_start,
            o_end,
        });
        for a in &self.alignments {
            alignments.push(Alignment {
                n_start: a.n_start + s.len(),
                n_end: a.n_end + s.len(),
                o_start: a.o_start,
                o_end: a.o_end,
            });
        }
        self.normalized = new_norm;
        self.alignments = alignments;
        self.splits = None;
    }

    /// 前置拼接（对应 HF `Prepend`，如 Metaspace 的 `▁`）。
    pub fn prepend(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        let mut new_norm = String::with_capacity(self.normalized.len() + s.len());
        new_norm.push_str(s);
        new_norm.push_str(&self.normalized);
        let mut alignments = vec![Alignment {
            n_start: 0,
            n_end: s.len(),
            o_start: 0,
            o_end: 0,
        }];
        for a in &self.alignments {
            alignments.push(Alignment {
                n_start: a.n_start + s.len(),
                n_end: a.n_end + s.len(),
                o_start: a.o_start,
                o_end: a.o_end,
            });
        }
        self.normalized = new_norm;
        self.alignments = alignments;
        self.splits = None;
    }

    /// 逐字符映射：返回 `Some(new)` 表示替换（`new` 可为多字符或空串），`None` 表示保留。
    ///
    /// 对齐规则：替换后的所有字符统一映射回原字符所在的 original 字节区间；
    /// 仅当"字符个数发生变化"时置 `degraded`（INV-4：退化不静默）。
    pub fn map_chars<F: FnMut(char) -> Option<String>>(&mut self, mut f: F) {
        let mut out = String::with_capacity(self.normalized.len());
        let mut alignments: Vec<Alignment> = Vec::with_capacity(self.alignments.len());
        let mut changed = false;
        for a in &self.alignments {
            let seg = self.normalized[a.n_start..a.n_end].to_string();
            for c in seg.chars() {
                match f(c) {
                    Some(new) => {
                        if new != c.to_string() {
                            changed = true;
                        }
                        if new.is_empty() {
                            continue;
                        }
                        let n_start = out.len();
                        out.push_str(&new);
                        alignments.push(Alignment {
                            n_start,
                            n_end: out.len(),
                            o_start: a.o_start,
                            o_end: a.o_end,
                        });
                    }
                    None => {
                        let n_start = out.len();
                        out.push(c);
                        alignments.push(Alignment {
                            n_start,
                            n_end: out.len(),
                            o_start: a.o_start,
                            o_end: a.o_end,
                        });
                    }
                }
            }
        }
        if !changed && out.len() == self.normalized.len() {
            return;
        }
        let arity_changed = out.chars().count() != self.normalized.chars().count();
        self.normalized = out;
        self.alignments = alignments;
        self.splits = None;
        if arity_changed {
            self.degraded = true;
        }
    }

    /// 字节级重映射（L2 `ByteLevel`）：original 的每个 **UTF-8 字节**变为一个映射字符。
    ///
    /// 这是"字节级"的精确实现，逐字节建立对齐（1 字符 ↔ 1 原始字节），不置 degraded。
    pub fn byte_level_expand(&mut self) {
        let mut out = String::new();
        let mut alignments = Vec::new();
        for a in &self.alignments {
            let seg = self.normalized[a.n_start..a.n_end].to_string();
            for (byte_cursor, b) in (a.o_start..).zip(seg.bytes()) {
                let n_start = out.len();
                out.push(crate::byte_level::byte_to_char(b));
                alignments.push(Alignment {
                    n_start,
                    n_end: out.len(),
                    o_start: byte_cursor,
                    o_end: byte_cursor + 1,
                });
            }
        }
        self.normalized = out;
        self.alignments = alignments;
        self.splits = None;
    }

    /// `ByteLevel` 的两阶段语义（与 HF 对齐）：**先按正则切分原始文本，再对每个 piece 做字节级映射**。
    ///
    /// `hits` 为**映射前**文本上的匹配区间（字节范围，不要求覆盖全串；未命中的空隙自成 piece）。
    /// 不能"先 expand 再切分"：映射字符如 `Ġ`(U+0120) 本身属于 `\p{L}`，会让正则把
    /// `"Hello world"` 视为同一个 `\p{L}+` 串，piece 边界与 HF 不一致（进而训练/编码结果不同）。
    pub fn byte_level_map_pieces(&mut self, hits: Vec<(usize, usize)>) {
        let segments = self.current_segments();
        let mut out = String::new();
        let mut alignments: Vec<Alignment> = Vec::new();
        let mut splits: Vec<(usize, usize)> = Vec::new();

        for (s, e) in segments {
            if e <= s || e > self.normalized.len() {
                continue;
            }
            let text = &self.normalized[s..e];
            // 只取落在本 segment 内的命中区间；与未命中的空隙共同覆盖整个 segment，避免静默丢字符
            let mut local: Vec<(usize, usize)> = hits
                .iter()
                .copied()
                .filter(|(hs, he)| *hs >= s && *he <= e && he > hs)
                .map(|(hs, he)| (hs - s, he - s))
                .collect();
            local.sort_unstable();
            let mut pieces: Vec<(usize, usize)> = Vec::new();
            let mut cursor = 0usize;
            for (hs, he) in local {
                if hs > cursor {
                    pieces.push((cursor, hs));
                }
                if he > cursor {
                    pieces.push((hs, he));
                    cursor = he;
                }
            }
            if cursor < text.len() {
                pieces.push((cursor, text.len()));
            }

            for (ps, pe) in pieces {
                let abs_start = s + ps;
                let abs_end = s + pe;
                let mut o_start: Option<usize> = None;
                let mut o_end = 0usize;
                for a in &self.alignments {
                    if a.n_end > abs_start && a.n_start < abs_end {
                        o_start = Some(o_start.map_or(a.o_start, |v: usize| v.min(a.o_start)));
                        o_end = o_end.max(a.o_end);
                    }
                }
                let piece_start = out.len();
                for (byte_cursor, b) in (o_start.unwrap_or(abs_start)..)
                    .zip(self.normalized[abs_start..abs_end].bytes())
                {
                    let n_start = out.len();
                    out.push(crate::byte_level::byte_to_char(b));
                    alignments.push(Alignment {
                        n_start,
                        n_end: out.len(),
                        o_start: byte_cursor,
                        o_end: byte_cursor + 1,
                    });
                }
                if out.len() > piece_start {
                    splits.push((piece_start, out.len()));
                }
            }
        }

        self.normalized = out;
        self.alignments = alignments;
        self.splits = Some(splits);
    }

    /// 用"匹配片段"覆盖切分：matches 必须是 normalized 中不重叠、单调递增的区间。
    pub fn split_by_matches(&mut self, matches: Vec<(usize, usize)>) {
        let segments = self.current_segments();
        let mut out = Vec::new();
        for (s, e) in segments {
            for (a, b) in matches.iter().copied() {
                if a >= s && b <= e && b > a {
                    out.push((a, b));
                }
            }
        }
        self.splits = Some(out);
    }

    /// 用"分隔符区间"覆盖切分，按 behavior 决定分隔符去向。
    pub fn split_by_delims(
        &mut self,
        delims: Vec<(usize, usize)>,
        behavior: SplitDelimiterBehavior,
    ) {
        let segments = self.current_segments();
        let mut out = Vec::new();
        for (s, e) in segments {
            let ds: Vec<(usize, usize)> = delims
                .iter()
                .copied()
                .filter(|(a, b)| *a >= s && *b <= e && b > a)
                .collect();
            out.extend(apply_split_one(&ds, s, e, behavior));
        }
        self.splits = Some(out);
    }

    /// 按正则切分。
    pub fn split_regex(
        &mut self,
        re: &regex::Regex,
        behavior: SplitDelimiterBehavior,
    ) -> Result<()> {
        let delims: Vec<(usize, usize)> = re
            .find_iter(&self.normalized)
            .map(|m| (m.start(), m.end()))
            .collect();
        self.split_by_delims(delims, behavior);
        Ok(())
    }

    /// 按谓词切分（连续满足谓词的字符视为一段分隔符）。
    pub fn split_on_pred<F: Fn(char) -> bool>(
        &mut self,
        pred: F,
        behavior: SplitDelimiterBehavior,
    ) {
        let mut delims = Vec::new();
        let mut cur: Option<(usize, usize)> = None;
        // Isolated：每个匹配字符各自成为一段分隔符（HF 语义，BertPreTokenizer 依赖）；
        // 其余行为（Removed / MergedWithPrevious / MergedWithNext）连续匹配字符合并为一段。
        let isolated = matches!(behavior, SplitDelimiterBehavior::Isolated);
        for (i, c) in self.normalized.char_indices() {
            if pred(c) {
                if isolated {
                    if let Some(d) = cur.take() {
                        delims.push(d);
                    }
                    delims.push((i, i + c.len_utf8()));
                } else {
                    match cur {
                        Some((s, _)) => cur = Some((s, i + c.len_utf8())),
                        None => cur = Some((i, i + c.len_utf8())),
                    }
                }
            } else if let Some(d) = cur.take() {
                delims.push(d);
            }
        }
        if let Some(d) = cur.take() {
            delims.push(d);
        }
        self.split_by_delims(delims, behavior);
    }

    fn current_segments(&self) -> Vec<(usize, usize)> {
        match &self.splits {
            Some(s) => s.clone(),
            None => vec![(0, self.normalized.len())],
        }
    }

    /// normalized 字节区间 → original 字节区间（取覆盖该区间的最小对齐跨度）。
    pub fn convert_offsets(&self, range: (usize, usize)) -> Option<Offset> {
        let (start, end) = range;
        if start >= end {
            return None;
        }
        let mut first: Option<Alignment> = None;
        let mut last: Option<Alignment> = None;
        for a in &self.alignments {
            if a.n_end > start && a.n_start < end {
                if first.is_none() {
                    first = Some(*a);
                }
                last = Some(*a);
            }
        }
        let f = first?;
        let l = last?;
        let s = f.o_start;
        let e = l.o_end.max(s);
        Some(Offset { start: s, end: e })
    }

    /// 产出预分词单元（含 original 偏移）。
    pub fn pretokens(&self) -> Vec<PreToken> {
        let segs = self.current_segments();
        segs.into_iter()
            .enumerate()
            .filter(|(_, (a, b))| b > a)
            .map(|(i, (a, b))| {
                let value = self.normalized[a..b].to_string();
                let offsets = match self.convert_offsets((a, b)) {
                    Some(o) => (o.start, o.end),
                    None => (a, b),
                };
                PreToken {
                    id: i,
                    value,
                    offsets,
                    n_offsets: (a, b),
                }
            })
            .collect()
    }
}

fn is_combining_mark(c: char) -> bool {
    use unicode_normalization::char::is_combining_mark;
    is_combining_mark(c)
}

/// 贪心单调对齐：为 new 的每个键在 old 中找对应位置（窗口前瞻 8 个键）。
fn greedy_align<T: PartialEq>(old: &[T], new: &[T]) -> Vec<usize> {
    let mut res = Vec::with_capacity(new.len());
    if old.is_empty() {
        return vec![0; new.len()];
    }
    let mut i = 0usize;
    let mut last = 0usize;
    for c in new {
        if i < old.len() && &old[i] == c {
            res.push(i);
            last = i;
            i += 1;
            continue;
        }
        let hi = (i + 8).min(old.len());
        let found: Option<usize> = old
            .iter()
            .enumerate()
            .take(hi)
            .skip(i)
            .find(|&(_, o)| o == c)
            .map(|(k, _)| k);
        match found {
            Some(k) => {
                res.push(k);
                last = k;
                i = k + 1;
            }
            None => res.push(last),
        }
    }
    res
}

/// 对齐配对键：NFKC 兼容折叠 → 去组合附标 → 小写。
///
/// 用于跨"写法差异"匹配字符（如全角 `Ａ` ↔ 半角 `A`、`é` ↔ `e`），
/// 使 realign 在规范化前后能找到稳定的 1:1 对应关系。
fn align_key(c: char) -> String {
    use unicode_normalization::UnicodeNormalization;
    let folded: String = c.to_string().nfkc().collect();
    let stripped: String = folded.nfd().filter(|x| !is_combining_mark(*x)).collect();
    let base = if stripped.is_empty() {
        folded
    } else {
        stripped
    };
    if base.is_empty() {
        c.to_string()
    } else {
        base.to_lowercase()
    }
}

/// 逐字符产出 `(配对键, 该字符在 s 中的字节跨度)`。
fn keys_with_spans(s: &str) -> (Vec<String>, Vec<(usize, usize)>) {
    let mut keys = Vec::with_capacity(s.len());
    let mut spans = Vec::with_capacity(s.len());
    for (i, c) in s.char_indices() {
        keys.push(align_key(c));
        spans.push((i, i + c.len_utf8()));
    }
    (keys, spans)
}

/// 上一轮 normalized 的字节跨度 `[start,end)` → original 字节区间（取覆盖该跨度的最小对齐包络）。
///
/// 无对齐重叠时（理论上只在旧对齐表为空时发生）退化为跨度本身，即假定该段未被变换过。
fn map_span_to_original(prev: &[Alignment], start: usize, end: usize) -> (usize, usize) {
    if start >= end {
        return (start, end);
    }
    let mut first: Option<&Alignment> = None;
    let mut last: Option<&Alignment> = None;
    for a in prev {
        if a.n_end > start && a.n_start < end {
            if first.is_none() {
                first = Some(a);
            }
            last = Some(a);
        }
    }
    match (first, last) {
        (Some(f), Some(l)) => (f.o_start, l.o_end.max(f.o_start)),
        _ => (start, end),
    }
}

fn apply_split_one(
    delims: &[(usize, usize)],
    start: usize,
    end: usize,
    behavior: SplitDelimiterBehavior,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut cursor = start;
    for (ds, de) in delims.iter().copied() {
        if de <= cursor {
            continue;
        }
        match behavior {
            SplitDelimiterBehavior::Removed => {
                if ds > cursor {
                    out.push((cursor, ds));
                }
                cursor = de;
            }
            SplitDelimiterBehavior::Isolated => {
                if ds > cursor {
                    out.push((cursor, ds));
                }
                if de > ds {
                    out.push((ds, de));
                }
                cursor = de;
            }
            SplitDelimiterBehavior::MergedWithPrevious => {
                out.push((cursor, de));
                cursor = de;
            }
            SplitDelimiterBehavior::MergedWithNext => {
                if ds > cursor {
                    out.push((cursor, ds));
                }
                cursor = ds;
            }
        }
    }
    if behavior == SplitDelimiterBehavior::MergedWithNext {
        if cursor < end {
            out.push((cursor, end));
        }
    } else if cursor < end {
        out.push((cursor, end));
    }
    out
}

/// 预分词单元：L2 的输出、L3 的输入单位。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreToken {
    pub id: usize,
    pub value: String,
    /// 在预分词输出串（所属 normalized 串）中的字符区间；
    /// 子词偏移以 `value` 为基准，叠加本区间起点才能回到 normalized 坐标。
    pub n_offsets: (usize, usize),
    /// 在**原始输入**中的字节区间（已通过对齐表换算）
    pub offsets: (usize, usize),
}

/// 子词单元：L3 的输出。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token {
    /// 词表 ID（未登录且已配置 unk 时为 unk_id）
    pub id: u32,
    /// 词表内字面值（BPE 为子词串，WordPiece 去掉 `##` 前缀）
    pub value: String,
    /// 相对**预分词单元 value** 的字节区间
    pub offsets: (usize, usize),
}

/// 一次编码的完整结果。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Encoding {
    pub ids: Vec<u32>,
    pub type_ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub offsets: Vec<(usize, usize)>,
    pub attention_mask: Vec<u32>,
    pub special_tokens_mask: Vec<u32>,
    /// 溢出片段（截断时填充）
    pub overflowing: Vec<Encoding>,
    /// 序列归属：键为序列序号（0/1/...），值为内容区（不含模板特殊 token）
    pub sequence_ranges: std::collections::BTreeMap<usize, (usize, usize)>,
    /// 对齐退化标记（L1 无法精确对齐时置位，端到端不静默）
    pub offsets_degraded: bool,
}

impl Encoding {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// 拼接另一个 Encoding（无后处理器时的 pair 退化路径）。
    pub fn merge(&mut self, other: Encoding, growing_offsets: bool) {
        let off = if growing_offsets { self.ids.len() } else { 0 };
        let base_key = self.sequence_ranges.len();
        self.ids.extend(other.ids);
        self.type_ids.extend(other.type_ids);
        self.tokens.extend(other.tokens);
        for (s, e) in other.offsets {
            self.offsets.push((s + off, e + off));
        }
        self.attention_mask.extend(other.attention_mask);
        self.special_tokens_mask.extend(other.special_tokens_mask);
        for (k, (s, e)) in other.sequence_ranges.iter() {
            self.sequence_ranges
                .insert(base_key + *k, (s + off, e + off));
        }
        self.offsets_degraded |= other.offsets_degraded;
        self.overflowing.extend(other.overflowing);
    }

    /// INV-1 校验。
    pub fn validate(&self) -> Result<()> {
        let n = self.ids.len();
        let dims = [
            ("type_ids", self.type_ids.len()),
            ("tokens", self.tokens.len()),
            ("offsets", self.offsets.len()),
            ("attention_mask", self.attention_mask.len()),
            ("special_tokens_mask", self.special_tokens_mask.len()),
        ];
        for (name, len) in dims {
            if len != n {
                return Err(TokenizerError::layer(
                    0,
                    format!("INV-1 失败：ids={} 与 {}={} 不等", n, name, len),
                ));
            }
        }
        Ok(())
    }

    /// 截断：返回 [首片, 溢出片...]；无截断时返回单元素。
    ///
    /// 截断到 `params.max_length`（模板新增的特殊 token 不占内容预算，与 HF 一致）。
    ///
    /// 返回 `[主片段, 溢出片段...]`：内容区取自 `sequence_ranges`（pair 时为两段），
    /// 按 `strategy` 决定先裁哪条序列；`stride = 0` 时只产出一个含全部被移出 token 的
    /// 溢出片段，`stride > 0` 时按"预算 - stride"步进产出重叠滑窗（HF `stride` 语义）。
    pub fn truncate(&self, params: &TruncationParams) -> Vec<Encoding> {
        let len = self.ids.len();
        if len <= params.max_length {
            return vec![self.clone()];
        }
        let n_special = self.special_tokens_mask.iter().filter(|m| **m == 1).count();
        let budget = params.max_length.saturating_sub(n_special);
        let regions = self.content_regions();
        let lens: Vec<usize> = regions.iter().map(|(s, e)| e.saturating_sub(*s)).collect();
        let total: usize = lens.iter().sum();
        if budget == 0 || total <= budget {
            return vec![self.clone()];
        }
        // 裁剪顺序：LongestFirst 按内容长度降序（并列时序列号小者优先）
        let mut order: Vec<usize> = match params.strategy {
            TruncationStrategy::OnlyFirst => vec![0],
            TruncationStrategy::OnlySecond => vec![1, 0],
            TruncationStrategy::LongestFirst => {
                let mut o: Vec<usize> = (0..lens.len()).collect();
                o.sort_by_key(|k| std::cmp::Reverse(lens[*k]));
                o
            }
        };
        order.retain(|k| *k < lens.len());
        let mut keep: Vec<(usize, usize)> = (0..lens.len()).map(|k| (0, lens[k])).collect();
        let mut excess = total - budget;
        for k in order {
            if excess == 0 {
                break;
            }
            let take = excess.min(keep[k].1 - keep[k].0);
            match params.direction {
                crate::config::TruncationDirection::Right => keep[k].1 -= take,
                crate::config::TruncationDirection::Left => keep[k].0 += take,
            }
            excess -= take;
        }
        // 主片段：模板特殊 token 全保留 + 各内容区保留窗口
        let mut keep_set: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        let mut drop_idx: Vec<usize> = Vec::new();
        for (k, (rs, re)) in regions.iter().copied().enumerate() {
            let (from, to) = keep[k];
            for off in 0..re.saturating_sub(rs) {
                let i = rs + off;
                if i >= len {
                    break;
                }
                if off >= from && off < to {
                    keep_set.insert(i);
                } else {
                    drop_idx.push(i);
                }
            }
        }
        let final_keep: Vec<usize> = (0..len)
            .filter(|i| {
                self.special_tokens_mask.get(*i).copied().unwrap_or(0) == 1 || keep_set.contains(i)
            })
            .collect();
        let primary = self.subset(&final_keep);
        if params.stride == 0 {
            let mut out = vec![primary];
            if !drop_idx.is_empty() {
                out.push(self.subset(&drop_idx));
            }
            return out;
        }
        // stride > 0：内容区按序拉平后滑窗（步长 = 预算 - stride），产出重叠溢出片段
        let flat: Vec<usize> = regions
            .iter()
            .flat_map(|(rs, re)| {
                let re = (*re).min(len);
                (*rs..re).collect::<Vec<usize>>()
            })
            .collect();
        let step = budget.saturating_sub(params.stride).max(1);
        let mut out = vec![primary];
        let mut offset = step;
        while offset < flat.len() {
            let to = (offset + budget).min(flat.len());
            if to <= offset {
                break;
            }
            out.push(self.subset(&flat[offset..to]));
            if offset + budget >= flat.len() {
                break;
            }
            offset += step;
        }
        out
    }

    /// 内容区（不含模板特殊 token）：优先 `sequence_ranges`，否则取非特殊 token 的连续主体。
    fn content_regions(&self) -> Vec<(usize, usize)> {
        let len = self.ids.len();
        let mut regions: Vec<(usize, usize)> = self
            .sequence_ranges
            .values()
            .map(|(s, e)| ((*s).min(len), (*e).min(len)))
            .filter(|(s, e)| e > s)
            .collect();
        if !regions.is_empty() {
            regions.sort_unstable();
            return regions;
        }
        let mut s = 0usize;
        while s < len && self.special_tokens_mask.get(s).copied().unwrap_or(0) == 1 {
            s += 1;
        }
        let mut e = len;
        while e > s && self.special_tokens_mask.get(e - 1).copied().unwrap_or(0) == 1 {
            e -= 1;
        }
        regions.push((s, e));
        regions
    }

    /// 按索引子集构造新 Encoding（保持给定顺序，清空 overflowing）。
    fn subset(&self, idx: &[usize]) -> Encoding {
        let mut e = self.clone();
        e.ids = idx.iter().map(|&i| self.ids[i]).collect();
        e.type_ids = idx.iter().map(|&i| self.type_ids[i]).collect();
        e.tokens = idx.iter().map(|&i| self.tokens[i].clone()).collect();
        e.offsets = idx.iter().map(|&i| self.offsets[i]).collect();
        e.attention_mask = idx.iter().map(|&i| self.attention_mask[i]).collect();
        e.special_tokens_mask = idx.iter().map(|&i| self.special_tokens_mask[i]).collect();
        e.overflowing = Vec::new();
        e
    }

    /// 填充到 target_len（就地）。
    pub fn pad(&mut self, params: &PaddingParams, target_len: usize) {
        let len = self.ids.len();
        if target_len <= len {
            return;
        }
        let n = target_len - len;
        match params.direction {
            PaddingDirection::Right => {
                for _ in 0..n {
                    self.ids.push(params.pad_id);
                    self.type_ids.push(params.pad_type_id);
                    self.tokens.push(params.pad_token.clone());
                    self.offsets.push((0, 0));
                    self.attention_mask.push(0);
                    self.special_tokens_mask.push(1);
                }
            }
            PaddingDirection::Left => {
                for _ in 0..n {
                    self.ids.insert(0, params.pad_id);
                    self.type_ids.insert(0, params.pad_type_id);
                    self.tokens.insert(0, params.pad_token.clone());
                    self.offsets.insert(0, (0, 0));
                    self.attention_mask.insert(0, 0);
                    self.special_tokens_mask.insert(0, 1);
                }
                for v in self.sequence_ranges.values_mut() {
                    v.0 += n;
                    v.1 += n;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_offsets_maps_back() {
        let ns = NormalizedString::new("Hello world");
        let t = ns.normalized().find("world").unwrap();
        let o = ns.convert_offsets((t, t + 5)).unwrap();
        assert_eq!((o.start, o.end), (6, 11));
    }

    #[test]
    fn lowercase_keeps_alignment() {
        let mut ns = NormalizedString::new("HeLLo");
        ns.lowercase();
        assert_eq!(ns.normalized(), "hello");
        let o = ns.convert_offsets((0, 5)).unwrap();
        assert_eq!((o.start, o.end), (0, 5));
        assert!(!ns.is_degraded());
    }

    #[test]
    fn split_removed_and_isolated() {
        let mut ns = NormalizedString::new("a b");
        let re = regex::Regex::new(" ").unwrap();
        ns.split_regex(&re, SplitDelimiterBehavior::Removed)
            .unwrap();
        assert_eq!(ns.splits().unwrap(), &[(0, 1), (2, 3)]);
        let mut ns2 = NormalizedString::new("a b");
        ns2.split_regex(&re, SplitDelimiterBehavior::Isolated)
            .unwrap();
        assert_eq!(ns2.splits().unwrap(), &[(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn prepend_shifts_splits() {
        let mut ns = NormalizedString::new("ab");
        ns.prepend("\u{2581}");
        assert_eq!(ns.normalized(), "\u{2581}ab");
        // 前缀字符本身映射回原文起点（零宽区间）
        let p = ns.convert_offsets((0, 3)).unwrap();
        assert_eq!((p.start, p.end), (0, 0));
        // 前缀之后的原文区间整体后移 3 字节，映射需还原为原始偏移
        let o = ns.convert_offsets((3, 5)).unwrap();
        assert_eq!((o.start, o.end), (0, 2));
    }

    #[test]
    fn truncate_produces_overflow_with_stride() {
        let mut e = Encoding::default();
        for i in 0..10u32 {
            e.ids.push(i);
            e.type_ids.push(0);
            e.tokens.push(format!("t{}", i));
            e.offsets.push((i as usize, i as usize + 1));
            e.attention_mask.push(1);
            e.special_tokens_mask.push(0);
        }
        let p = TruncationParams {
            max_length: 4,
            strategy: TruncationStrategy::LongestFirst,
            stride: 1,
            direction: Default::default(),
        };
        let chunks = e.truncate(&p);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].ids, vec![0, 1, 2, 3]);
        assert_eq!(chunks[1].ids, vec![3, 4, 5, 6]);
        assert_eq!(chunks[2].ids, vec![6, 7, 8, 9]);
    }

    #[test]
    fn pad_left_updates_ranges() {
        let mut e = Encoding {
            ids: vec![1, 2],
            type_ids: vec![0, 0],
            tokens: vec!["a".into(), "b".into()],
            offsets: vec![(0, 1), (1, 2)],
            attention_mask: vec![1, 1],
            special_tokens_mask: vec![0, 0],
            ..Default::default()
        };
        e.sequence_ranges.insert(0, (0, 2));
        let p = PaddingParams {
            strategy: crate::config::PaddingStrategy::Fixed(4),
            direction: PaddingDirection::Left,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".into(),
        };
        e.pad(&p, 4);
        assert_eq!(e.ids, vec![0, 0, 1, 2]);
        assert_eq!(e.attention_mask, vec![0, 0, 1, 1]);
        assert_eq!(e.sequence_ranges.get(&0).copied(), Some((2, 4)));
        e.validate().unwrap();
    }
}
