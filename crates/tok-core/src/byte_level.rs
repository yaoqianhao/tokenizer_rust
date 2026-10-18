//! 字节级映射表（GPT-2 风格可打印化映射）。
//!
//! 这是**唯一一份**字节↔字符映射实现：L2 `ByteLevel`、L6 `ByteLevel` 解码器、
//! L3 BPE 的初始字母表共用它（对应架构方案 §4.3 的"仅此一份"约束）。

use std::collections::HashMap;
use std::sync::OnceLock;

/// GPT-2 预分词正则（tiktoken 的 `pat_str` 与此同源，必须可被配置覆写）。
pub const GPT2_BYTE_LEVEL_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

static BYTE_TO_CHAR: OnceLock<[char; 256]> = OnceLock::new();
static CHAR_TO_BYTE: OnceLock<HashMap<char, u8>> = OnceLock::new();

fn build_map() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u32> = Vec::new();
    let mut cs: Vec<u32> = Vec::new();
    for b in 0x21u32..=0x7e {
        bs.push(b);
        cs.push(b);
    }
    for b in 0xa1u32..=0xac {
        bs.push(b);
        cs.push(b);
    }
    for b in 0xaeu32..=0xff {
        bs.push(b);
        cs.push(b);
    }
    let mut n = 0u32;
    for b in 0u32..=0xff {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut arr = ['\u{0}'; 256];
    let mut rev: HashMap<char, u8> = HashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(*c).expect("valid byte-level char");
        arr[*b as usize] = ch;
        rev.insert(ch, *b as u8);
    }
    (arr, rev)
}

fn forward() -> &'static [char; 256] {
    BYTE_TO_CHAR.get_or_init(|| build_map().0)
}

fn reverse() -> &'static HashMap<char, u8> {
    CHAR_TO_BYTE.get_or_init(|| build_map().1)
}

/// 单字节 → 可打印字符
pub fn byte_to_char(b: u8) -> char {
    forward()[b as usize]
}

/// 可打印字符 → 单字节（非映射字符返回 None）
pub fn char_to_byte(c: char) -> Option<u8> {
    reverse().get(&c).copied()
}

/// 字节级编码：每个 UTF-8 字节映射为一个字符。
pub fn byte_level_encode(s: &str) -> String {
    s.bytes().map(byte_to_char).collect()
}

/// 字节级解码：把映射字符还原为字节序列（不做 UTF-8 校验，交由调用方 lossy 处理）。
pub fn byte_level_decode(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    for c in s.chars() {
        if let Some(b) = char_to_byte(c) {
            out.push(b);
        }
    }
    out
}

/// 供 BPE 使用的 256 字节初始字母表（按字节序）。
pub fn byte_alphabet() -> Vec<char> {
    (0u8..=255).map(byte_to_char).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_is_bijective() {
        let mut seen = std::collections::HashSet::new();
        for b in 0u8..=255 {
            let c = byte_to_char(b);
            assert!(seen.insert(c), "duplicate char for byte {}", b);
            assert_eq!(char_to_byte(c), Some(b));
        }
        assert_eq!(byte_to_char(0x21), '!');
        assert_eq!(byte_to_char(0x20), '\u{0120}'); // 空格 → Ġ
    }

    #[test]
    fn encode_decode_roundtrip() {
        let s = "Hello 世界 🌍";
        let mapped = byte_level_encode(s);
        assert_eq!(byte_level_decode(&mapped), s.as_bytes());
    }
}
