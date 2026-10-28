//! 角色链路吞吐复核（P2）。
//!
//! 目的不是刷数字，而是复核一件事：**E6 引入的 `roles` 是否进入了 encode 热路径**。
//! 设计上 roles 只在装配/落盘期使用（写 tokenizer_config.json、special_tokens_mask 判定靠
//! `special_set`），所以「带角色」与「不带角色」两条路径的吞吐应当没有系统性差异；
//! 若差异超过阈值，说明有角色相关内容漏进热路径，需要回查。
//!
//! 跑法（务必 release，debug 的数字没有参考价值）：
//! ```text
//! cargo run --release --example role_throughput -p tok-pipeline
//! ```
//! 输出：两种装配的 tokens/s、每 token 纳秒、以及相对差异百分比。

use std::collections::BTreeMap;
use std::time::Instant;

use tok_pipeline::tokenizer::Tokenizer;
use tok_pipeline::{bert_like, hf::TokenRoles};

/// 构造一份 ~900 词的 wordpiece 词表：[UNK]/[PAD]/[CLS]/[SEP]/[MASK] + w0..w799 + ##0..##99。
fn demo_vocab() -> BTreeMap<String, u32> {
    let mut vocab = BTreeMap::new();
    let mut id = 0u32;
    for t in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] {
        vocab.insert(t.to_string(), id);
        id += 1;
    }
    for i in 0..800 {
        vocab.insert(format!("w{i}"), id);
        id += 1;
    }
    for i in 0..100 {
        vocab.insert(format!("##{i}"), id);
        id += 1;
    }
    vocab
}

fn roles_full() -> TokenRoles {
    TokenRoles {
        unk_token: Some("[UNK]".to_string()),
        bos_token: Some("<|begin|>".to_string()),
        eos_token: Some("<|end|>".to_string()),
        pad_token: Some("[PAD]".to_string()),
        cls_token: Some("[CLS]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        mask_token: Some("[MASK]".to_string()),
        additional_special_tokens: (0..8).map(|i| format!("<|extra{i}|>")).collect(),
    }
}

fn build(with_roles: bool) -> Tokenizer {
    let roles = if with_roles {
        roles_full()
    } else {
        // 只差 `roles` 一处：前/后处理、模型、解码器必须完全同构，否则测到的是装配差异，
        // 不是 roles 的开销。
        TokenRoles::default()
    };
    let cfg = bert_like(demo_vocab(), roles);
    Tokenizer::from_config(&cfg).expect("装配 tokenizer")
}

/// 第三组对照：**不带 roles**，但把角色表里的 token 以普通后加 token 登记进去。
/// 若它与「无角色」的差距 ≈「带角色」的差距，说明开销来自「后加 token 数量」而非 roles 结构。
fn build_with_plain_added() -> Tokenizer {
    let mut tk = build(false);
    let tokens: Vec<tok_pipeline::core::AddedToken> = roles_full()
        .all()
        .iter()
        .map(|t| tok_pipeline::core::AddedToken::new(t.clone(), true))
        .collect();
    tk.register_added(&tokens).expect("登记后加 token");
    tk
}

/// 第四组对照：只有 5 个基础角色（无 bos/eos/additional），用来看开销是否随角色数线性增长。
fn build_min_roles() -> Tokenizer {
    let roles = TokenRoles {
        unk_token: Some("[UNK]".to_string()),
        pad_token: Some("[PAD]".to_string()),
        cls_token: Some("[CLS]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        mask_token: Some("[MASK]".to_string()),
        ..TokenRoles::default()
    };
    Tokenizer::from_config(&bert_like(demo_vocab(), roles)).expect("装配 tokenizer")
}

/// 200 词一段，混合长短，尽量贴近真实语料的切分压力。
fn sample_text() -> Vec<String> {
    let mut out = Vec::new();
    for round in 0..16 {
        let mut line = String::new();
        for i in 0..200 {
            line.push_str(&format!("w{}", (round * 197 + i * 7) % 800));
            line.push(if i % 9 == 8 { '.' } else { ' ' });
        }
        out.push(line);
    }
    out
}

fn bench(tk: &Tokenizer, texts: &[String], repeat: usize) -> (f64, usize, usize) {
    // 预热：让分支预测/分配器进入稳态，避免把首次调用的一次性开销算进来
    for t in texts.iter().take(2) {
        let _ = tk.encode(t, true).expect("encode");
    }
    let mut tokens = 0usize;
    let mut bytes = 0usize;
    let start = Instant::now();
    for _ in 0..repeat {
        for t in texts {
            let enc = tk.encode(t, true).expect("encode");
            tokens += enc.ids.len();
            bytes += t.len();
        }
    }
    (start.elapsed().as_secs_f64(), tokens, bytes)
}

fn main() {
    let repeat = std::env::var("REPEAT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(40);
    let rounds = std::env::var("ROUNDS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3);
    let texts = sample_text();

    println!(
        "角色链路吞吐复核（release，repeat={repeat}，rounds={rounds}，{} 段文本）",
        texts.len()
    );
    println!(
        "{:<14}{:>12}{:>14}{:>14}{:>12}",
        "装配", "tokens", "tokens/s", "MB/s", "ns/token"
    );

    // 四套装配各建一次，交替测多轮取**最好**成绩：单次测量受 CPU 频率/缓存抖动影响很大，
    // 交替跑能把「谁先跑谁占便宜」的顺序偏置摊平。
    let cases = [
        ("无角色", build(false)),
        ("最小角色(5)", build_min_roles()),
        ("无角色+12后加", build_with_plain_added()),
        ("带角色(E6)", build(true)),
    ];
    let mut best: Vec<Option<(f64, usize, usize)>> = vec![None; cases.len()];
    for _ in 0..rounds {
        for (slot, (_, tk)) in cases.iter().enumerate() {
            let run = bench(tk, &texts, repeat);
            let tps = run.1 as f64 / run.0;
            let better = best[slot].is_none_or(|b| tps > b.1 as f64 / b.0);
            if better {
                best[slot] = Some(run);
            }
        }
    }

    let mut results = Vec::new();
    for (slot, (name, _)) in cases.iter().enumerate() {
        let (secs, tokens, bytes) = best[slot].expect("至少测一轮");
        let tps = tokens as f64 / secs;
        let mbps = bytes as f64 / secs / (1024.0 * 1024.0);
        let ns = secs * 1e9 / tokens as f64;
        println!("{name:<14}{tokens:>12}{tps:>14.0}{mbps:>14.2}{ns:>12.1}");
        results.push((name, tps, mbps, ns));
    }

    let (base, with) = (results[0], results[3]);
    let plain_added = results[2];
    let delta = (with.1 - base.1) / base.1 * 100.0;
    println!("\n带角色相对无角色吞吐差异：{delta:+.2}%（正为更快）");
    println!(
        "无角色+12后加 相对无角色差异：{:+.2}%（用来区分「roles 结构」与「后加 token 数量」）",
        (plain_added.1 - base.1) / base.1 * 100.0
    );
    // 单机噪声通常 ±5%；超过 15% 说明 roles 真的进了热路径，需要回查。
    if delta.abs() > 15.0 {
        println!("结论：差异超出噪声区间，roles 可能漏进 encode 热路径。");
        std::process::exit(1);
    }
    println!("结论：差异在噪声区间内，roles 未进入 encode 热路径。");
}
