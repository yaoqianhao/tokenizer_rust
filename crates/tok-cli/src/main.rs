//! `tok` — tokenizer 流水线命令行入口。
//!
//! 子命令：
//! * `train`   ：语料 → 训练 → 导出 HF fast tokenizer 三件套（tokenizer.json / tokenizer_config.json / special_tokens_map.json）
//! * `encode`  ：文本 → id / token / offset / 特殊位掩码
//! * `decode`  ：id 序列 → 文本
//! * `inspect` ：体检 tokenizer.json 结构摘要 + 目录产物清单
//! * `compare` ：两个 tokenizer 逐条比对编码 id（自洽校验 / 跨版本回归）
//! * `eval`    ：语料级量化评测（tok/char、UNK 率、round-trip 率、吞吐）

mod commands;

use clap::{Parser, Subcommand};
use tok_core::Result;

#[derive(Debug, Parser)]
#[command(
    name = "tok",
    version,
    about = "Rust tokenizer 流水线 CLI（L1 normalizer → L2 pre_tokenizer → L3 model → L5 post_processor → L6 decoder）"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// 训练：语料 → HF fast tokenizer 三件套
    // Train 参数最多（模型 / 预分词 / 后处理 / 解码器 / 词表），装箱避免拖大整个枚举
    Train(Box<commands::TrainArgs>),
    /// 编码：文本 → id 序列
    Encode(commands::EncodeArgs),
    /// 解码：id 序列 → 文本
    Decode(commands::DecodeArgs),
    /// 体检：tokenizer.json 结构摘要与目录产物清单
    Inspect(commands::InspectArgs),
    /// 比对：两个 tokenizer 对同一批文本的编码 id 是否一致
    Compare(commands::CompareArgs),
    /// 语料级评测：tok/char、UNK 率、round-trip 率与吞吐
    Eval(commands::EvalArgs),
}

fn main() {
    let cli = Cli::parse();
    let outcome: Result<()> = match cli.command {
        Command::Train(args) => commands::run_train(*args),
        Command::Encode(args) => commands::run_encode(args),
        Command::Decode(args) => commands::run_decode(args),
        Command::Inspect(args) => commands::run_inspect(args),
        Command::Compare(args) => commands::run_compare(args),
        Command::Eval(args) => commands::run_eval(args),
    };
    if let Err(err) = outcome {
        eprintln!("错误: {err}");
        std::process::exit(1);
    }
}
