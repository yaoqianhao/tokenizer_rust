//! `tok` — tokenizer 流水线命令行入口。
//!
//! 子命令：
//! * `train`   ：语料 → 训练 → 导出 HF fast tokenizer 三件套（tokenizer.json / tokenizer_config.json / special_tokens_map.json）
//! * `encode`  ：文本 → id / token / offset / 特殊位掩码
//! * `decode`  ：id 序列 → 文本
//! * `inspect` ：体检 tokenizer.json 结构摘要 + 目录产物清单

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
    Train(commands::TrainArgs),
    /// 编码：文本 → id 序列
    Encode(commands::EncodeArgs),
    /// 解码：id 序列 → 文本
    Decode(commands::DecodeArgs),
    /// 体检：tokenizer.json 结构摘要与目录产物清单
    Inspect(commands::InspectArgs),
}

fn main() {
    let cli = Cli::parse();
    let outcome: Result<()> = match cli.command {
        Command::Train(args) => commands::run_train(args),
        Command::Encode(args) => commands::run_encode(args),
        Command::Decode(args) => commands::run_decode(args),
        Command::Inspect(args) => commands::run_inspect(args),
    };
    if let Err(err) = outcome {
        eprintln!("错误: {err}");
        std::process::exit(1);
    }
}
