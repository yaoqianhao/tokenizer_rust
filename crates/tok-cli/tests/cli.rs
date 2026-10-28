//! `tok` 命令行集成测试：真的拉起二进制进程，覆盖 train / encode / decode / inspect / compare。
//!
//! 测试约定的边界：
//!
//! * 只断言**进程可观测行为**（退出码、stdout 结构、stderr 是否报错），不读内部私有函数；
//! * 期望值与 `tok-pipeline` 门面在同一份产物上的结果交叉对照，避免"CLI 自己和自己比"；
//! * 报错文案只断言"以 `错误:` 开头且退出码为 1"这类稳定契约，不断言系统 locale
//!   相关文本（`io error` 的中文描述随系统语言变化）；
//! * 每个用例独占一个临时沙箱目录，`Drop` 时清理。

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

/// 多行多语言语料：覆盖英文、中文、emoji、重音，保证 BPE 能学到可用的合并。
const CORPUS: &str = "\
hello world this is a tokenizer corpus
你好世界 这是一个测试语料
emoji 😀🎉 mixed with café naïve résumé
the quick brown fox jumps over the lazy dog
hello world again and again
中文English混排test123数字456
tokenization pipeline 负责把文本切成子词
落盘 tokenizer.json 之后可以被 HF 直接加载
sentence with  multiple   spaces and\ttab
最后一行的中文语料
";

const SAMPLE_TEXT: &str = "hello world 你好";

fn tok_bin() -> &'static str {
    env!("CARGO_BIN_EXE_tok-cli")
}

static SEQ: AtomicUsize = AtomicUsize::new(0);

/// 临时沙箱：每个用例一个独立目录，用例结束自动删除。
struct Sandbox {
    dir: PathBuf,
}

impl Sandbox {
    fn new(tag: &str) -> Self {
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("tok-cli-it-{}-{}-{}", std::process::id(), tag, n));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("创建沙箱目录");
        Self { dir }
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.dir.join(rel)
    }

    fn write(&self, rel: &str, content: &str) -> PathBuf {
        let p = self.path(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("创建父目录");
        }
        std::fs::write(&p, content).expect("写入测试文件");
        p
    }

    /// 训练一份最小 BPE 产物，返回输出目录（目录内应有 HF 三件套）。
    fn train_bpe(&self, name: &str, vocab_size: usize) -> PathBuf {
        let corpus = self.write(&format!("{name}-corpus.txt"), CORPUS);
        let out = self.path(name);
        let r = Command::new(tok_bin())
            .arg("train")
            .arg("--corpus")
            .arg(&corpus)
            .arg("--out")
            .arg(&out)
            .arg("--vocab-size")
            .arg(vocab_size.to_string())
            .arg("--special")
            .arg("[UNK]")
            .arg("[PAD]")
            .output()
            .expect("运行 tok train");
        assert!(
            r.status.success(),
            "tok train 失败（stdout={} stderr={}）",
            String::from_utf8_lossy(&r.stdout),
            String::from_utf8_lossy(&r.stderr)
        );
        out
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn run<I, S>(args: I) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    Command::new(tok_bin())
        .args(args)
        .output()
        .expect("运行 tok")
}

fn stdout_of(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn stderr_of(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

/// 断言失败退出且 stderr 走统一错误通道。
fn assert_fails(o: &Output, hint: &str) {
    assert!(!o.status.success(), "{hint}：应以非零码退出");
    let err = stderr_of(o);
    assert!(
        err.contains("错误:"),
        "{hint}：stderr 应以 `错误:` 开头，实际为 {err:?}"
    );
}

fn json_stdout(o: &Output) -> Value {
    let text = stdout_of(o);
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("stdout 不是合法 JSON（{e}）: {text:.400}"))
}

// ------------------------------------------------------------------ train

#[test]
fn train_writes_hf_triplet_that_pipeline_can_load() {
    let sb = Sandbox::new("train");
    let out = sb.train_bpe("art", 300);

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        let p = out.join(name);
        assert!(p.is_file(), "训练后应落盘 {name}");
        let raw = std::fs::read_to_string(&p).expect("读取产物");
        serde_json::from_str::<Value>(&raw).unwrap_or_else(|e| panic!("{name} 不是合法 JSON: {e}"));
    }

    // 产物必须能被门面重新加载并真正编码——这是 CLI 与库之间最重要的契约。
    let tk = tok_pipeline::load_tokenizer_json(&out.join("tokenizer.json")).expect("加载产物");
    let enc = tk.encode(SAMPLE_TEXT, true).expect("编码");
    assert!(!enc.ids.is_empty(), "重新加载后应能编码出 id");
    assert_eq!(enc.ids.len(), enc.tokens.len());
    assert_eq!(enc.ids.len(), enc.offsets.len());
    assert!(enc.validate().is_ok(), "编码结果应通过内部一致性校验");

    // 落盘清单必须列出真实路径（report_json 默认开启）。
    let stdout = stdout_of(&run([
        "inspect",
        "--tokenizer",
        out.to_str().expect("utf8 路径"),
    ]));
    assert!(stdout.contains("tokenizer.json"), "inspect 应报告产物清单");
}

#[test]
fn train_smoke_prints_self_check() {
    let sb = Sandbox::new("smoke");
    let corpus = sb.write("corpus.txt", CORPUS);
    let out = sb.path("smoke-art");
    let o = run([
        "train",
        "--corpus",
        corpus.to_str().expect("utf8"),
        "--out",
        out.to_str().expect("utf8"),
        "--vocab-size",
        "300",
        "--special",
        "[UNK]",
        "--smoke",
    ]);
    assert!(
        o.status.success(),
        "train --smoke 应成功: {}",
        stderr_of(&o)
    );
    let stdout = stdout_of(&o);
    assert!(stdout.contains("训练完成"), "应打印训练报告");
    assert!(stdout.contains("冒烟自检"), "应打印冒烟自检");
    assert!(
        stdout.contains("冒烟自检：decode(ids)="),
        "冒烟自检应包含 decode 结果"
    );
}

// ------------------------------------------------------------------ encode

#[test]
fn encode_json_matches_pipeline_output() {
    let sb = Sandbox::new("encode");
    let out = sb.train_bpe("art", 300);
    let tk = tok_pipeline::load_tokenizer_json(&out.join("tokenizer.json")).expect("加载产物");
    let expect = tk.encode(SAMPLE_TEXT, true).expect("库编码");

    let o = run([
        "encode",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--text",
        SAMPLE_TEXT,
        "--json",
    ]);
    assert!(o.status.success(), "encode 应成功: {}", stderr_of(&o));
    let v = json_stdout(&o);
    let arr = v.as_array().expect("--json 应输出数组");
    assert_eq!(arr.len(), 1, "单条 --text 也应输出长度 1 的数组");
    let item = &arr[0];

    assert_eq!(item["text"].as_str(), Some(SAMPLE_TEXT));
    let ids: Vec<u32> = item["ids"]
        .as_array()
        .expect("ids 数组")
        .iter()
        .map(|x| x.as_u64().expect("id 是整数") as u32)
        .collect();
    let tokens: Vec<String> = item["tokens"]
        .as_array()
        .expect("tokens 数组")
        .iter()
        .map(|x| x.as_str().expect("token 是字符串").to_string())
        .collect();
    let offsets: Vec<(usize, usize)> = item["offsets"]
        .as_array()
        .expect("offsets 数组")
        .iter()
        .map(|p| {
            let pair = p.as_array().expect("offset 是二元数组");
            (
                pair[0].as_u64().expect("offset start") as usize,
                pair[1].as_u64().expect("offset end") as usize,
            )
        })
        .collect();

    assert_eq!(ids, expect.ids, "CLI 的 ids 应与库一致");
    assert_eq!(tokens, expect.tokens, "CLI 的 tokens 应与库一致");
    assert_eq!(offsets, expect.offsets, "CLI 的 offsets 应与库一致");

    // 三个长度必须对齐，否则下游 offsets 会错位。
    for key in ["type_ids", "attention_mask", "special_tokens_mask"] {
        let len = item[key].as_array().expect("数组字段").len();
        assert_eq!(len, ids.len(), "{key} 长度应与 ids 一致");
    }

    // 偏移必须落在原文的字符范围内且不倒退。
    let chars = SAMPLE_TEXT.chars().count();
    let mut prev = 0usize;
    for &(s, e) in &offsets {
        assert!(s <= e && e <= chars, "偏移越界: ({s}, {e}) > {chars}");
        assert!(s >= prev, "偏移起点倒退: {s} < {prev}");
        prev = s;
    }
}

#[test]
fn encode_no_special_skips_post_processing() {
    let sb = Sandbox::new("encode-nospecial");
    let out = sb.train_bpe("art", 300);
    let tk = tok_pipeline::load_tokenizer_json(&out.join("tokenizer.json")).expect("加载产物");
    let expect = tk.encode(SAMPLE_TEXT, false).expect("库编码");

    let o = run([
        "encode",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--text",
        SAMPLE_TEXT,
        "--no-special",
        "--json",
    ]);
    assert!(
        o.status.success(),
        "encode --no-special 应成功: {}",
        stderr_of(&o)
    );
    let v = json_stdout(&o);
    let ids: Vec<u32> = v[0]["ids"]
        .as_array()
        .expect("ids 数组")
        .iter()
        .map(|x| x.as_u64().expect("id") as u32)
        .collect();
    assert_eq!(
        ids, expect.ids,
        "--no-special 应与库的 add_special=false 一致"
    );
}

#[test]
fn encode_pair_concatenates_both_segments() {
    let sb = Sandbox::new("encode-pair");
    let out = sb.train_bpe("art", 300);
    let tk = tok_pipeline::load_tokenizer_json(&out.join("tokenizer.json")).expect("加载产物");
    let first = "hello world";
    let second = "你好世界";
    let expect = tk.encode_pair(first, second, true).expect("库 encode_pair");

    let o = run([
        "encode",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--text",
        first,
        "--pair",
        second,
        "--json",
    ]);
    assert!(
        o.status.success(),
        "encode --pair 应成功: {}",
        stderr_of(&o)
    );
    let v = json_stdout(&o);
    let ids: Vec<u32> = v[0]["ids"]
        .as_array()
        .expect("ids 数组")
        .iter()
        .map(|x| x.as_u64().expect("id") as u32)
        .collect();
    assert_eq!(ids, expect.ids, "--pair 应与库 encode_pair 一致");
    assert!(
        ids.len() > 1,
        "pair 编码结果应包含两段内容（实际 {} 个 id）",
        ids.len()
    );
}

#[test]
fn encode_from_file_emits_one_entry_per_line() {
    let sb = Sandbox::new("encode-file");
    let out = sb.train_bpe("art", 300);
    let file = sb.write("inputs.txt", "hello world\n你好世界\n\nthird line\n");

    let o = run([
        "encode",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--file",
        file.to_str().expect("utf8"),
    ]);
    assert!(
        o.status.success(),
        "encode --file 应成功: {}",
        stderr_of(&o)
    );
    let v = json_stdout(&o);
    let arr = v.as_array().expect("多行输入应输出数组");
    let texts: Vec<&str> = arr
        .iter()
        .map(|x| x["text"].as_str().expect("text"))
        .collect();
    // 空行也会被 lines() 保留为一条空文本，这里只校验条数与内容顺序。
    assert_eq!(texts.len(), 4, "每行一条（含空行）: {texts:?}");
    assert_eq!(texts[0], "hello world");
    assert_eq!(texts[1], "你好世界");
    assert_eq!(texts[2], "");
    assert_eq!(texts[3], "third line");
}

#[test]
fn encode_without_input_reports_error() {
    let sb = Sandbox::new("encode-noinput");
    let out = sb.train_bpe("art", 300);
    let o = run(["encode", "--tokenizer", out.to_str().expect("utf8")]);
    assert_fails(&o, "缺少 --text / --file");
    assert!(
        stderr_of(&o).contains("--text") && stderr_of(&o).contains("--file"),
        "报错应提示需要 --text 或 --file，实际: {}",
        stderr_of(&o)
    );
}

// ------------------------------------------------------------------ decode

#[test]
fn decode_roundtrips_encode() {
    let sb = Sandbox::new("decode");
    let out = sb.train_bpe("art", 300);
    let text = "hello world";

    let enc = run([
        "encode",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--text",
        text,
        "--json",
    ]);
    assert!(enc.status.success(), "encode 应成功: {}", stderr_of(&enc));
    let v = json_stdout(&enc);
    let ids: Vec<u32> = v[0]["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|x| x.as_u64().expect("id") as u32)
        .collect();
    let ids_arg = ids
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let dec = run([
        "decode",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--ids",
        &ids_arg,
    ]);
    assert!(dec.status.success(), "decode 应成功: {}", stderr_of(&dec));
    assert_eq!(
        stdout_of(&dec).trim_end(),
        text,
        "byte-level BPE 应能无损往返"
    );
}

#[test]
fn decode_rejects_illegal_id() {
    let sb = Sandbox::new("decode-bad");
    let out = sb.train_bpe("art", 300);
    let o = run([
        "decode",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--ids",
        "1,abc",
    ]);
    assert_fails(&o, "非法 id");
    assert!(
        stderr_of(&o).contains("abc"),
        "报错应指出非法片段，实际: {}",
        stderr_of(&o)
    );
}

// ------------------------------------------------------------------ inspect

#[test]
fn inspect_reports_structure_and_inventory() {
    let sb = Sandbox::new("inspect");
    let out = sb.train_bpe("art", 300);

    let o = run([
        "inspect",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--head",
        "5",
    ]);
    assert!(o.status.success(), "inspect 应成功: {}", stderr_of(&o));
    let stdout = stdout_of(&o);
    assert!(stdout.contains("\"model_type\": \"BPE\""), "应报告模型类型");
    assert!(
        stdout.contains("\"has_pre_tokenizer\": true"),
        "应报告预分词器存在"
    );
    assert!(
        stdout.contains("\"added_tokens\": 2"),
        "应统计 added_tokens"
    );
    assert!(stdout.contains("目录产物体检"), "应输出目录产物体检");
    assert!(stdout.contains("\"complete\": true"), "三件套应齐全");

    // --head N 打印 N 行 `id  token`
    let head_lines = stdout
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
        })
        .count();
    assert_eq!(head_lines, 5, "--head 5 应打印 5 行词表条目");

    let v: Value = serde_json::from_str(stdout.split("\n目录产物体检").next().expect("首段 JSON"))
        .expect("inspect 主段应是合法 JSON");
    assert_eq!(v["model_type"].as_str(), Some("BPE"));
    assert!(
        v["vocab_size"].as_u64().unwrap_or(0) >= 256,
        "BPE 词表应至少覆盖 256 个字节 token"
    );
}

#[test]
fn inspect_missing_tokenizer_reports_error() {
    let sb = Sandbox::new("inspect-missing");
    let missing = sb.path("nope");
    let o = run(["inspect", "--tokenizer", missing.to_str().expect("utf8")]);
    assert_fails(&o, "缺失 tokenizer 路径");
    assert!(stdout_of(&o).is_empty(), "失败时不应向 stdout 输出半截报告");
}

// ------------------------------------------------------------------ compare

#[test]
fn compare_identical_tokenizers_match() {
    let sb = Sandbox::new("compare-same");
    let out = sb.train_bpe("art", 300);
    let o = run([
        "compare",
        "--left",
        out.to_str().expect("utf8"),
        "--right",
        out.to_str().expect("utf8"),
    ]);
    assert!(
        o.status.success(),
        "compare 同一产物应成功: {}",
        stderr_of(&o)
    );
    let stdout = stdout_of(&o);
    // 统计 JSON 是最后一段以 `{` 独占一行的 pretty 输出，前面是逐条 OK/DIFF。
    let start = stdout
        .rfind("\n{")
        .map(|i| i + 1)
        .unwrap_or_else(|| panic!("compare 末尾应输出统计 JSON: {stdout:.400}"));
    let tail: Value = serde_json::from_str(&stdout[start..]).expect("统计 JSON 可解析");
    assert_eq!(tail["mismatched"].as_u64(), Some(0));
    assert_eq!(tail["texts"].as_u64(), Some(7), "缺省应使用 7 条内置样本");
}

#[test]
fn compare_detects_diff_and_strict_mode_fails() {
    let sb = Sandbox::new("compare-diff");
    let left = sb.train_bpe("left", 300);
    let right = sb.train_bpe("right", 120);

    let loose = run([
        "compare",
        "--left",
        left.to_str().expect("utf8"),
        "--right",
        right.to_str().expect("utf8"),
        "--only-diff",
    ]);
    assert!(
        loose.status.success(),
        "未加 --strict 时有差异也应退出 0: {}",
        stderr_of(&loose)
    );
    assert!(
        stdout_of(&loose).contains("DIFF"),
        "词表差异应被检出为 DIFF"
    );

    let strict = run([
        "compare",
        "--left",
        left.to_str().expect("utf8"),
        "--right",
        right.to_str().expect("utf8"),
        "--only-diff",
        "--strict",
    ]);
    assert!(!strict.status.success(), "--strict 遇到差异应以非零码退出");
    assert!(
        stderr_of(&strict).contains("不一致"),
        "--strict 报错应说明文本编码不一致，实际: {}",
        stderr_of(&strict)
    );
}
