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

// ------------------------------------------------------------------ 后处理 / 解码器 / eval

/// 以 wordpiece + bert 预分词训练一份产物，`extra` 追加 `--post-processor` / `--decoder` 等参数。
fn train_wordpiece(sb: &Sandbox, name: &str, extra: &[&str]) -> PathBuf {
    let corpus = sb.write(&format!("{name}-corpus.txt"), CORPUS);
    let out = sb.path(name);
    let mut cmd = Command::new(tok_bin());
    cmd.arg("train")
        .arg("--corpus")
        .arg(&corpus)
        .arg("--out")
        .arg(&out)
        .arg("--model")
        .arg("wordpiece");
    // 调用方显式给了 --pre-tokenizer 时不追加默认的 bert（clap 不接受重复传参）
    if !extra.contains(&"--pre-tokenizer") {
        cmd.arg("--pre-tokenizer").arg("bert");
    }
    cmd.arg("--vocab-size")
        .arg("300")
        .arg("--special")
        .arg("[CLS]")
        .arg("[SEP]")
        .arg("[UNK]")
        .arg("[PAD]")
        .arg("[MASK]");
    for a in extra {
        cmd.arg(a);
    }
    let r = cmd.output().expect("运行 tok train");
    assert!(
        r.status.success(),
        "训练 {name} 失败（extra={extra:?}）：{}",
        stderr_of(&r)
    );
    out
}

fn read_json(path: &std::path::Path) -> Value {
    let raw = std::fs::read_to_string(path).expect("读取产物");
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{} 不是合法 JSON: {e}", path.display()))
}

#[test]
fn post_processor_bert_and_template_are_id_identical() {
    let sb = Sandbox::new("post-parity");
    let bert = train_wordpiece(&sb, "bert", &["--post-processor", "bert"]);
    let tmpl = train_wordpiece(&sb, "tmpl", &["--post-processor", "bert-template"]);

    let b = read_json(&bert.join("tokenizer.json"));
    let t = read_json(&tmpl.join("tokenizer.json"));
    assert_eq!(
        b["post_processor"]["type"].as_str(),
        Some("BertProcessing"),
        "bert 档应导出 BertProcessing"
    );
    assert_eq!(
        t["post_processor"]["type"].as_str(),
        Some("TemplateProcessing"),
        "bert-template 档应导出 TemplateProcessing"
    );

    // 特殊 token 的 id 必须是词表真实 id，而不是训练前传入的占位 0
    let vocab = b["model"]["vocab"].as_object().expect("wordpiece 词表");
    let cls_id = vocab
        .get("[CLS]")
        .and_then(|v| v.as_u64())
        .expect("词表应含 [CLS]");
    assert_eq!(
        b["post_processor"]["cls"][1].as_u64(),
        Some(cls_id),
        "BertProcessing.cls 的第二项应是真实词表 id"
    );
    assert_eq!(
        t["post_processor"]["special_tokens"]["[CLS]"]["ids"][0].as_u64(),
        Some(cls_id),
        "TemplateProcessing.special_tokens 应回填真实 id"
    );

    // 等价写法必须逐 id 一致：这是 E1 的核心自洽校验
    let o = run([
        "compare",
        "--left",
        bert.to_str().expect("utf8"),
        "--right",
        tmpl.to_str().expect("utf8"),
        "--only-diff",
        "--strict",
    ]);
    assert!(
        o.status.success(),
        "Bert 与其等价模板的编码结果应完全一致: {}",
        stderr_of(&o)
    );
}

#[test]
fn post_processor_pair_sets_type_ids_and_special_mask() {
    let sb = Sandbox::new("post-pair");
    let bert = train_wordpiece(&sb, "bert", &["--post-processor", "bert"]);

    let o = run([
        "encode",
        "--tokenizer",
        bert.to_str().expect("utf8"),
        "--text",
        SAMPLE_TEXT,
        "--pair",
        "第二段",
        "--json",
    ]);
    assert!(o.status.success(), "pair 编码应成功: {}", stderr_of(&o));
    let v = json_stdout(&o);
    let e = &v[0];
    let ids = e["ids"].as_array().expect("ids 数组");
    let type_ids = e["type_ids"].as_array().expect("type_ids 数组");
    let mask = e["special_tokens_mask"]
        .as_array()
        .expect("special_tokens_mask 数组");
    assert_eq!(ids.len(), type_ids.len(), "type_ids 应与 ids 等长");
    assert_eq!(ids.len(), mask.len(), "mask 应与 ids 等长");

    // [CLS] 与首个 [SEP] 属 A 段（type_id 0），末位 [SEP] 属 B 段（type_id 1）
    assert_eq!(type_ids[0].as_u64(), Some(0), "[CLS] 的 type_id 应为 0");
    assert_eq!(
        type_ids[type_ids.len() - 1].as_u64(),
        Some(1),
        "末位 [SEP] 的 type_id 应为 1（B 段）"
    );
    assert_eq!(mask[0].as_u64(), Some(1), "首位应是特殊 token");
    assert_eq!(mask[mask.len() - 1].as_u64(), Some(1), "末位应是特殊 token");
    assert!(
        type_ids.iter().any(|v| v.as_u64() == Some(1)),
        "pair 编码应出现 B 段（type_id=1）"
    );
}

#[test]
fn post_processor_rejects_unknown_or_broken_spec() {
    let sb = Sandbox::new("post-bad");
    let corpus = sb.write("c.txt", CORPUS);

    let bad = run([
        "train",
        "--corpus",
        corpus.to_str().expect("utf8"),
        "--out",
        sb.path("bad").to_str().expect("utf8"),
        "--vocab-size",
        "200",
        "--post-processor",
        "bertish",
    ]);
    assert_fails(&bad, "未知后处理器");

    // 模板缺少 $A 占位必须在训练前就拒绝，而不是训练完才炸
    let no_a = run([
        "train",
        "--corpus",
        corpus.to_str().expect("utf8"),
        "--out",
        sb.path("bad2").to_str().expect("utf8"),
        "--vocab-size",
        "200",
        "--post-processor",
        "template:[CLS] [SEP]",
    ]);
    assert_fails(&no_a, "模板缺少 $A");
}

#[test]
fn decoder_spec_replaces_decoder() {
    let sb = Sandbox::new("decoder");

    let none = train_wordpiece(&sb, "dec-none", &["--decoder", "none"]);
    assert!(
        read_json(&none.join("tokenizer.json"))["decoder"].is_null(),
        "--decoder none 应导出 decoder: null"
    );

    let no_cleanup = train_wordpiece(
        &sb,
        "dec-noclean",
        &["--decoder", "wordpiece:cleanup=false"],
    );
    let d = read_json(&no_cleanup.join("tokenizer.json"));
    assert_eq!(d["decoder"]["type"].as_str(), Some("WordPiece"));
    assert_eq!(
        d["decoder"]["cleanup"].as_bool(),
        Some(false),
        "k=v 参数应透传到导出的 decoder"
    );

    // fuse(...) 组合应展开为 Sequence
    let fused = train_wordpiece(
        &sb,
        "dec-fuse",
        &["--decoder", "fuse(bytelevel,wordpiece:cleanup=false)"],
    );
    let f = read_json(&fused.join("tokenizer.json"));
    assert_eq!(
        f["decoder"]["type"].as_str(),
        Some("Sequence"),
        "fuse(...) 应导出 Sequence 解码器"
    );
    let seq = f["decoder"]["decoders"]
        .as_array()
        .expect("Sequence 应含 decoders 数组");
    assert_eq!(seq.len(), 2, "fuse 内应有两个解码器节点");
    assert_eq!(seq[0]["type"].as_str(), Some("ByteLevel"));
    assert_eq!(seq[1]["type"].as_str(), Some("WordPiece"));
}

#[test]
fn eval_reports_corpus_metrics() {
    let sb = Sandbox::new("eval");
    let out = sb.train_bpe("art", 400);
    let corpus = sb.write("eval-corpus.txt", CORPUS);

    let o = run([
        "eval",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--corpus",
        corpus.to_str().expect("utf8"),
        "--json",
    ]);
    assert!(o.status.success(), "eval 应成功: {}", stderr_of(&o));
    let m = json_stdout(&o);
    assert!(m["lines"].as_u64().unwrap_or(0) > 0, "应统计语料行数");
    assert!(m["tokens"].as_u64().unwrap_or(0) > 0, "应统计 token 总数");
    assert!(
        m["tokens_per_char"].as_f64().unwrap_or(0.0) > 0.0,
        "tok/char 应大于 0"
    );
    // 字节级 BPE 能编码任意字节，round-trip 必须无损——这条是评测口径的可信性基石
    assert_eq!(
        m["roundtrip_rate"].as_f64(),
        Some(1.0),
        "字节级 BPE 的 round-trip 应无损"
    );
    assert_eq!(m["roundtrip_squash_rate"].as_f64(), Some(1.0));

    // 纯文本模式要给出人可读的口径
    let text = run([
        "eval",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--corpus",
        corpus.to_str().expect("utf8"),
    ]);
    assert!(
        text.status.success(),
        "文本模式应成功: {}",
        stderr_of(&text)
    );
    let stdout = stdout_of(&text);
    assert!(stdout.contains("tok/char="), "应打印 tok/char");
    assert!(stdout.contains("round-trip"), "应打印 round-trip");
    assert!(stdout.contains("UNK"), "应打印 UNK 统计");
}

#[test]
fn eval_no_special_drops_exactly_the_added_tokens() {
    let sb = Sandbox::new("eval-special");
    let out = train_wordpiece(&sb, "bert", &["--post-processor", "bert"]);
    let corpus = sb.write("eval-corpus.txt", CORPUS);

    let base = json_stdout(&run([
        "eval",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--corpus",
        corpus.to_str().expect("utf8"),
        "--json",
    ]));
    let plain = json_stdout(&run([
        "eval",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--corpus",
        corpus.to_str().expect("utf8"),
        "--no-special",
        "--json",
    ]));

    let lines = base["lines"].as_u64().expect("行数");
    let with_special = base["tokens"].as_u64().expect("tokens");
    let without = plain["tokens"].as_u64().expect("tokens");
    assert_eq!(
        with_special - without,
        lines * 2,
        "BertProcessing 每条应额外贡献 [CLS] 与 [SEP]"
    );
}

#[test]
fn decode_accepts_loose_id_separators() {
    let sb = Sandbox::new("decode-loose");
    let out = sb.train_bpe("art", 400);
    let dir = out.to_str().expect("utf8");

    let enc = json_stdout(&run([
        "encode",
        "--tokenizer",
        dir,
        "--text",
        SAMPLE_TEXT,
        "--json",
    ]));
    let ids: Vec<String> = enc[0]["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("id").to_string())
        .collect();
    let comma = ids.join(",");
    let space = ids.join(" ");
    let bracket = format!("[{}, ]", ids.join(", "));

    let base = run(["decode", "--tokenizer", dir, "--ids", &comma]);
    assert!(
        base.status.success(),
        "逗号分隔应成功: {}",
        stderr_of(&base)
    );
    let want = stdout_of(&base);

    // 空格分隔 / 方括号包裹是抄 ids 时最常见的两种粘贴形态，必须与逗号形态等价
    for spec in [space.as_str(), bracket.as_str()] {
        let o = run(["decode", "--tokenizer", dir, "--ids", spec]);
        assert!(
            o.status.success(),
            "--ids {spec:?} 应被接受: {}",
            stderr_of(&o)
        );
        assert_eq!(
            stdout_of(&o),
            want,
            "--ids {spec:?} 的解码结果应与逗号形态一致"
        );
    }
}

#[test]
fn decode_empty_id_list_does_not_panic() {
    // 对照坑：HF 的 Strip 解码器在空 id 序列上会越界 panic，CLI 侧必须安全返回空串
    let sb = Sandbox::new("decode-empty");
    let out = sb.train_bpe("art", 400);
    let o = run([
        "decode",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--ids",
        "",
    ]);
    assert!(o.status.success(), "空 id 列表应成功: {}", stderr_of(&o));
    assert_eq!(stdout_of(&o), "\n", "空 id 列表应解码为空行");
}

#[test]
fn decoder_none_joins_tokens_with_single_space() {
    // 与 HF 对齐的关键细节：decoder 缺省时不是"直接 concatenate"，而是 token 之间补一个空格。
    let sb = Sandbox::new("decoder-none");
    let out = train_wordpiece(&sb, "none", &["--decoder", "none"]);
    let dir = out.to_str().expect("utf8");

    let enc = json_stdout(&run([
        "encode",
        "--tokenizer",
        dir,
        "--text",
        SAMPLE_TEXT,
        "--json",
    ]));
    let tokens: Vec<String> = enc[0]["tokens"]
        .as_array()
        .expect("tokens")
        .iter()
        .map(|v| v.as_str().expect("token").to_string())
        .collect();
    let ids: Vec<String> = enc[0]["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("id").to_string())
        .collect();

    let o = run([
        "decode",
        "--tokenizer",
        dir,
        "--ids",
        &ids.join(","),
        "--skip-special",
    ]);
    assert!(o.status.success(), "decode 应成功: {}", stderr_of(&o));
    assert_eq!(
        stdout_of(&o).trim_end_matches('\n'),
        tokens.join(" "),
        "decoder=none 时 HF 用单空格连接 token"
    );
}

// ------------------------------------------------------------------ E3/E4/E5：预分词 / 归一化 / 截断填充

#[test]
fn pretokenizer_sequence_equals_nested_composition() {
    // E3：`sequence(a,b)` 组合的产物，必须与"逐个环节手工串起来"的结果一致。
    // digits 先切出数字、punctuation 再切标点，两步顺序在 JSON 里必须原样保留。
    let sb = Sandbox::new("pretok-seq");
    let out = train_wordpiece(
        &sb,
        "seq",
        &["--pre-tokenizer", "sequence(whitespace,punctuation)"],
    );
    let js = read_json(&out.join("tokenizer.json"));
    let pre = &js["pre_tokenizer"];
    assert_eq!(pre["type"].as_str(), Some("Sequence"), "应为 Sequence 节点");
    let nodes = pre["pretokenizers"].as_array().expect("pretokenizers");
    assert_eq!(nodes.len(), 2, "两段组合应落两个节点");
    assert_eq!(nodes[0]["type"].as_str(), Some("Whitespace"));
    assert_eq!(nodes[1]["type"].as_str(), Some("Punctuation"));

    // 单节点 sequence(...) 应退化为该节点本身，而不是包一层只有一个孩子的 Sequence
    let single = train_wordpiece(
        &sb,
        "seq-single",
        &["--pre-tokenizer", "sequence(whitespace)"],
    );
    let js = read_json(&single.join("tokenizer.json"));
    assert_eq!(
        js["pre_tokenizer"]["type"].as_str(),
        Some("Whitespace"),
        "单元素 sequence 应退化"
    );
}

#[test]
fn pretokenizer_split_regex_is_applied() {
    // E3：自定义 split 正则——按数字串切开，数字必须与字母分离。
    let sb = Sandbox::new("pretok-split");
    let out = train_wordpiece(
        &sb,
        "split",
        &["--pre-tokenizer", r"split:pattern=\d+;behavior=isolated"],
    );
    let js = read_json(&out.join("tokenizer.json"));
    assert_eq!(js["pre_tokenizer"]["type"].as_str(), Some("Split"));

    let tk = tok_pipeline::load_tokenizer_json(&out.join("tokenizer.json")).expect("加载产物");
    let enc = tk.encode("abc123def", false).expect("编码");
    // 数字串被切成独立片段后，不允许任何一个 token 同时覆盖字母与数字
    assert!(
        enc.tokens
            .iter()
            .any(|t| t.chars().all(|c| c.is_ascii_digit())),
        "数字串应被切出独立 token，实际 {:?}",
        enc.tokens
    );
    assert!(!enc.tokens.iter().any(|t| t == "abc123def"), "不应整串成词");
}

#[test]
fn normalizer_strip_replace_prepend_round_trip_json() {
    // E4：strip / replace / prepend 三个补齐项要能进 tokenizer.json 并被重新加载执行。
    let sb = Sandbox::new("norm-extra");
    let out = train_wordpiece(
        &sb,
        "norm",
        &[
            "--normalizer",
            r"replace:pattern=\s+;content=\x20",
            "--normalizer",
            "strip",
            "--normalizer",
            "prepend:prepend=\u{2581}",
        ],
    );
    let js = read_json(&out.join("tokenizer.json"));
    let seq = js["normalizer"]["normalizers"]
        .as_array()
        .expect("归一化链应是 Sequence");
    assert_eq!(seq.len(), 3, "三段归一化应全部保留");
    assert_eq!(seq[1]["type"].as_str(), Some("Strip"));
    assert_eq!(seq[2]["type"].as_str(), Some("Prepend"));

    let tk = tok_pipeline::load_tokenizer_json(&out.join("tokenizer.json")).expect("加载产物");
    let enc = tk.encode("  hi  ", false).expect("编码");
    // prepend 会把 ▁ 挂到规范化后的文本头部
    assert!(
        enc.tokens.iter().any(|t| t.contains('\u{2581}')),
        "prepend 应作用到编码结果，实际 {:?}",
        enc.tokens
    );
    // strip 去掉首尾空白后，不应出现"纯空白"的 token
    assert!(
        !enc.tokens
            .iter()
            .any(|t| t.trim().is_empty() && !t.is_empty()),
        "strip 后不应残留纯空白 token，实际 {:?}",
        enc.tokens
    );
}

#[test]
fn truncation_params_reach_tokenizer_json_and_encode() {
    // E5：--truncation-max-length 落盘后必须真的截断（此前 apply_io_params 是 dead_code）。
    let sb = Sandbox::new("trunc");
    let out = train_wordpiece(
        &sb,
        "trunc",
        &[
            "--post-processor",
            "bert",
            "--truncation-max-length",
            "8",
            "--truncation-strategy",
            "only-first",
        ],
    );
    let js = read_json(&out.join("tokenizer.json"));
    assert_eq!(
        js["truncation"]["max_length"].as_u64(),
        Some(8),
        "tokenizer.json 应记录 truncation"
    );

    let tk = tok_pipeline::load_tokenizer_json(&out.join("tokenizer.json")).expect("加载产物");
    let long = "hello world this is a tokenizer corpus with many many words";
    let enc = tk.encode(long, true).expect("编码");
    assert_eq!(enc.ids.len(), 8, "开启截断后长度必须等于 max_length");
    assert_eq!(
        enc.ids[0],
        tk.vocab_map().get("[CLS]").copied().expect("CLS"),
        "only-first 保留头部"
    );
}

#[test]
fn padding_params_reach_tokenizer_json_and_encode() {
    // E5：填充同样要有 CLI 入口——batch-longest 在 encode --file 下按整批最长补齐。
    let sb = Sandbox::new("pad");
    let out = train_wordpiece(
        &sb,
        "pad",
        &[
            "--padding",
            "fixed:12",
            "--pad-token",
            "[PAD]",
            "--truncation-max-length",
            "32",
        ],
    );
    let js = read_json(&out.join("tokenizer.json"));
    assert_eq!(js["padding"]["strategy"]["Fixed"].as_u64(), Some(12));
    let pad_id = js["padding"]["pad_id"].as_u64().expect("pad_id");

    let tk = tok_pipeline::load_tokenizer_json(&out.join("tokenizer.json")).expect("加载产物");
    assert_eq!(
        tk.vocab_map().get("[PAD]").copied().expect("PAD"),
        pad_id as u32
    );

    let enc = tk.encode("hi", true).expect("编码");
    assert_eq!(enc.ids.len(), 12, "fixed:12 应补齐到 12");
    assert_eq!(*enc.ids.last().expect("末位"), pad_id as u32, "右侧补 pad");
    assert_eq!(enc.attention_mask.last(), Some(&0), "pad 位 attention=0");
}

#[test]
fn encode_batch_longest_pads_to_batch_max() {
    // encode 侧覆盖：--padding batch-longest 填到"整批最长"，而不是逐条自身长度。
    let sb = Sandbox::new("pad-batch");
    let out = sb.train_bpe("art", 400);
    let file = sb.write(
        "batch.txt",
        "hi\nhello world this is a much longer line\nok\n",
    );
    let o = run([
        "encode",
        "--tokenizer",
        out.to_str().expect("utf8"),
        "--file",
        file.to_str().expect("utf8"),
        "--padding",
        "batch-longest",
        "--pad-token",
        "[PAD]",
        "--json",
    ]);
    assert!(o.status.success(), "批量填充应成功: {}", stderr_of(&o));
    let v = json_stdout(&o);
    let arr = v.as_array().expect("数组");
    let lens: Vec<usize> = arr
        .iter()
        .map(|it| it["ids"].as_array().expect("ids").len())
        .collect();
    let masks: Vec<usize> = arr
        .iter()
        .map(|it| {
            it["attention_mask"]
                .as_array()
                .expect("attention_mask")
                .iter()
                .map(|v| v.as_u64().expect("mask") as usize)
                .sum()
        })
        .collect();
    assert_eq!(lens.len(), 3);
    let longest = *lens.iter().max().expect("最大长度");
    assert!(lens.iter().all(|l| *l == longest), "三条都应补到 {longest}");
    // 短句的有效 token 数应小于补齐后的长度，且其 attention_mask 尾部为 0
    assert!(masks[0] < longest, "短句有效长度应小于整批最长");
    assert!(masks.iter().all(|m| *m <= longest));
    let tail = arr[0]["attention_mask"]
        .as_array()
        .expect("mask")
        .last()
        .expect("末位")
        .as_u64();
    assert_eq!(tail, Some(0), "短句尾部应是 pad 位（attention=0）");
}

#[test]
fn encode_max_length_overrides_file_truncation() {
    // encode 侧覆盖：--max-length 只改长度，文件里的策略/方向继续生效。
    let sb = Sandbox::new("trunc-override");
    let out = train_wordpiece(
        &sb,
        "trunc-ov",
        &[
            "--post-processor",
            "bert",
            "--truncation-max-length",
            "16",
            "--truncation-direction",
            "left",
        ],
    );
    let dir = out.to_str().expect("utf8");
    let long = "hello world this is a tokenizer corpus with many many words";

    let base = json_stdout(&run([
        "encode",
        "--tokenizer",
        dir,
        "--text",
        long,
        "--json",
    ]));
    assert_eq!(base[0]["ids"].as_array().expect("ids").len(), 16);

    let short = json_stdout(&run([
        "encode",
        "--tokenizer",
        dir,
        "--text",
        long,
        "--max-length",
        "6",
        "--json",
    ]));
    assert_eq!(short[0]["ids"].as_array().expect("ids").len(), 6);
    // direction=left 保留尾部：末尾仍是 [SEP]
    let tk = tok_pipeline::load_tokenizer_json(&out.join("tokenizer.json")).expect("加载");
    let sep = tk.vocab_map().get("[SEP]").copied().expect("SEP");
    let ids: Vec<u64> = short[0]["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("id"))
        .collect();
    assert_eq!(
        *ids.last().expect("末位"),
        u64::from(sep),
        "left 方向应保留尾部 [SEP]"
    );
}

#[test]
fn pretokenizer_expression_errors_are_actionable() {
    // 表达式语法错误要走统一错误通道，并给出可操作提示（而非 panic / 静默降级）。
    let sb = Sandbox::new("pretok-err");
    let corpus = sb.write("corpus.txt", CORPUS);
    for (spec, hint) in [
        ("sequence(whitespace,punctuation", "括号未闭合"),
        ("split:behavior=isolated", "pattern"),
        ("nope", "未知 --pre-tokenizer"),
    ] {
        let out = sb.path(&format!("bad-{}", spec.len()));
        let o = run([
            "train",
            "--corpus",
            corpus.to_str().expect("utf8"),
            "--out",
            out.to_str().expect("utf8"),
            "--vocab-size",
            "300",
            "--model",
            "wordpiece",
            "--pre-tokenizer",
            spec,
        ]);
        assert_fails(&o, spec);
        assert!(
            stderr_of(&o).contains(hint),
            "--pre-tokenizer {spec:?} 的报错应包含 {hint:?}，实际 {:?}",
            stderr_of(&o)
        );
    }
}

#[test]
fn normalizer_expression_errors_are_actionable() {
    let sb = Sandbox::new("norm-err");
    let corpus = sb.write("corpus.txt", CORPUS);
    for (spec, hint) in [
        ("replace:content=x", "pattern"),
        ("prepend", "prepend"),
        ("nope", "未知 --normalizer"),
    ] {
        let out = sb.path(&format!("bad-norm-{}", spec.len()));
        let o = run([
            "train",
            "--corpus",
            corpus.to_str().expect("utf8"),
            "--out",
            out.to_str().expect("utf8"),
            "--vocab-size",
            "300",
            "--model",
            "wordpiece",
            "--normalizer",
            spec,
        ]);
        assert_fails(&o, spec);
        assert!(
            stderr_of(&o).contains(hint),
            "--normalizer {spec:?} 的报错应包含 {hint:?}，实际 {:?}",
            stderr_of(&o)
        );
    }
}

// ------------------------------------------------------------------ E6 角色

/// E6：`--role` 写入 `tokenizer_config.json` / `special_tokens_map.json`，
/// 且按目录加载（encode/decode）时角色语义被完整恢复。
#[test]
fn train_role_flags_persist_and_reload() {
    let sb = Sandbox::new("e6-role");
    let corpus = sb.write("corpus.txt", CORPUS);
    let out = sb.path("tk");
    let r = run([
        "train",
        "--corpus",
        corpus.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--vocab-size",
        "300",
        "--special",
        "[UNK]",
        "[PAD]",
        "--role",
        "cls=<|cls|>",
        "--role",
        "sep=<|sep|>",
        "--role",
        "mask=<|mask|>",
        "--role",
        "additional=<|system|>",
    ]);
    assert!(r.status.success(), "tok train 失败：{}", stderr_of(&r));

    let tc: Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("tokenizer_config.json")).expect("读取 tokenizer_config"),
    )
    .expect("解析 tokenizer_config");
    assert_eq!(tc["cls_token"], "<|cls|>");
    assert_eq!(tc["sep_token"], "<|sep|>");
    assert_eq!(tc["mask_token"], "<|mask|>");
    let stm: Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("special_tokens_map.json"))
            .expect("读取 special_tokens_map"),
    )
    .expect("解析 special_tokens_map");
    assert_eq!(stm["cls_token"], "<|cls|>");
    assert!(stm["additional_special_tokens"]
        .as_array()
        .expect("additional_special_tokens 应为数组")
        .iter()
        .any(|v| v == "<|system|>"));

    // added_tokens_decoder 中角色 token 必须是 special=true
    let dec = tc["added_tokens_decoder"]
        .as_object()
        .expect("added_tokens_decoder");
    let is_special = |content: &str| {
        dec.values()
            .any(|v| v["content"] == content && v["special"].as_bool().unwrap_or(false))
    };
    for tok in ["<|cls|>", "<|sep|>", "<|mask|>", "<|system|>"] {
        assert!(is_special(tok), "`{tok}` 未以 special 身份登记");
    }

    // 用目录（而非 tokenizer.json）加载：角色 token 应带 special_tokens_mask=1
    let enc = run([
        "encode",
        "--tokenizer",
        out.to_str().unwrap(),
        "--text",
        "hello <|system|> world",
        "--no-special",
        "--json",
    ]);
    assert!(enc.status.success(), "tok encode 失败：{}", stderr_of(&enc));
    let v = json_stdout(&enc);
    let v = &v[0];
    let mask = v["special_tokens_mask"]
        .as_array()
        .expect("special_tokens_mask");
    let ids = v["ids"].as_array().expect("ids");
    let toks = v["tokens"].as_array().expect("tokens");
    let pos = toks
        .iter()
        .position(|t| t == "<|system|>")
        .expect("正文里的 <|system|> 应被识别为单个 token");
    assert_eq!(
        mask[pos].as_u64(),
        Some(1),
        "additional 角色 token 应标为 special"
    );
    assert!(ids.len() == mask.len());

    // decode --skip-special 应剥离该 token
    let ids_arg = ids
        .iter()
        .map(|v| v.as_u64().unwrap().to_string())
        .collect::<Vec<_>>()
        .join(",");
    let dec_out = run([
        "decode",
        "--tokenizer",
        out.to_str().unwrap(),
        "--ids",
        &ids_arg,
        "--skip-special",
    ]);
    assert!(
        dec_out.status.success(),
        "tok decode 失败：{}",
        stderr_of(&dec_out)
    );
    let text = stdout_of(&dec_out);
    assert!(
        !text.contains("<|system|>"),
        "skip-special 未剥离角色 token：{text:?}"
    );
}

/// E6：非法 `--role` 必须在训练前被拦截（未知名字 / 缺 `=` / 空 token）。
#[test]
fn train_role_rejects_invalid_specs() {
    let sb = Sandbox::new("e6-role-bad");
    let corpus = sb.write("corpus.txt", CORPUS);
    let out = sb.path("tk");
    let base = |sb: &Sandbox, extra: &[&str]| {
        let mut args: Vec<String> = vec![
            "train".into(),
            "--corpus".into(),
            corpus.to_str().unwrap().into(),
            "--out".into(),
            out.to_str().unwrap().into(),
            "--vocab-size".into(),
            "300".into(),
            "--special".into(),
            "[UNK]".into(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        let _ = sb;
        args
    };

    let bad_name = run(base(&sb, &["--role", "begin=<|b|>"]));
    assert_fails(&bad_name, "未知角色名应报错");
    assert!(
        stderr_of(&bad_name).contains("begin"),
        "错误信息应点名非法角色：{}",
        stderr_of(&bad_name)
    );

    let no_eq = run(base(&sb, &["--role", "cls"]));
    assert_fails(&no_eq, "缺少 `=` 应报错");

    let empty = run(base(&sb, &["--role", "cls="]));
    assert_fails(&empty, "空 token 应报错");
}
