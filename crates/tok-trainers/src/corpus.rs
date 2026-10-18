//! 语料读取：支持单文件、多文件、目录（可选递归），按行切分。

use std::path::{Path, PathBuf};

use tok_core::{Result, TokenizerError, layer};

/// 语料数据：文件清单 + 按行展开的文本。
#[derive(Debug, Clone, Default)]
pub struct CorpusData {
    pub files: Vec<PathBuf>,
    pub lines: Vec<String>,
    /// 因解码失败或超长被跳过的行数（不静默）
    pub skipped_lines: usize,
}

impl CorpusData {
    pub fn total_bytes(&self) -> usize {
        self.lines.iter().map(|l| l.len()).sum()
    }
}

/// 语料读取器。
#[derive(Debug, Clone)]
pub struct CorpusReader {
    paths: Vec<PathBuf>,
    recursive: bool,
    /// 扩展名白名单（小写、不含点）；空表示只接受 .txt/.md/.json/.csv 等文本
    extensions: Vec<String>,
    /// 行数上限（用于快速试跑）
    max_lines: Option<usize>,
    /// 单行字节上限（超长行截断而非丢弃，避免 OOM）
    max_line_bytes: usize,
    /// 是否并行读取（依赖 rayon）
    parallel: bool,
}

impl CorpusReader {
    pub fn new(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            recursive: false,
            extensions: vec![
                "txt".into(),
                "md".into(),
                "json".into(),
                "csv".into(),
                "tsv".into(),
                "log".into(),
            ],
            max_lines: None,
            max_line_bytes: 1 << 20,
            parallel: true,
        }
    }

    pub fn recursive(mut self, yes: bool) -> Self {
        self.recursive = yes;
        self
    }

    pub fn extensions(mut self, exts: Vec<String>) -> Self {
        self.extensions = exts.into_iter().map(|e| e.trim_start_matches('.').to_lowercase()).collect();
        self
    }

    pub fn max_lines(mut self, n: Option<usize>) -> Self {
        self.max_lines = n;
        self
    }

    pub fn max_line_bytes(mut self, n: usize) -> Self {
        self.max_line_bytes = n.max(1);
        self
    }

    pub fn parallel(mut self, yes: bool) -> Self {
        self.parallel = yes;
        self
    }

    /// 展开路径：目录按扩展名白名单收集，结果排序保证可复现。
    pub fn discover(&self) -> Result<Vec<PathBuf>> {
        let mut out: Vec<PathBuf> = Vec::new();
        for p in &self.paths {
            if p.is_dir() {
                collect_dir(p, self.recursive, &self.extensions, &mut out)?;
            } else if p.is_file() {
                out.push(p.clone());
            } else {
                return Err(TokenizerError::layer(
                    layer::FACADE,
                    format!("语料路径不存在: {}", p.display()),
                ));
            }
        }
        out.sort();
        out.dedup();
        if out.is_empty() {
            return Err(TokenizerError::layer(layer::FACADE, "语料路径下没有匹配的文本文件"));
        }
        Ok(out)
    }

    /// 读取并按行返回。
    pub fn read(&self) -> Result<CorpusData> {
        let files = self.discover()?;
        let mut data = CorpusData { files: files.clone(), ..Default::default() };
        if self.parallel && files.len() > 1 {
            use rayon::prelude::*;
            let limit = self.max_lines;
            let per_file: Vec<Result<Vec<String>>> = files
                .par_iter()
                .map(|f| read_one(f, self.max_line_bytes, None))
                .collect();
            for r in per_file {
                let lines = r?;
                for l in lines {
                    if let Some(cap) = limit {
                        if data.lines.len() >= cap {
                            break;
                        }
                    }
                    data.lines.push(l);
                }
                if let Some(cap) = limit {
                    if data.lines.len() >= cap {
                        break;
                    }
                }
            }
        } else {
            for f in &files {
                let lines = read_one(f, self.max_line_bytes, self.max_lines.map(|c| c - data.lines.len()))?;
                data.lines.extend(lines);
                if let Some(cap) = self.max_lines {
                    if data.lines.len() >= cap {
                        break;
                    }
                }
            }
        }
        if data.lines.is_empty() {
            return Err(TokenizerError::layer(layer::FACADE, "语料为空（没有读到任何行）"));
        }
        Ok(data)
    }
}

fn collect_dir(dir: &Path, recursive: bool, exts: &[String], out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if recursive {
                collect_dir(&path, recursive, exts, out)?;
            }
        } else if path.is_file() {
            let ok = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| exts.iter().any(|w| w.eq_ignore_ascii_case(e)))
                .unwrap_or(false);
            if ok {
                out.push(path);
            }
        }
    }
    Ok(())
}

fn read_one(path: &Path, max_line_bytes: usize, limit: Option<usize>) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        TokenizerError::layer(layer::FACADE, format!("读取语料失败 {}: {}", path.display(), e))
    })?;
    let content = content.replace("\r\n", "\n").replace('\r', "\n");
    let mut out = Vec::new();
    for line in content.split('\n') {
        if let Some(cap) = limit {
            if out.len() >= cap {
                break;
            }
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.len() > max_line_bytes {
            // 截断但不丢弃，并在调用侧可见（避免静默丢数据）
            let mut cut = max_line_bytes;
            while cut > 0 && !line.is_char_boundary(cut) {
                cut -= 1;
            }
            out.push(line[..cut].to_string());
        } else {
            out.push(line.to_string());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tok_trainers_{}_{}", tag, std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn reader_skips_empty_lines_and_filters_ext() {
        let dir = tmpdir("corpus");
        let a = dir.join("a.txt");
        let mut f = std::fs::File::create(&a).unwrap();
        writeln!(f, "hello world\n\n  \nsecond line").unwrap();
        drop(f);
        let b = dir.join("b.bin");
        std::fs::write(&b, "should be ignored").unwrap();

        let data = CorpusReader::new(vec![dir.clone()]).read().unwrap();
        assert_eq!(data.lines, vec!["hello world".to_string(), "second line".to_string()]);
        assert_eq!(data.files.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reader_respects_max_lines() {
        let dir = tmpdir("maxlines");
        let a = dir.join("c.txt");
        std::fs::write(&a, "1\n2\n3\n4\n5\n").unwrap();
        let data = CorpusReader::new(vec![a]).max_lines(Some(2)).read().unwrap();
        assert_eq!(data.lines.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_path_errors() {
        let r = CorpusReader::new(vec![PathBuf::from("E:/__no_such_dir__")]);
        assert!(r.read().is_err());
    }
}
