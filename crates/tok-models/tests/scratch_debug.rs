use tok_core::{Model, Vocabulary};
use tok_models::WordPieceModel;

#[test]
fn debug_wordpiece_state() {
    let mut v = Vocabulary::new();
    for (t, id) in [("[UNK]", 0u32), ("un", 1), ("##want", 2), ("##ed", 3), ("he", 4), ("llo", 5)] {
        v.insert(t, id).unwrap();
    }
    println!("len={} contains_he={} contains_llo={} id_he={:?}", v.len(), v.contains("he"), v.contains("llo"), v.token_to_id("he"));
    let m = WordPieceModel::new(v, "[UNK]");
    for s in ["hello", "zzz", "unwanted"] {
        let toks = m.tokenize(s).unwrap();
        let vals: Vec<String> = toks.iter().map(|t| format!("{}#{}@({},{})", t.value, t.id, t.offsets.0, t.offsets.1)).collect();
        println!("{:?} -> {:?}", s, vals);
    }
}
