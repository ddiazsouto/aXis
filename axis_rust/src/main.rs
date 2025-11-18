use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use ndarray::Array1;
use std::collections::HashMap;
use serde_json::Value;

#[derive(Serialize, Deserialize, Clone)]
struct VectorRegistry {
    vectors: Vec<Vec<f32>>,
    payloads: Vec<HashMap<String, Value>>,
}

struct AXisDB {
    path: String,
    vector_registry: VectorRegistry,
}

impl AXisDB {
    fn new(path: &str) -> Self {
        let vector_registry = if Path::new(path).exists() {
            let data = fs::read_to_string(path).expect("Failed to read");
            serde_json::from_str(&data).expect("Bad JSON")
        } else {
            VectorRegistry {
                vectors: Vec::new(),
                payloads: Vec::new(),
            }
        };
        Self {
            path: path.to_string(),
            vector_registry,
        }
    }

    fn insert(&mut self, payload: HashMap<String, Value>) {
        let vector = vec![0.1; 384];  // Fake vector
        self.vector_registry.vectors.push(vector);
        self.vector_registry.payloads.push(payload);
        self.save();
    }

    fn search(&self, top_k: usize) -> Vec<(f32, &HashMap<String, Value>)> {
        let query = Array1::from_vec(vec![0.1; 384]);
        let mut results: Vec<_> = self.vector_registry.vectors.iter().zip(&self.vector_registry.payloads).map(|(v, p)| {
            let v_arr = Array1::from_vec(v.clone());
            (Self::cosine(&query, &v_arr), p)
        }).collect();
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        results.into_iter().take(top_k).collect()
    }

    fn cosine(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot = a.dot(b);
        let norm_a = a.mapv(|x| x * x).sum().sqrt();
        let norm_b = b.mapv(|x| x * x).sum().sqrt();
        dot / (norm_a * norm_b + 1e-8)
    }

    fn save(&self) {
        let data = serde_json::to_string_pretty(&self.vector_registry).expect("Serialize failed");
        fs::write(&self.path, data).expect("Write failed");
    }
}

fn main() {
    let mut db = AXisDB::new("axis_rust.db");

    let mut payload = HashMap::new();
    payload.insert("text".to_string(), Value::String("Test sentence.".to_string()));

    db.insert(payload);

    let results = db.search(1);
    let best = &results[0];
    println!("Score: {:.4}, Text: {}", best.0, best.1["text"]);
}