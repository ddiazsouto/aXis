use serde::{Serialize, Deserialize};


struct Point {
    vector: Vec<f32>,
    payload: serde_json::Value
}

struct TinyVectorDB {
    points: Vec<Point>,
    path: String,
}

fn main() {
    println!("Hello, world!");
}
