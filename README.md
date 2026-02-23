# aXis Vector Database

A lightweight, fast vector database with semantic search capabilities built on Polars and Delta Lake.

## What is aXis?

aXis is a simple vector database that enables semantic search over text data. It uses sentence transformers to convert text into embeddings and stores them efficiently in Delta Lake format for fast similarity searches.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Server

```bash
python axis_webapp.py
```

The server will start on `http://localhost:5005`

### Basic Usage

#### Insert Data

```python
from axis_python.axis_db import aXisDB

db = aXisDB("main.axis")
db.insert("What is the capital of France?", {"answer": "Paris"})
db.vector_registry.save()
```

#### Search

```python
results = db.search("French capital city", top_k=3)
for payload, text, index in results:
    print(f"Answer: {payload}")
```

## Web API

### Search
```bash
POST /api/search
{
  "query": "your search query"
}
```

### Insert
```bash
POST /api/insert
{
  "text": "text to embed",
  "payload": {"key": "value"}
}
```

### Status
```bash
GET /api/status
```

## Database Files

`.axis` files are Delta Lake directories containing:
- Vector embeddings (384 dimensions)
- Text content
- Payloads (stored as strings)
- Timestamps and indices

## Features

- **Fast semantic search** using cosine similarity
- **Delta Lake storage** for ACID compliance
- **Lazy loading** for memory efficiency
- **Batch insertions** via DataFrames
- **REST API** for easy integration
- **Local embeddings** (no external API calls)

## Architecture

- `axis_db.py` - Core database class
- `vector_registry.py` - Vector storage and retrieval
- `axis_webapp.py` - Flask web server
- `models/` - Local sentence transformer model

---

Built with ❤️ using Polars, Delta Lake, and Sentence Transformers
