# aXis Vector Database

A lightweight, fast vector database with semantic search capabilities built on Polars and Delta Lake. Some vector databases solve scale problems while here we just want to allow the user for a quick way to access the information and for RAG systems to have a simple interface to interact with.

It supports MCP servers to query via REST API

## What is aXis?

aXis is a simple vector database that enables semantic search over text data. It uses sentence transformers to convert text into embeddings and stores them efficiently in Delta Lake format for fast similarity searches.

## Quick Start

You can use the default main.axis data, which has been loaded from under the data folder, using the general knowledge dataset

```bash
pip install -r requirements.txt
```

### Run the Server

```bash
python axis_webapp.py
```

The server will start on `http://localhost:5005`


## How to load more data ans programatic usage


### Insert Data ad-hoc

```python
from axis_python.axis_db import aXisDB

db = aXisDB("main.axis")
db.insert("What is the capital of France?", "Paris")
db.vector_registry.save()
```

### Loading a whole dataframe at once

This is the example of how we generated main.axis

```python
from axis_python.axis_db import aXisDB
import polars as pl


df = pl.read_csv("./data/general_knowledge.csv")

db = aXisDB("main")
db.insert_dataframe(
    dataframe=df,
    vectorise_col="text",
    payload_col="payload"
)
db.vector_registry.save()
```

### Search

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

## Potential Improvements

- aXis is not optimized for concurrency
- The Vector retrieval algorithm is inefficient, reducing nodes to search through implementation of hnsw would be beneficial
- UI could be improved to suggest better exploration of similar answers

