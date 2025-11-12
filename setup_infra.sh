


docker run -p 6333:6333 -p 6334:6334 \
    -v ~/vector_databases/qdrant/storage \
    --name qdrant_local \
    qdrant/qdrant:latest