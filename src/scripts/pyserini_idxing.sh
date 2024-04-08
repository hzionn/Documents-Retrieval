# build index with pyserini
python -m pyserini.index.lucene \
    --collection TrecwebCollection \
    --input WT2G \
    --index indexes/wt2g-idx \
    --generator DefaultLuceneDocumentGenerator \
    --threads 16 \
    --storePositions --storeDocvectors --storeRaw
