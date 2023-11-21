import pyterrier as pt
import ir_measures
import os

pt.init()

# list of filenames to index
files = pt.io.find_files("WT2G/")

# build the index
indexer = pt.TRECCollectionIndexer("./wt2g_index", verbose=True, blocks=False)
if "data.properties" not in os.listdir("wt2g_index"):
    indexref = indexer.index(files)

    # load the index, print the statistics
    index = pt.IndexFactory.of(indexref)
    print(index.getCollectionStatistics().toString())
else:
    index = pt.IndexFactory.of("./wt2g_index/data.properties")

topics = pt.io.read_topics("topics_401_450.txt")
qrels = pt.io.read_qrels("qrels.trec8.small_web")
#print((topics))

tfidf = pt.BatchRetrieve(index, controls={"wmodel": "TF_IDF"})
bm25 = pt.BatchRetrieve(index, controls={"wmodel": "BM25"})
"""
tfidf.setControl("wmodel", "TF_IDF")
tfidf.setControls({"wmodel": "TF_IDF"})
"""
#res = tfidf.transform(topics)

res = pt.Experiment(
    [tfidf, bm25],
    topics,
    qrels,
    eval_metrics=[ir_measures.MAP@1000, ir_measures.NDCG@1000]
)

print(res)
