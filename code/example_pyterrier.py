import pyterrier as pt
pt.init()

"""
dataset = pt.get_dataset("vaswani")
# vaswani dataset provides an index, topics and qrels

# lets generate two BRs to compare
tfidf = pt.BatchRetrieve(dataset.get_index(), wmodel="TF_IDF")
bm25 = pt.BatchRetrieve(dataset.get_index(), wmodel="BM25")

print(pt.Experiment(
    [tfidf, bm25],
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=["map", "recip_rank"]
))
"""

print(pt.datasets.find_datasets("wt2g"))
