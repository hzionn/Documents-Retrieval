import os

import pyterrier as pt
import xgboost as xgb
from pyterrier.measures import MAP, NDCG, P, R, Rprec
from sklearn.ensemble import RandomForestRegressor

if not pt.started():
    pt.init()

working_dir = os.path.join(os.path.dirname(os.getcwd()))
working_dir = ""

WT2G_dir = os.path.join(working_dir, 'WT2G')
files = pt.io.find_files(WT2G_dir)

# build the index
index_path = os.path.join(os.path.dirname(os.getcwd()), "wt2g_index")
indexer = pt.TRECCollectionIndexer(
    index_path,
    verbose=True,
    blocks=False,
)
index_ref = indexer.index(files)

topics = pt.io.read_topics(working_dir + "/topics.401-450.txt")
qrels = pt.io.read_qrels(working_dir + "/qrels.trec8.small_web")

# retrieval models
tfidf = pt.BatchRetrieve(index_path, wmodel="TF_IDF")
tfidf_new = pt.BatchRetrieve(index_path, wmodel="TF_IDF", controls={"tf_idf.k_1":2, "tf_idf.b":0.75})
lemurtfidf = pt.BatchRetrieve(index_path, wmodel="LemurTF_IDF")
bm25 = pt.BatchRetrieve(index_path, controls={"wmodel": "BM25"})
pl2 = pt.BatchRetrieve(index_path, controls={"wmodel": "PL2"})
hiemstra = pt.BatchRetrieve(index_path, controls={"wmodel": "Hiemstra_LM"})
dirichlet = pt.BatchRetrieve(index_path, controls={"wmodel": "DirichletLM"})

rerank_1 = (bm25 % 100) >> dirichlet

# retrieval
res_tfidf = tfidf.transform(topics)
res_tfidf_new = tfidf_new.transform(topics)
res_lemurtfidf = lemurtfidf.transform(topics)
res_bm25 = bm25.transform(topics)
res_pl2 = pl2.transform(topics)
res_hiemastra = hiemstra.transform(topics)
res_dirichlet = dirichlet.transform(topics)
res_rerank_1 = rerank_1.transform(topics)

pipeline = bm25 >> (tfidf ** pl2)
rf = RandomForestRegressor(n_estimators=400)
rf_pipe = pipeline >> pt.ltr.apply_learned_model(rf)
rf_pipe.fit(topics, qrels)
pt.Experiment(
    [bm25, rf_pipe], 
    topics, 
    qrels, 
    ["map"], 
    names=["BM25 Baseline", "LTR"]
)

# this configures XGBoost as LambdaMART
lmart_x = xgb.sklearn.XGBRanker(
      objective='rank:ndcg',
      learning_rate=0.1,
      gamma=1.0,
      min_child_weight=0.1,
      max_depth=10,
      verbose=2,
      random_state=42
)
lmart_x_pipe = pipeline >> pt.ltr.apply_learned_model(lmart_x, form="ltr")
lmart_x_pipe.fit(topics, qrels, topics, qrels)

# generate results
results = pt.Experiment(
    [tfidf, tfidf_new, lemurtfidf, bm25, pl2, hiemstra, dirichlet, rerank_1, rf_pipe, lmart_x_pipe],
    topics,
    qrels,
    eval_metrics=[R@10, P@10, P@200, MAP, MAP@10, MAP@100, NDCG@100, Rprec]
)
print(results)
