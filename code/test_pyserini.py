from pyserini.search.lucene import LuceneSearcher
import os

#os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
hits = searcher.search('what is a lobster roll?')

for i in range(0, 10):
    break
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')

