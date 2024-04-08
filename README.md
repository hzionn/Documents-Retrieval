# Documents Retrieval - Project 2

We are ask to implement several different retrieval methods. 

Some of these retrieval methods will be the implementation of the basic retrieval models studied in the class (e.g. TF-IDF, BM25, Language Models with different Smoothing). 

Various tools are build on top of [Lemur Project](http://www.lemurproject.org/) toolkits, includes search engines, browser toolbars, text analysis tools, and data resources that support research and development of information retrieval and text mining.

## Requirements

TODO: add requirements for this project.

while installing `pyserini`, it might fail to install `nmslib`.

[here](https://github.com/nmslib/nmslib/issues/538#issuecomment-1735283499)'s a work around to install `nmslib` on python3.11 environment.

## Directory and Files

(assuming you have these files)

- Document Corpus
  - `WT2g`: a collection contains Web documents, with being a 2GB corpus. Will use the corpus to test the retrieval algorithms, and run experiments.
- Queries
  - `topics.401-450.txt`: a set of 50 TREC queries for the corpus, with the standard TREC format having topic title, description and narrative. Documents from the corpus have been judged with respect to their relevance to these queries by NIST assessors.

## Evaluation

Evaluation tools:

- `trec_eval.pl` - provides a number of statistics about how well the retrieval function corresponding to the results_file did on the corresponding queries.
- `ireval.jar`

for using `trec_eval.pl`, you can run the following command:

```bash
perl trec_eval.pl -[q] qrel_file results_file
```

to reproduce the results from `pyterrier`, run the following command:

```bash
make pyterrier
```

## Works

We need to run the set of queries against the WT2g collection, return a ranked list of documents (the top 1000) in a particular format, and then evaluate the ranked lists.
see [WSM Project 2.pdf](WSM%20Project%202.pdf) for project report.
