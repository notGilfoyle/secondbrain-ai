# Offline retrieval evaluation report

Run with `venv/bin/python scripts/evaluate_retrieval.py --k 3`.

- Dataset: 10 questions over 8 synthetic chunks
- Answerable questions scored: 8
- Unanswerable questions excluded from ranking metrics: 2
- Recall@3: 1.000
- Mean reciprocal rank (MRR): 1.000
- Answerable queries with no relevant chunk in top 3: 0

The deterministic fixture calls the production `SemanticRetrievalService`
with fixed query and chunk vectors. Recall@k averages, across answerable
questions, the fraction of expected supporting chunk IDs found in the first k
results. MRR averages the reciprocal rank of the first relevant chunk for each
answerable question. The miss count is the number of answerable questions with
no expected chunk among their top-k results.

These results verify retrieval wiring and metric calculation only. Synthetic
controlled vectors do not measure the embedding model, and unanswerable cases
do not measure abstention because this retrieval service always returns its
highest-ranked candidates. This report contains no LLM answer-quality score and
also excludes retrieval false-positive evaluation for unanswerable questions.
It does not represent a live Cohere evaluation. The corpus and query set are
small, synthetic, and not representative of personal documents or a production
workload.
