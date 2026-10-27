# Evaluation methodology

`python evaluation/evaluate.py` registers an isolated throwaway account and indexes `evaluation/datasets/sample.txt`. Each question has an expected source and factual substring. The script performs the same authenticated search with CrossEncoder disabled and enabled. A hit is counted when a returned source has the expected filename and contains the expected factual substring. Hit@3 is the hit fraction. MRR averages reciprocal rank, or zero for no hit. `retrieval_ms` is server-side search time; `http_latency_ms` includes HTTP overhead. The evaluation does not call Qwen and does not measure end-to-end response latency.

The sample is intentionally tiny, synthetic and suitable only to verify plumbing. It cannot establish general answer faithfulness or prove reranking improves quality. Actual measurements are written to `evaluation/results/latest.json` only after execution. No benchmark results are included in the repository.
