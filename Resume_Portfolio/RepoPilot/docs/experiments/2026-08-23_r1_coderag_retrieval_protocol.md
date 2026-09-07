# R1 CodeRAG-Bench RepoEval retrieval protocol and candidate freeze

Status: final holdout execution authorised for the frozen `bm25_top_k`
candidate only.  This document records an offline retrieval evaluation; it is
not a SWE-bench patch-resolution result.

## Frozen input

- Upstream code: `external/CodeRAG-Bench` at
  `f9e100ca9ed94b8f1983b356ae81966e30210cf4`.
- Upstream materialisation: `repoeval_repo`, `function` split,
  `context_length=2k`, `window_size=50`, `slice_size=5`.
- Complete materialised repositories: 5; query count: 341; corpus windows:
  4,198.  The absent sixth repository is not silently imputed.
- Dataset digest: `1daabc2a320a0336fd6b15c395972017fd41ca76c0ea2836962d63c5167406d5`.
- Valid manifest:
  `evaluation/retrieval/coderag/manifests/repoeval_r1_20260823_v2.json`, canonical manifest digest
  `d0ba58a36108744dbfddc81bda5f09a7120bb0ca3f2c785de0df9fc02f65041a`.
  It assigns 67/137/137 queries to development/validation/final holdout and
  stratifies by source repository.

The earlier `repoeval_r1_20260823.json` is retained as an immutable failed
diagnostic.  It was produced before the grouped allocator was fixed and has a
non-20/40/40 allocation; it is excluded from every result below.

## Development ablation (not a final claim)

| Variant | NDCG@10 | Recall@10 | MRR@10 | Median retrieval latency |
|---|---:|---:|---:|---:|
| BM25 top-10 | 0.916341 | 0.577128 | 0.992537 | 164.63 ms |
| Hybrid RRF | 0.828973 | 0.498408 | 0.982587 | 253.78 ms |
| Routed hybrid RRF | 0.846905 | 0.512468 | 0.982587 | 257.31 ms |
| Routed hybrid + expansion | 0.670775 | 0.380786 | 0.982587 | 244.77 ms |
| Adaptive code hybrid | 0.623583 | 0.377572 | 0.985075 | 245.44 ms |

All rows use the same 67 frozen development queries.  `nomic-embed-text`
embeddings were generated once into an ignored content-validated cache and
reused across hybrid rows.  The raw rankings and their immutable receipts live
under `artifacts/evaluations/r1/` on the local experiment machine.

## Validation and final candidate decision

The development primary metric (`NDCG@10`) selects the fixed `bm25_top_k`
baseline.  No hyperparameter or implementation changed after this selection.
On the untouched 137-query validation partition, this candidate obtained
NDCG@10 0.907190, Recall@10 0.615483, MRR@10 0.985401, and 163.37 ms median
retrieval latency.  Therefore exactly one final-holdout command is authorised:
the same `bm25_top_k` implementation and controls, with no adaptive component.

The final result must be reported only as CodeRAG-Bench RepoEval retrieval
metrics.  It cannot support an improvement claim against hybrid retrieval or a
SWE-bench resolved-rate claim, because no paired final comparison or Docker
SWE evaluation has been run.

## One-time final holdout result

The guarded `bm25_top_k` final command ran once on the 137 assigned queries:
NDCG@10 **0.903961**, Recall@10 **0.606940**, MRR@10 **0.984185**; median
retrieval latency was **164.80 ms** (p95 203.21 ms).  Its raw result SHA-256 is
`b3928dc2ad24c7541aa3ea1e48ae53ac0841ee06c7e82b22b5d2c8a43ba98343`, with a
neighbouring immutable receipt in `artifacts/evaluations/r1/`.
