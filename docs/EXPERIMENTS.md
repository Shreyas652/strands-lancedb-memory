# Development experiments

## Reproduce the inputs

The official LongMemEval repository links to the cleaned dataset on Hugging Face. Run `python scripts/fetch_longmemeval.py` to download it into the ignored `data/` directory. The fetcher checks the exact SHA-256 used by these reports and refuses a changed file. Do not commit the dataset or a model key. Run `python -m evals.split --dataset data/longmemeval_s_cleaned.json --output results/longmemeval_dev_split.json --seed 20261124` to recreate the split. Model evaluations read `OPENAI_API_KEY` from an ignored `.env` or the process environment and enforce a $20 ledger ceiling; offline retrieval and checks need no key. The optional vector grid needs pinned FastEmbed 0.9.1 and the cached ONNX weights fingerprint recorded in `results/vector_dev.json`.

## Protocol

LongMemEval-S is loaded from the public cleaned dataset. `evals.split` makes a fixed, question-type-stratified 100-question development split and seals the remaining 400 IDs by hash. The split manifest and raw dataset SHA-256 are in `results/longmemeval_dev_split.json`. All model and retrieval decisions use only development questions. Gold answers enter only the judge call; search and the reader receive the question and retrieved conversation, never a gold answer or evidence label.

The retrieval-only harness indexes each complete conversation session as one memory in three stores: this package with FTS, Strands `FileMemoryStore`, and Strands `TestMemoryStore`. A duplicate session ID remains a separate memory occurrence through its ordinal. Each system sees the same input sessions. It records evidence-label hit@3/5/10, search latency, and selected session IDs and ordinals. This tests retrieval, not answer quality. `evals.answer_dev` and `evals.frozen_answers` apply the same reader and judge to the retained results.

The reader is `gpt-4o-mini-2024-07-18` at temperature 0. The pinned official-style judge is `gpt-4o-2024-08-06` at temperature 0 using the prompt copied from LongMemEval commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`. `results/official_prompt_verification.json` records exact output agreement for all six question types and abstention. Each model request records its prompt hash, model ID, tokens, response, and billed cost in the local ignored ledger. The report publishes per-question outputs and aggregate cost. The shared ledger refuses a request that could cross the $20 ceiling.

## Retrieval-only result on the development split

Source: `results/dev_retrieval.json`, 100 questions and 300 raw rows. These results are development evidence, not a held-out test score.

| Store | Hit@3 | Hit@5 | Hit@10 | p50 search | p95 search |
| --- | ---: | ---: | ---: | ---: | ---: |
| LanceDB FTS | 94% | 97% | 98% | 608.7 ms | 1284.2 ms |
| FileMemoryStore | 90% | 94% | 96% | 1025.9 ms | 1513.5 ms |
| TestMemoryStore | 87% | 93% | 95% | 54.2 ms | 87.4 ms |

The FTS store led these two baselines on evidence hit rate. It was faster than the file store on this host but much slower than the in-memory test store. Search timing includes reopening the local LanceDB table. This is a small dev comparison, so the 3-point hit@5 gap against the file store is not a claim of general superiority or answer-quality improvement.

## Answer quality on the development split

Source: `results/dev_answers.json`, 100 questions × 10 arms = 1,000 raw answer and judgment rows. The same reader and judge were used for all arms. The report accounts for $0.41973455 across 745 unique request records, with repeated prompts served from the durable ledger. This is **not an official leaderboard score**; it uses the official prompt and pinned GPT-4o judge inside this repository's reader/retrieval harness.

| Arm | Correct of 100 |
| --- | ---: |
| No memory | 3 |
| FileMemoryStore, top 5 | 55 |
| LanceDB FTS, top 5 | 53 |
| TestMemoryStore, top 5 | 45 |

At top 5, FTS had a higher evidence hit rate than the file store but two fewer correct answers. The result falsifies the idea that increasing evidence hit rate alone guarantees better answers. By question type, FTS versus FileMemoryStore was 8 versus 9 of 16 for knowledge updates, 10 versus 8 of 27 for multi-session, 10 versus 11 of 11 for assistant statements, 1 versus 0 of 6 for preferences, 14 versus 13 of 14 for user statements, and 10 versus 14 of 26 for temporal reasoning. The temporal gap is a concrete target for the fact and time handling ablations; these groups are small and should not be treated as stable population estimates.

## Local vector and weighted hybrid retrieval

Source: `results/vector_dev.json`, 100 dev questions and 500 raw rows. The only embedding model was a locally cached `BAAI/bge-small-en-v1.5` ONNX model through FastEmbed 0.9.1; the report fingerprints its weights. No model weights were downloaded during this run and API cost was zero. Each question indexed the same 4,772 total sessions in FTS, vector, and four weighted-hybrid variants. The modes used identical stored text and the same evidence labels.

| Mode | Hit@3 | Hit@5 | Hit@10 |
| --- | ---: | ---: | ---: |
| FTS only, separate baseline run | 94% | 97% | 98% |
| Hybrid, vector weight 0.30 | 94% | 97% | 99% |
| Hybrid, vector weight 0.50 | 93% | 96% | 98% |
| Hybrid, vector weight 0.70 | 93% | 96% | 98% |
| Hybrid, vector weight 0.90 | 92% | 95% | 96% |
| Vector only | 90% | 93% | 96% |

Weight 0.30 tied FTS at hit@5 and improved hit@10 by one question; equal weighting lost one hit@5 relative to FTS. Thus the data do not justify replacing the FTS default on retrieval alone. The FTS and vector jobs ran at different times with other activity on the host, so their latency numbers are retained in JSON but not interpreted as a controlled cross-job speed comparison.

`results/vector_answers_dev.json` contains 1,600 judged answer rows for the five vector/hybrid modes at top 3, 5, and 10, plus no-memory controls. The strongest hybrid arms answered 50/100 correctly, versus 54/100 for the full-session FTS top-3 arm in `results/dev_answers.json`. The vector-only top-3 arm reached 48/100. These are development results, not held-out scores.

## Extracted-fact ablation

`results/dev_facts.json` records query-independent extraction from 1,233 session batches into 13,250 facts, at $1.98745065 attributable model cost. `results/facts_retrieval_dev.json` and `results/fact_answers_dev.json` compare append-only facts and keyed current-value ranking. At top 5, append-only facts retrieved labeled evidence for 74/100 questions and keyed facts for 73/100; full-session FTS retrieved evidence for 97/100 in the separate baseline run. The best fact-shaped answer arms reached 28/100 correct, versus 54/100 for the frozen full-session top-3 arm. The two retrieval runs differ in memory representation and timing, so these numbers establish an ablation outcome for this harness, not a universal ranking of extraction strategies. This extractor is separate from Strands' production `ModelExtractor` and should not be presented as its measured quality.

## Predeclared freeze rule

`evals.freeze` considers the full-session package candidates only: FTS, vector-only, and each measured weighted hybrid, at result limits 3, 5, and 10. It requires every arm in the grid and selects the one with the most correct development answers. Ties go to FTS before hybrid before vector, then to the smaller result limit, then lexical arm name. It verifies identical reader and judge settings and stores the source report hashes before opening the sealed test split. Fact-shaped memories are evaluated as an ablation but are not a frozen package candidate because their query-independent extraction harness is separate from Strands' production `ModelExtractor` behavior. The resulting `results/frozen_config.json` selects LanceDB FTS at top 3, with 54/100 correct development answers. This exact configuration was used for the 400-question held-out run.

The config's `frozen_at_utc` timestamp is self-reported, and the config was committed together with the result files, so the timestamp alone does not independently prove the freeze preceded testing. Future phases should commit or tag the frozen config before running a held-out set. The recorded `freeze_sha256` is for the original Windows CRLF encoding (`6dabcbc3…`); the committed LF encoding hashes to `df351c4e…`. The integrity check accepts either encoding of the same config without changing recorded hashes.

## Frozen held-out test result

`results/frozen_retrieval.json` contains 1,200 retrieval rows: 400 questions in each of the three stores. `results/frozen_answers.json` contains 1,600 judged rows: the three stores at top 3 plus no-memory controls. The test harness did not reselect a weight or result limit after seeing these data.

| Store | Evidence hit@3 | Evidence hit@5 | Correct answers / 400 | Answer accuracy |
| --- | ---: | ---: | ---: | ---: |
| LanceDB FTS | 95.25% | 97.00% | 201 | 50.25% |
| FileMemoryStore | 88.00% | 92.25% | 186 | 46.50% |
| TestMemoryStore | 84.00% | 89.00% | 174 | 43.50% |
| No memory | — | — | 27 | 6.75% |

The FTS and file arms agreed on 165 correct questions; FTS alone was correct on 36, and the file arm alone on 21. Thus the net held-out difference is 15/400, or 3.75 percentage points. The exact two-sided McNemar test gives p = 0.0627 (0.063 to three decimals), so the difference is not statistically significant at the 5% level. `results/paired_significance.json` records the paired counts, p-value, and source report hash. This single benchmark run and model judge do not establish a general superiority claim. On the AWS retrieval host, FTS search p50 was 491.4 ms versus 59.2 ms for FileMemoryStore and 20.8 ms for the in-memory test store; these timings include each store's measured search path, not ingestion or model latency.

The retrieval stage used a temporary Ubuntu 24.04 `c7i.xlarge` EC2 instance. Frozen answer grading used a temporary `c7i.large` instance and a scoped Secrets Manager secret; the key was not included in the benchmark bundle or logs. Both runs resumed from their durable checkpoints/ledger. The verified reports and ledger were downloaded before the temporary instances, roles, security groups, buckets, and secret were removed. Cloud compute charges are separate from the model API ledger.

## All 500 questions and judge check

`results/all_500_summary.json` joins the 100 development questions to the 400 frozen test questions without changing the configuration. LanceDB FTS answered 255/500 (51.0%) correctly; FileMemoryStore answered 240/500 (48.0%), TestMemoryStore 219/500 (43.8%), and no-memory 30/500 (6.0%). This is a combined descriptive result, not an additional held-out score or an official leaderboard entry. The question-type breakdown is in the raw report and README; preferences (2/30 for FTS) and multi-session questions (42/133) remain weak points.

`results/second_judge.json` regraded a fixed-seed sample of 50 held-out answer rows with `gpt-4o-mini-2024-07-18`. It agreed with the primary judge on 49/50 (98%). This is the same model family as the reader that wrote the answers, so the agreement is a weak independence check, not a substitute for human adjudication. **Future work:** Re-judge the same 50-question sample with a different model family. The shared model ledger totals $4.14651835 after this check, below the $20 cap. This excludes AWS compute and storage charges.

## Search scale

`experiments/scale_memory.py` bulk seeds 32-dimensional synthetic vectors at 1k, 10k, and 100k rows. It measures exact and IVF_FLAT indexed search through the public store API, and compares each indexed top-10 ranking with its exact ranking. The raw results and hardware are in `results/scale_1k.json`, `results/scale_10k.json`, and `results/scale_100k.json`. At 100k rows, indexed p50 was 62.9 ms versus exact p50 66.9 ms, with only 78.0% top-10 recall. The small latency gain does not justify that recall loss as a default. Bulk seed throughput is not public `add` throughput.

## Interpretation and next work

The development result at top 5 did not favor FTS over the file baseline, while the frozen top-3 configuration led by 15 questions on the held-out split. Report both; neither justifies a universal ranking. The fact-shaped ablation covers query-independent extraction and keyed ranking on the development split, not production `ModelExtractor` quality. A real-user study, human adjudication of judge disagreements, and a scale run using the public `add` method remain open work.
