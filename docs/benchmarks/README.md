# Benchmark results and methodology

[Project overview](../../README.md)

Results describe several different experiments. Keep their datasets, runtimes, scoring rules and evidence status separate. Aggregate in-domain wins do not establish general superiority; the external benchmark exposed substantial failures.

## Evidence and artifacts

| Experiment | Status | Evidence |
|---|---|---|
| Published Kev calibration/test evaluation | Historical published result; not rerun here | [Summary](published-evaluation.json) |
| Kev vs Laya English, 17,476 rows | Historical paired CUDA run; published artifact inspected | [Report](laya-full.json) |
| Kev vs Jev, 100 rows | Completed source-balanced pilot, seed 42 | [Report](jev-pilot.json) |
| Kev vs Jev, 17,476 rows | Completed paired evaluation | [Report](jev-full.json) |
| Luni benchmark suite | Completed Kev-only evaluation, concurrency 1 | [Report](luni.json) |
| CUDA inference optimization | Real Kev on RTX 4090; latency, precision, compilation, and API batching | [Report](inference-rtx4090.json) |

Committed JSON files contain aggregate results, revisions and input hashes. Machine-local paths have been removed; raw examples, prediction logs and experimental Mac runners remain in ignored `runs/` directories and are not included in this commit. The maintained CUDA/Laya runner is [`kev.compare`](../../kev/compare.py); see the [evaluation guide](../evaluation.md). Model/data revision pins identify the evaluated artifacts, but aggregate summaries alone cannot recreate every request.

The Jev and Luni runs used the same Kev checkpoint `aa9eb8668e2ab5675da1479993ae069c28124f10`, temperature 1.3734692667114208, Apple M3 Pro, MPS FP32, PyTorch 2.13.0 and Transformers 5.3.0. The experimental local adapter retained Kev formatting, calibration and limits; the supported production loader requires CUDA. No benchmark test targets entered model inference or calibration.

## CUDA inference optimization

Measured on 26 September 2026 using the real `skundu42/kev` checkpoint above, RTX 4090, PyTorch 2.11.0+cu130, and Transformers 5.16.1. The latency workload contains 40 real `LocalLLaMA/typed-decisions` test cases: ten per workflow, 200 decisions, 710 candidate pairs. Each configuration has a full first pass, an additional warmup pass, and three measured passes. Timings include tokenization and synchronized GPU/CPU scores. [Input revisions, hashes, all configurations, memory, and probability deltas](inference-rtx4090.json) are retained.

| Single-request configuration | Median ms | p95 ms |
|---|---:|---:|
| Original implementation, eight pairs | 80.37 | 102.27 |
| Consolidated score transfer and asynchronous input copies | 78.17 | 99.76 |
| Eight pairs, length grouping, exact padding (default) | **77.90** | **96.90** |
| 32 pairs, exact padding, 16,384-token budget | 73.09 | 105.90 |
| BF16 + compilation, 32 pairs, 256-token buckets, 8,192-token budget | 69.45 | 82.82 |

The default retains eight pairs because the larger batch's median gain came with worse tail latency. Its 8,192-token budget does not bind on this sample. Transfer changes alone produced identical probabilities. Length grouping changed three winners out of 1,000 source-balanced held-out decisions: accuracy 80.1% versus 80.0%, Brier 0.282164 versus 0.281977, and ECE 0.026221 versus 0.028768. These small differences are drift measurements, not evidence of an accuracy gain; no temperature was refitted.

With 32 pairs and 64-token buckets, direct BF16 weights reduced peak allocated VRAM from **1.79 to 1.06 GiB**. In the separate 1,000-decision, exact-padding check, BF16/SDPA retained 80.0% accuracy, changed four winners, increased Brier by 0.000489 and ECE by 0.000584, and had mean absolute probability drift 0.00305. BF16 remains opt-in. Eager FlashAttention was slower than SDPA on the typed sample: 83.11 versus 75.67 ms median with BF16 weights.

Compilation with 64-token buckets sometimes reached PyTorch's recompilation limit; those timings include eager fallback and the initial FP32 first pass took 182 seconds. The 256-token profile in the table captured four graphs with no graph breaks or recompilation-limit warnings. Its first pass took 34.7 seconds **with the compiler's disk cache already warm from the earlier sweep**. It changed three winners among 200 decisions relative to eager BF16 with the same padding. Compilation remains opt-in; these warmup costs and probability changes matter when selecting a serving profile.

The real HTTP app was also exercised through in-process ASGI, including authentication, JSON serialization, queueing, and model inference. At eight concurrent clients and **zero collection delay**, merging up to eight requests improved throughput **10.22 → 14.16 requests/sec (+38.5%)** and p95 **904.6 → 643.5 ms (−28.9%)** compared with the same queue processing one request at a time. All 1,560 measured requests across the two queue sweeps returned 200. These results exclude socket, TLS, Uvicorn connection-limit, and network overhead. The zero-delay default batches existing queued work; a configurable positive collection window trades idle-request latency for nearby arrivals.

Use the [runtime benchmark commands](../inference.md#benchmark-runtime-settings) on representative inputs before changing precision or batch limits. The compiled profile measured above is:

```bash
python -m kev.serve --model /path/to/final \
  --weight-dtype bfloat16 --pair-batch-size 32 --max-batch-tokens 8192 \
  --compile --pad-to-multiple-of 256
```

This command also requires `KEV_API_KEY` as described in the [API guide](../inference.md#http-api).

## Published Kev evaluation

The historical export evaluation covers 17,476 in-domain held-out decisions. Temperature was fitted separately on 16,267 calibration decisions.

| Metric | Uncalibrated | Calibrated |
|---|---:|---:|
| Accuracy | 79.86% | 79.86% |
| Log loss | 0.492302 | 0.474615 |
| Brier score | 0.280210 | 0.276855 |
| ECE-15, target mass | 0.038409 | 0.014262 |
| Ordinal MAE, 997 rows | 0.372761 | 0.392342 |

Calibration did not change argmax accuracy and increased ordinal MAE while improving log loss, Brier and ECE. These figures are separate from the later CUDA comparison and MPS run; small numerical differences across runs are retained rather than merged.

## Paired Kev vs Laya English

The published full comparison ran sequentially on an NVIDIA RTX 4090, PyTorch 2.11.0+cu130, Laya 0.3.20. This is an inspected historical report, not a new Laya run. Laya weights: `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`; runtime: `970dc8c5f63d7b886a68409493f37d569424f933`.

| Metric | Kev | Laya English |
|---|---:|---:|
| Accuracy | 79.93% | 51.63% |
| Log loss | 0.474736 | 1.246543 |
| Brier score | 0.276944 | 0.657224 |
| ECE-15, target mass | 0.014998 | 0.205194 |
| Ordinal MAE | 0.391878 | 0.826489 |
| Median latency (ms) | 17.83 | 20.59 |
| p95 latency (ms) | 22.65 | 22.05 |
| Decisions/second | 52.45 | 48.55 |

Laya's native limits altered **1,597** inputs. On the same **15,879 unaltered rows**, accuracy was **79.44% Kev** versus **52.93% Laya**. Formatting, precision, shipped calibration and probability rounding differ; training overlap is unknown.

An earlier 1,000-row Laya pilot was recorded as 80.4% vs 55.5% accuracy, with median latency 18.406 vs 21.738 ms. Its raw report is not retained here; treat those as historical reported observations, not separately verified evidence or an independent replication. The full published JSON above is the auditable result.

## Paired Kev vs Jev

Both models received the same pinned Kev test partition, `skundu42/kev-prepared@5cea4539b6676edfb4fd1fb60878e9aee602b0cd`. Jev served `typesafe/jev-1.13-20260917` through OpenRouter. Kev had in-domain training and calibration; Jev training overlap is unknown. The 100-row pilot selected 10 examples/source with seed 42; it is a subset of the full test pool, not independent evidence.

| Run | Rows | Kev accuracy | Jev accuracy | Kev median / p95 | Jev median / p95 |
|---|---:|---:|---:|---:|---:|
| Pilot | 100 | 80.00% | 81.00% | 129.7 / 401.4 ms | 390.4 / 529.5 ms |
| Full | 17,476 | 79.96% | 73.54% | 132.7 / 607.3 ms | 388.4 / 499.5 ms |

| Full-run metric | Kev | Jev |
|---|---:|---:|
| Brier score | 0.276888 | 0.382336 |
| ECE-15, target mass | 0.015148 | 0.097899 |
| Ordinal MAE, 997 rows | 0.392345 | 0.356360 |
| Raw log loss, floor 1e-300 | 0.474689 | 9.942500 |
| Log loss, floor 1e-6 | 0.474689 | 0.839409 |

Kev had 1,121 more accepted answers (+6.41 percentage points); Jev had lower ordinal error and lower p95 request latency. Of the paired outcomes, both were correct on 11,070 rows, only Kev on 2,903, only Jev on 1,782, and neither on 1,721.

| Source | Rows | Kev accuracy | Jev accuracy |
|---|---:|---:|---:|
| Rowan/hellaswag | 2,000 | 86.15% | 93.45% |
| facebook/anli | 2,000 | 58.10% | 74.35% |
| google/boolq | 828 | 86.23% | 91.79% |
| nyu-mll/multi_nli | 2,000 | 90.75% | 86.00% |
| tasksource/FOL-nli | 2,000 | 83.20% | 60.30% |
| tasksource/bigbench | 1,896 | 76.37% | 71.15% |
| tasksource/defeasible-nli | 2,000 | 87.40% | 88.80% |
| tasksource/doc-nli | 1,776 | 86.82% | 86.20% |
| tasksource/tasksource-instruct-v0 | 997 | 71.41% | 67.20% |
| tasksource/zero-shot-label-nli | 1,979 | 73.02% | 24.46% |

Jev won on HellaSwag, ANLI, BoolQ and defeasible NLI. Kev’s aggregate advantage was concentrated particularly in FOL-NLI and zero-shot-label-NLI.

### Timing and probability caveats

- Pilot: one request at a time for each model. Full run: Kev sequential, Jev four HTTP workers; both workloads overlapped on the desktop. These are per-request timings, not equal-concurrency service benchmarks.
- Kev timings include tokenization and GPU completion; Jev includes network/provider time. Model load and warmups are excluded. Background load and thermal conditions were not controlled.
- Kev’s full-run mean was 220.7 ms, versus a 132.7 ms median. Summed measured inference took 64.29 minutes for 54,402 prompt–candidate pairs; this is not a separately measured end-to-end wall timer.
- All 17,476 rows ultimately completed. Twenty-six missing Jev rows were recovered after the first pass. Failed-attempt time is excluded from successful recovery latency. Reported API cost was $0.343704 for recorded full-run predictions, excluding warmups and unrecorded retry charges; the pilot cost $0.001934.
- Jev’s hundredth-rounded probabilities were renormalized within accumulated rounding tolerance. Rounded zeroes dominate raw log loss; both raw and 1e-6-floored values are shown. The pilot used the stricter original probability-sum check.

## External evaluation: Luni benchmark suite

The [Luni repository](https://huggingface.co/datasets/Luni/laya-jev-benchmark/tree/d75081b2a4b2ad772793d6a7f5f5b4fdca00d557) contains scripts and reference figures, not dataset rows. Kev was evaluated on the underlying datasets using those prompts, its native input formatting and existing calibration. One case/request ran at a time, with no truncation or dropped cases. No Platt fitting, fine-tuning or threshold search was performed.

| Dataset / metric | Kev measured | Jev published reference, not rerun |
|---|---:|---:|
| PhishNChips core, 2,000 emails: accuracy | 50.25% | 62.6% |
| Phishing AUROC | 0.34821 | 0.689 |
| Phishing recall | 100.00% | 43.2% |
| Phishing precision | 50.13% | Not supplied |
| Typed decisions, 400 cases / 2,000 decisions: accuracy | 44.15% | 72.7% |
| Typed score MAE, 800 score decisions | 0.610459 | Not supplied |
| Typed scores within one level | 83.62% | Not supplied |

**Phishing failed operationally:** Kev flagged 1,995/2,000 emails as phishing: TP=1,000, FP=995, TN=5, FN=0. Its 100% recall is accompanied by a 99.5% false-positive rate and an AUROC below 0.5. Median/p95 latency was 123.4/202.6 ms per email.

Typed accuracy was 883/2,000 decisions. Median/p95 latency was 2,008.5/3,376.9 ms per five-question case. Measured inference across both data suites totaled 17.97 minutes. The published Jev latency references are 239 ms/email and 710 ms/case, from different hardware/service conditions.

| Typed workflow | Decisions | Kev accuracy |
|---|---:|---:|
| agent_trace_observability | 500 | 36.60% |
| customer_service | 500 | 48.40% |
| invoice_processing | 500 | 44.20% |
| security_incidents | 500 | 47.40% |

Fourteen additional requests implemented the small behavioral suite: grounding **4/5**, complement-consistency checks **3/3** under the author’s ±0.35 sum tolerance, and routing variants **1/3**. No grounding error had derived binary concentration ≥0.8. Passing the loose consistency threshold is not proof of logical consistency; these are hand-written diagnostics, not a broad benchmark.

The reference repository also reports raw Laya accuracy of 50.5% on phishing and 36.0% on typed decisions; its recalibrated/fine-tuned variants use different protocols and are not direct untouched-model comparisons. None of those models were run again here. Reference revisions and identical row order are not established, so external published comparisons are contextual rather than new paired results.

## Metric definitions and limits

- Kev-mixture accuracy accepts any candidate with positive gold target mass. Typed-decision accuracy instead requires the exact supplied gold label; phishing uses the binary gold label and a fixed p ≥ 0.5 threshold. Do not compare these percentages as if they used identical targets.
- Mixture Brier is summed squared error over the full target distribution. External typed Brier (0.297839) covers choice/noul only, matching the benchmark evaluator; phishing binary Brier (0.315730) uses `(p_yes − label)²`.
- Mixture ECE-15 compares confidence to target mass at the top candidate. External hard-label ECE-15 is 0.230164 for phishing and 0.124346 for typed decisions; choice/noul soft-target ECE is also available in JSON. Binning for published external ECE references was not verified, so these are not interchangeable calibration comparisons.
- Ordinal MAE measures error in expected rubric index; external typed scoring compares with the supplied numeric gold score. Accuracy uses the modal label, so better ordinal MAE can coexist with lower categorical accuracy.
- Phishing AUROC uses average ranks for ties and was independently checked by pairwise comparisons. All case IDs, probability distributions and hard-label accuracies were validated.
- These data do not establish unseen pretraining content, universal superiority, or deployment readiness. The external failures qualify the favorable in-domain comparison.

## Pinned external data

| Resource | Revision |
|---|---|
| Luni/laya-jev-benchmark | `d75081b2a4b2ad772793d6a7f5f5b4fdca00d557` |
| AreLit/PhishNChips | `89afcc39610084298c4679159cb2e27d9ffffa46` |
| LocalLLaMA/typed-decisions | `f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8` |
| skundu42/kev | `aa9eb8668e2ab5675da1479993ae069c28124f10` |
