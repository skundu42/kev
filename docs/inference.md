# Inference and HTTP API

[Project overview](../README.md)

Run these commands from the repository root on a configured GPU pod with a completed local export. See [training](training.md) for the runtime and export workflow. Examples use `runs/demo/final`; substitute your run name as needed.

## CLI and Python

The direct CLI reads one JSON request from a file or stdin:

```bash
python -m kev.inference --model /workspace/kev-run/runs/demo/final --input examples/request.json
cat examples/request.json | python -m kev.inference --model /workspace/kev-run/runs/demo/final
```

The same interface is available from Python on the pod:

```python
import json
from pathlib import Path
from kev.inference import DecisionModel

request = json.loads(Path("examples/request.json").read_text())
model = DecisionModel.from_pretrained("/workspace/kev-run/runs/demo/final")
result = model.predict(request["state"], request["questions"])
print(json.dumps(result, indent=2))
```

Candidate pairs are grouped by encoded length before each forward pass, with scores restored to the original question and candidate order. Defaults are eight pairs and at most 8,192 **padded tokens** per forward; long pairs automatically reduce the batch size. The token budget must fit at least one pair at the export's maximum length. Scores stay on the GPU until one CPU transfer completes the request or combined request group.

The CLI, Python loader, API, calibration, and evaluation accept the same runtime settings:

| CLI option | Python argument | Default |
|---|---|---|
| `--pair-batch-size` | `pair_batch_size` | `8` |
| `--max-batch-tokens` | `max_batch_tokens` | `8192` |
| `--no-group-by-length` | `group_by_length=False` | Grouping enabled |
| `--weight-dtype` | `weight_dtype` | `float32` |
| `--attn-implementation` | `attn_implementation` | `sdpa` |
| `--compile` | `compile_model=True` | Disabled |
| `--compile-mode` | `compile_mode` | `default` |
| `--pad-to-multiple-of` | `pad_to_multiple_of` | `1` (exact padding) |

`bfloat16` loads parameters directly in BF16; `auto` selects BF16 on supported CUDA devices and FP32 otherwise. FP32 retains BF16 autocast where supported. Explicit BF16 fails on unsupported hardware. Precision changes can move probabilities, so compare the saved-temperature metrics before adopting them. `flash_attention_2` requires BF16 weights and a compatible installed FlashAttention package; unavailable backends fail explicitly.

Compilation is opt-in. Sequence lengths round up to the configured multiple, capped at the export's token limit; compiled batches also pad pair counts to bounded power-of-two buckets. Duplicated padding pairs do not appear in the result, and their tokens count toward the budget. New shapes can trigger expensive compilation or reach PyTorch's recompilation limit. Warm representative shapes before serving and compare cold and steady timings using the [benchmark command](#benchmark-runtime-settings).

```bash
python -m kev.inference --model /workspace/kev-run/runs/demo/final \
  --input examples/request.json --pair-batch-size 32 --max-batch-tokens 16384 \
  --weight-dtype bfloat16
```

Python callers can combine requests with `model.predict_many([request_a, request_b])`; responses remain in input order.

Requests contain `state` and a `questions` object. Each question has `type`, `instructions`, and criteria:

```json
{
  "state": "The delivery arrived two days early and everything worked.",
  "questions": {
    "sentiment": {
      "type": "choice",
      "instructions": "Classify the review sentiment.",
      "criteria": {"negative": null, "neutral": null, "positive": null}
    },
    "satisfied": {
      "type": "noul",
      "instructions": "Is the reviewer satisfied?",
      "criteria": {"false": "No", "true": "Yes"}
    },
    "rating": {
      "type": "score",
      "instructions": "Rate the review sentiment.",
      "criteria": ["negative", "neutral", "positive"]
    }
  }
}
```

- `choice`: returns the selected criterion name and probabilities over the supplied criteria.
- `noul`: returns the probability of `true`, between 0 and 1. Its criteria are exactly `false` and `true`; omitting them defaults to No/Yes.
- `score`: returns the expected **zero-based criterion index** and its distribution, not an independently learned continuous scale. A five-point rubric has scores from 0 to 4.
- For choice/score, `confidence = (K × max_probability − 1) / (K − 1)`: zero at a uniform distribution and one at certainty. This is a confidence summary, not a validated probability of correctness.

## HTTP API

The API exposes `POST /predict` with the same JSON request and `{"answers": ...}` response as the Python interface, including choice, noul, and score. It loads one **local `final/` export** and its saved calibration temperature at startup. It requires a CUDA GPU and the container's existing inference dependencies; it does not download a model, prepare data, or invoke Halo training.

Run these commands **on the GPU pod containing your export**. Start from the container's Python environment (leave the comparison/data virtual environment first if active). Install the small API requirements in an isolated environment that inherits the container's Torch/CUDA packages:

```bash
cd /workspace/kev
git pull --ff-only
python -m venv --system-site-packages /workspace/kev-run/venv-api
source /workspace/kev-run/venv-api/bin/activate
python -m pip install -r requirements-api.txt
tmux new -s kev-api
```

Inside tmux, select the completed run and create a persistent bearer token. Replace the run name below if your export is elsewhere:

```bash
cd /workspace/kev
source /workspace/kev-run/venv-api/bin/activate
export KEV_WORKDIR=/workspace/kev-run
export KEV_RUN_NAME=demo
umask 077
if [[ ! -s "$KEV_WORKDIR/api.key" ]]; then
  python -c 'import secrets; print(secrets.token_urlsafe(32))' > "$KEV_WORKDIR/api.key"
fi
export KEV_API_KEY="$(cat "$KEV_WORKDIR/api.key")"
bash scripts/runpod.sh serve --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete`. Detach with **Ctrl+B, then D**; reconnect with `tmux attach -t kev-api`. The pod must keep running. Stop the server with Ctrl+C inside that session. One process owns the model; do not launch multiple workers or a second server on the same GPU. Without `--host`, the server binds only to `127.0.0.1`.

From a second pod terminal, test readiness and all three decision types:

```bash
cd /workspace/kev
export KEV_API_KEY="$(cat /workspace/kev-run/api.key)"
curl --fail-with-body http://127.0.0.1:8000/healthz
curl --fail-with-body http://127.0.0.1:8000/predict \
  -H "Authorization: Bearer $KEV_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

To call it from your application, expose **HTTP port 8000** in the pod settings and use the HTTPS endpoint `https://<POD_ID>-8000.proxy.runpod.net/predict`, following [Runpod's HTTP connection guide](https://docs.runpod.io/pods/connect-to-a-pod). Send the same bearer token from your backend; keep it out of browser code and version control. Use HTTPS for remote calls. `/healthz` is unauthenticated and reports readiness after the model is loaded; it does not perform a prediction.

HTTP requests are limited to 256 KiB and 16 questions, with the exported model's candidate and token limits (normally 16 candidates and 1,024 tokens **per encoded prompt/candidate pair**). Inputs are never silently truncated. A single worker runs inference outside the HTTP event loop and merges queued requests into shared GPU batches. Invalid or overlength requests fail independently of valid peers.

| Queue option | Default | Meaning |
|---|---|---|
| `--queue-capacity` | `32` | Maximum requests waiting for the worker. |
| `--max-batch-requests` | `8` | Maximum requests combined by one worker invocation. |
| `--batch-wait-ms` | `0` | Optional collection window measured from the oldest queued arrival. |
| `--queue-timeout-ms` | `1000` | Maximum wait until the worker starts preparation/inference. |

The default merges requests already queued when the worker becomes available, without an intentional collection delay. A positive window (for example, `--batch-wait-ms 2`) adds latency to idle requests to collect nearby arrivals. Full or expired queues return `503` with `Retry-After: 1`; retry with backoff. Running inference is not cancelled by the queue deadline or by a disconnected caller. Shutdown rejects queued work and drains the active batch. Set `--max-batch-requests 1 --batch-wait-ms 0` to disable merging while retaining the bounded queue. Queue options have equivalent underscored arguments in `create_app`.

| Status | Meaning |
|---|---|
| `200` | Prediction succeeded, or model is ready for `/healthz`. |
| `401` | Missing or incorrect bearer token. |
| `408` / `413` / `415` | Body took too long to arrive, exceeded the size limit, or was not JSON. |
| `422` | Invalid request schema/criteria or an overlength decision. |
| `503` | Model is not ready, the queue is full or expired, shutdown has begun, or the connection limit was reached. |
| `500` | Internal inference failure; response does not expose internal exception details. |

Offline tests cover authentication, request limits, concurrency, disconnects, lifecycle, and all three output types. The [RTX 4090 measurements](benchmarks/README.md#cuda-inference-optimization) also exercise this API with real Kev weights through in-process HTTP. Verify the network deployment on your pod with the curl commands above.

## Benchmark runtime settings

Run on an idle GPU with a local Kev export and real requests. A request input file is JSONL with one `{ "state": ..., "questions": ... }` object per line:

```bash
python -m kev.benchmark --model /workspace/kev-run/runs/demo/final \
  --input requests.jsonl --output /workspace/kev-run/benchmarks/runtime.json \
  --batch-sizes 8 16 32 --request-batch-sizes 1 4 --max-batch-tokens 8192 \
  --flash-attention
```

For probability and calibration checks, replace `--input requests.jsonl` with `--data /workspace/kev-run/data/mixture-v1/test.jsonl`. Prepared rows form groups of five decisions by default (`--decisions-per-request`); their text and labels are preserved. Defaults use the first 100 request groups, one additional warmup pass after the first full pass, and three measured repetitions. Use a representative input sample or `--max-requests 0` for all rows.

For compilation, add `--compile --pad-to-multiple-of 256` to test a small set of sequence-length buckets. This adds padding work; compare against exact padding and retain the separate warmup timings.

The sweep compares FP32/BF16 at 8/16/32 pairs and combines one/four requests; optional compiled and FlashAttention runs use the largest selected pair batch. Repeat with a fresh output path and another token budget to test memory limits. The reference is ungrouped FP32/SDPA with eight pairs and exact-length padding, using the same score-transfer implementation as the other variants.

Reports include synchronized median/p95 combined-batch latency, throughput, load and warmup times, peak allocated/reserved VRAM, probability drift, changed winning candidates, and labelled calibration metrics with the export's existing temperature. No temperature is fitted on test data. Combined-batch latency applies to every participating request; amortized time per request is a throughput measure and excludes HTTP queueing/network latency. Failed configurations remain in the report and produce a nonzero exit status. Existing report paths are never overwritten.
