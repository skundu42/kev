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

HTTP requests are limited to 256 KiB and 16 questions, with the exported model's candidate and token limits (normally 16 candidates and 1,024 tokens **per encoded prompt/candidate pair**). Inputs are never silently truncated. Inference runs outside the HTTP event loop and only one prediction executes at a time; a competing request gets `503` with `Retry-After: 1`. Retry with backoff. `--pair-batch-size` controls candidate pairs per GPU forward pass and defaults to 8; it does not change request limits.

| Status | Meaning |
|---|---|
| `200` | Prediction succeeded, or model is ready for `/healthz`. |
| `401` | Missing or incorrect bearer token. |
| `408` / `413` / `415` | Body took too long to arrive, exceeded the size limit, or was not JSON. |
| `422` | Invalid request schema/criteria or an overlength decision. |
| `503` | Model is not ready, is busy, or the server's connection limit was reached. |
| `500` | Internal inference failure; response does not expose internal exception details. |

Authentication, request limits, concurrency, lifecycle, and all three output types are tested locally with a synthetic model and network access blocked. Real GPU HTTP inference must still be verified on the pod with the curl commands above.
