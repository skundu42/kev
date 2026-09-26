"""Authenticated, single-worker HTTP inference over a local Kev export."""

import argparse
import asyncio
import json
import logging
import math
import os
import secrets
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .core import InputTooLongError, parse_questions
from .inference import (
    DEFAULT_MAX_BATCH_TOKENS,
    DEFAULT_PAD_MULTIPLE,
    DEFAULT_PAIR_BATCH_SIZE,
    DecisionModel,
    add_inference_arguments,
    inference_kwargs,
    validate_batch_options,
)

MAX_BODY_BYTES = 256 * 1024
MAX_QUESTIONS = 16
BODY_TIMEOUT_SECONDS = 15
LOG = logging.getLogger("kev.api")


@dataclass
class _PendingPrediction:
    payload: dict
    result: asyncio.Future
    started: asyncio.Future
    queued_at: float
    deadline: float


class _PredictionBatcher:
    def __init__(self, model, queue_capacity, max_batch_requests, batch_wait_ms, queue_timeout_ms):
        self.model = model
        self.queue_capacity = queue_capacity
        self.max_batch_requests = max_batch_requests
        self.batch_wait = batch_wait_ms / 1000
        self.queue_timeout = queue_timeout_ms / 1000
        self.pending = deque()
        self.available = asyncio.Event()
        self.closing = False
        self.worker = asyncio.create_task(self._run())

    async def predict(self, payload):
        if self.closing:
            return 503, "Model is shutting down; retry later"
        if len(self.pending) >= self.queue_capacity:
            return 503, "Prediction queue is full; retry later"
        loop = asyncio.get_running_loop()
        now = loop.time()
        request = _PendingPrediction(payload, loop.create_future(), loop.create_future(),
                                     now, now + self.queue_timeout)
        self.pending.append(request)
        self.available.set()
        try:
            try:
                async with asyncio.timeout(self.queue_timeout):
                    await asyncio.shield(request.started)
            except TimeoutError:
                return 503, "Prediction queue wait expired; retry later"
            return await request.result
        finally:
            # Cancelled clients release queue space without cancelling a running GPU batch.
            if request in self.pending:
                self.pending.remove(request)
                self.available.set()
            if not request.started.done():
                request.started.cancel()
            if not request.result.done():
                request.result.cancel()

    @staticmethod
    def _failure(error):
        if isinstance(error, InputTooLongError):
            return 422, str(error)
        # Exception text can contain user content or local paths; do not log it.
        LOG.error("Prediction failed (%s)", type(error).__name__)
        return 500, "Inference failed; check the server"

    def _infer(self, payloads):
        outcomes = [None] * len(payloads)
        prepared, indices = [], []
        for index, payload in enumerate(payloads):
            try:
                prepared.append(self.model.prepare_request(payload["state"], payload["questions"]))
                indices.append(index)
            except Exception as error:
                outcomes[index] = self._failure(error)
        if prepared:
            try:
                responses = self.model.predict_prepared(prepared)
                if len(responses) != len(indices):
                    raise ValueError("Prediction response count differs from request count")
                for index, response in zip(indices, responses, strict=True):
                    outcomes[index] = 200, response
            except Exception as error:
                failure = self._failure(error)
                for index in indices:
                    outcomes[index] = failure
        return outcomes

    @staticmethod
    def _finish(request, outcome):
        if not request.started.done():
            request.started.set_result(None)
        if not request.result.done():
            request.result.set_result(outcome)

    async def _run(self):
        while not self.closing:
            if not self.pending:
                self.available.clear()
                await self.available.wait()
                continue
            loop = asyncio.get_running_loop()
            wait = self.pending[0].queued_at + self.batch_wait - loop.time()
            if len(self.pending) < self.max_batch_requests and wait > 0:
                self.available.clear()
                try:
                    async with asyncio.timeout(wait):
                        await self.available.wait()
                except TimeoutError:
                    pass
                continue
            batch = []
            while self.pending and len(batch) < self.max_batch_requests:
                request = self.pending.popleft()
                if request.result.done():
                    continue
                if request.deadline <= loop.time():
                    self._finish(request, (503, "Prediction queue wait expired; retry later"))
                else:
                    batch.append(request)
            if batch:
                for request in batch:
                    request.started.set_result(None)
                outcomes = await asyncio.to_thread(self._infer, [request.payload for request in batch])
                for request, outcome in zip(batch, outcomes, strict=True):
                    self._finish(request, outcome)

    async def close(self):
        self.closing = True
        while self.pending:
            self._finish(self.pending.popleft(), (503, "Model is shutting down; retry later"))
        self.available.set()
        # The model must stay alive until the thread finishes, even on client cancellation.
        await asyncio.shield(self.worker)


def decode_request(body, max_candidates):
    def invalid_constant(value):
        raise ValueError("JSON must not contain NaN or Infinity")

    try:
        payload = json.loads(body.decode("utf-8"), parse_constant=invalid_constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid UTF-8 JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"state", "questions"}:
        raise ValueError("Request must contain exactly state and questions")
    if not isinstance(payload["state"], (str, dict, list)):
        raise ValueError("state must be a string, object, or array")
    questions = payload["questions"]
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f"Expected 1-{MAX_QUESTIONS} questions")
    try:
        parse_questions(payload["state"], questions, max_candidates)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid request structure") from error
    return payload


def valid_api_key(key):
    return (isinstance(key, str) and len(key) >= 32 and key.isascii()
            and all(33 <= ord(c) <= 126 for c in key))


def authorized(header, key):
    scheme, _, token = (header or "").partition(" ")
    return scheme.lower() == "bearer" and secrets.compare_digest(token.encode("utf-8"), key.encode("ascii"))


async def _wait_for_disconnect(request):
    while (await request.receive())["type"] != "http.disconnect":
        pass


def create_app(model_path, api_key, pair_batch_size=DEFAULT_PAIR_BATCH_SIZE, *,
               max_batch_tokens=DEFAULT_MAX_BATCH_TOKENS,
               weight_dtype="float32", attn_implementation="sdpa", compile_model=False,
               compile_mode="default", pad_to_multiple_of=DEFAULT_PAD_MULTIPLE, group_by_length=True,
               queue_capacity=32, max_batch_requests=8, batch_wait_ms=0,
               queue_timeout_ms=1000, model_loader=None):
    # Lazy imports preserve offline, dependency-free request validation tests.
    from fastapi import FastAPI, HTTPException, Request  # noqa: PLC0415

    if not valid_api_key(api_key):
        raise ValueError("Set KEV_API_KEY to at least 32 printable ASCII characters without spaces")
    validate_batch_options(pair_batch_size, max_batch_tokens, pad_to_multiple_of)
    for name, value in (("queue_capacity", queue_capacity), ("max_batch_requests", max_batch_requests)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("batch_wait_ms", batch_wait_ms), ("queue_timeout_ms", queue_timeout_ms)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0 or (name == "queue_timeout_ms" and value == 0):
            raise ValueError(f"{name} must be {'positive' if name == 'queue_timeout_ms' else 'nonnegative'}")
    if model_loader is None:
        model_loader = DecisionModel.from_pretrained

    @asynccontextmanager
    async def lifespan(app):
        app.state.model = model_loader(model_path, pair_batch_size=pair_batch_size,
                                       max_batch_tokens=max_batch_tokens, weight_dtype=weight_dtype,
                                       attn_implementation=attn_implementation, compile_model=compile_model,
                                       compile_mode=compile_mode, pad_to_multiple_of=pad_to_multiple_of,
                                       group_by_length=group_by_length)
        app.state.batcher = _PredictionBatcher(app.state.model, queue_capacity, max_batch_requests,
                                               batch_wait_ms, queue_timeout_ms)
        try:
            yield
        finally:
            try:
                await app.state.batcher.close()
            finally:
                app.state.model = None
                app.state.batcher = None

    app = FastAPI(title="Kev API", lifespan=lifespan, docs_url=None,
                  redoc_url=None, openapi_url=None)
    app.state.model = None
    app.state.batcher = None

    @app.get("/healthz")
    async def health():
        if app.state.model is None or app.state.batcher.closing:
            raise HTTPException(503, "Model is not ready")
        return {"status": "ready"}

    @app.post("/predict")
    async def predict(request: Request):
        if not authorized(request.headers.get("authorization"), api_key):
            raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})
        if app.state.model is None or app.state.batcher.closing:
            raise HTTPException(503, "Model is not ready")
        batcher = app.state.batcher
        max_candidates = app.state.model.config["max_candidates"]
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(415, "Use Content-Type: application/json")
        length = request.headers.get("content-length")
        if length is not None:
            if not length.isascii() or not length.isdecimal():
                raise HTTPException(400, "Invalid Content-Length")
            if len(length) > 10 or int(length) > MAX_BODY_BYTES:
                raise HTTPException(413, "Request exceeds 256 KiB")
        body = bytearray()
        try:
            async with asyncio.timeout(BODY_TIMEOUT_SECONDS):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > MAX_BODY_BYTES:
                        raise HTTPException(413, "Request exceeds 256 KiB")
                    body.extend(chunk)
        except TimeoutError:
            raise HTTPException(408, "Request body timed out") from None
        try:
            payload = decode_request(body, max_candidates)
        except (ValueError, RecursionError, UnicodeError):
            # Keep validation responses independent of user-provided field names/text.
            raise HTTPException(422, "Invalid request; check state, questions, criteria, and limits") from None
        # ASGI servers report disconnects through receive without cancelling the handler.
        prediction = asyncio.create_task(batcher.predict(payload))
        disconnect = asyncio.create_task(_wait_for_disconnect(request))
        try:
            done, _ = await asyncio.wait((prediction, disconnect), return_when=asyncio.FIRST_COMPLETED)
            if disconnect in done:
                await disconnect
                raise HTTPException(499, "Client disconnected")
            status, result = await prediction
        finally:
            prediction.cancel()
            disconnect.cancel()
            await asyncio.gather(prediction, disconnect, return_exceptions=True)
        if status != 200:
            raise HTTPException(status, result, headers={"Retry-After": "1"} if status == 503 else None)
        return result

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local final/ export directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    add_inference_arguments(parser)
    parser.add_argument("--queue-capacity", type=int, default=32,
                        help="Maximum requests waiting for the model worker")
    parser.add_argument("--max-batch-requests", type=int, default=8,
                        help="Maximum requests merged into one inference batch")
    parser.add_argument("--batch-wait-ms", type=float, default=0,
                        help="Maximum time to collect peers after the oldest request arrives")
    parser.add_argument("--queue-timeout-ms", type=float, default=1000,
                        help="Maximum queue wait before inference starts; running inference is not timed out")
    args = parser.parse_args()
    key = os.environ.get("KEV_API_KEY", "")
    if not valid_api_key(key):
        parser.error("Set KEV_API_KEY to a random token with at least 32 non-space ASCII characters")
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1-65535")
    try:
        import uvicorn  # noqa: PLC0415
        app = create_app(args.model, key, **inference_kwargs(args),
                         queue_capacity=args.queue_capacity, max_batch_requests=args.max_batch_requests,
                         batch_wait_ms=args.batch_wait_ms, queue_timeout_ms=args.queue_timeout_ms)
    except ImportError:
        parser.error("Install the API dependencies on the pod: python -m pip install -r requirements-api.txt")
    except ValueError as error:
        parser.error(str(error))
    # One worker owns one GPU model; no reload and no process-wide model copies.
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False,
                proxy_headers=False,
                limit_concurrency=max(32, args.queue_capacity + args.max_batch_requests + 8),
                timeout_keep_alive=5)


if __name__ == "__main__":
    main()
