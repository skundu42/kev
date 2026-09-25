"""Authenticated, single-worker HTTP inference over a local Kev export."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import secrets
import threading

from .core import InputTooLongError, parse_questions

MAX_BODY_BYTES = 256 * 1024
MAX_QUESTIONS = 16
BODY_TIMEOUT_SECONDS = 15
LOG = logging.getLogger("kev.api")


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


def create_app(model_path, api_key, pair_batch_size=8, *, model_loader=None):
    # Lazy imports preserve offline, dependency-free request validation tests.
    from fastapi import FastAPI, HTTPException, Request
    from starlette.concurrency import run_in_threadpool

    if not valid_api_key(api_key):
        raise ValueError("Set KEV_API_KEY to at least 32 printable ASCII characters without spaces")
    if pair_batch_size < 1:
        raise ValueError("pair_batch_size must be positive")
    if model_loader is None:
        from .inference import DecisionModel
        model_loader = DecisionModel.from_pretrained

    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.model = model_loader(model_path, pair_batch_size=pair_batch_size)
        try:
            yield
        finally:
            app.state.model = None

    app = FastAPI(title="Kev API", lifespan=lifespan, docs_url=None,
                  redoc_url=None, openapi_url=None)
    app.state.model = None

    @app.get("/healthz")
    async def health():
        if app.state.model is None:
            raise HTTPException(503, "Model is not ready")
        return {"status": "ready"}

    def infer(payload):
        # The worker owns this lock until CUDA inference really ends, even if a client
        # disconnects. Concurrent predictions fail quickly instead of building a queue.
        if not lock.acquire(blocking=False):
            raise HTTPException(503, "Model is busy; retry later", headers={"Retry-After": "1"})
        try:
            return app.state.model.predict(payload["state"], payload["questions"])
        except InputTooLongError as error:
            raise HTTPException(422, str(error)) from None
        except Exception as error:
            # Exception text can contain user content or local paths; do not log it.
            LOG.error("Prediction failed (%s)", type(error).__name__)
            raise HTTPException(500, "Inference failed; check the server") from None
        finally:
            lock.release()

    @app.post("/predict")
    async def predict(request: Request):
        if not authorized(request.headers.get("authorization"), api_key):
            raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})
        if app.state.model is None:
            raise HTTPException(503, "Model is not ready")
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
            payload = decode_request(body, app.state.model.config["max_candidates"])
        except (ValueError, RecursionError, UnicodeError):
            # Keep validation responses independent of user-provided field names/text.
            raise HTTPException(422, "Invalid request; check state, questions, criteria, and limits") from None
        return await run_in_threadpool(infer, payload)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local final/ export directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pair-batch-size", type=int, default=8)
    args = parser.parse_args()
    key = os.environ.get("KEV_API_KEY", "")
    if not valid_api_key(key):
        parser.error("Set KEV_API_KEY to a random token with at least 32 non-space ASCII characters")
    if args.pair_batch_size < 1 or not 1 <= args.port <= 65535:
        parser.error("pair-batch-size must be positive; port must be 1-65535")
    try:
        import uvicorn
        app = create_app(args.model, key, args.pair_batch_size)
    except ImportError:
        parser.error("Install the API dependencies on the pod: python -m pip install -r requirements-api.txt")
    # One worker owns one GPU model; no reload and no process-wide model copies.
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False,
                proxy_headers=False, limit_concurrency=32, timeout_keep_alive=5)


if __name__ == "__main__":
    main()
