"""Pure validation tests plus opt-in ASGI tests with an injected, offline fake model."""

import asyncio
import json
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

from kev.core import InputTooLongError, answer, parse_questions
from kev.inference import DEFAULT_MAX_BATCH_TOKENS, DEFAULT_PAD_MULTIPLE, DEFAULT_PAIR_BATCH_SIZE
from kev.serve import (
    MAX_BODY_BYTES,
    MAX_QUESTIONS,
    _PredictionBatcher,
    authorized,
    create_app,
    decode_request,
    valid_api_key,
)

REQUEST = json.loads((Path(__file__).resolve().parents[1] / "examples/request.json").read_text())
TOKEN = "test-only-token-" + "x" * 32
HEADERS = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}


def named_request(name):
    return {"state": name, "questions": {name: REQUEST["questions"]["route"]}}


async def wait_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(.001)


class RequestTests(unittest.TestCase):
    def test_native_shapes_and_malformed_requests(self):
        for state in (REQUEST["state"], {"text": "hello"}, ["first", "second"]):
            payload = {**REQUEST, "state": state}
            self.assertEqual(decode_request(json.dumps(payload).encode(), 16), payload)
        for raw in (b"[]", b"{}", b"not json", b"\xff", b'{"state":NaN,"questions":{}}',
                    json.dumps({**REQUEST, "model": "other"}).encode(),
                    json.dumps({**REQUEST, "state": None}).encode(),
                    json.dumps({**REQUEST, "questions": {}}).encode(),
                    json.dumps({**REQUEST, "questions": {str(i): REQUEST["questions"]["route"]
                                                       for i in range(MAX_QUESTIONS + 1)}}).encode()):
            with self.subTest(raw=raw[:40]), self.assertRaises(ValueError):
                decode_request(raw, 16)
        with self.assertRaises(ValueError):
            decode_request(json.dumps(REQUEST).encode(), 2)

    def test_authentication_has_no_optional_mode(self):
        self.assertTrue(valid_api_key(TOKEN))
        for key in (None, "", "short", "a" * 32 + "\n", "é" * 32):
            self.assertFalse(valid_api_key(key))
        self.assertTrue(authorized("bearer " + TOKEN, TOKEN))
        for header in (None, "", "Basic " + TOKEN, "Bearer wrong", "Bearer é"):
            self.assertFalse(authorized(header, TOKEN))


class FakeModel:
    config = {"max_candidates": 16}

    def __init__(self):
        self.calls = 0
        self.active = 0
        self.peak_active = 0
        self.batches = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.error = None
        self.prepare_errors = {}

    def prepare_request(self, state, questions):
        if isinstance(state, str) and state in self.prepare_errors:
            raise self.prepare_errors[state]
        rows, metadata = parse_questions(state, questions)
        return state, rows, metadata

    def predict_prepared(self, prepared):
        self.calls += 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        self.batches.append([state for state, _, _ in prepared])
        try:
            self.started.set()
            if self.block and not self.release.wait(5):
                raise RuntimeError("Test timed out")
            if self.error:
                raise self.error
            return [{"answers": {qid: answer(row["kind"], row["candidates"], keys,
                                               [1 / len(keys)] * len(keys))
                                  for row, (qid, keys) in zip(rows, metadata, strict=True)}}
                    for _, rows, metadata in prepared]
        finally:
            self.active -= 1


class AsgiRequest:
    def __init__(self, app, payload):
        self.incoming = asyncio.Queue()
        self.incoming.put_nowait({"type": "http.request", "body": json.dumps(payload).encode(),
                                  "more_body": False})
        self.messages = []
        self.receivers = 0
        self.cancelled_receives = 0
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                 "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/predict",
                 "raw_path": b"/predict", "query_string": b"", "root_path": "",
                 "headers": [(key.lower().encode(), value.encode()) for key, value in HEADERS.items()],
                 "client": ("127.0.0.1", 50000), "server": ("testserver", 80)}
        self.task = asyncio.create_task(app(scope, self.receive, self.send))

    async def receive(self):
        self.receivers += 1
        try:
            return await self.incoming.get()
        except asyncio.CancelledError:
            self.cancelled_receives += 1
            raise
        finally:
            self.receivers -= 1

    async def send(self, message):
        self.messages.append(message)

    def disconnect(self):
        self.incoming.put_nowait({"type": "http.disconnect"})

    async def status(self):
        await asyncio.wait_for(self.task, 2)
        return next(message["status"] for message in self.messages if message["type"] == "http.response.start")


class BatcherTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.model = FakeModel()

    def batcher(self, **options):
        settings = {"queue_capacity": 32, "max_batch_requests": 8,
                    "batch_wait_ms": 2, "queue_timeout_ms": 1000, **options}
        batcher = _PredictionBatcher(self.model, **settings)

        async def close():
            self.model.release.set()
            await batcher.close()

        self.addAsyncCleanup(close)
        return batcher

    async def wait_for_model(self):
        self.assertTrue(await asyncio.to_thread(self.model.started.wait, 2))

    async def test_collection_window_merges_requests_and_preserves_response_order(self):
        batcher = self.batcher(max_batch_requests=2, batch_wait_ms=500)
        first = asyncio.create_task(batcher.predict(named_request("first")))
        await wait_until(lambda: len(batcher.pending) == 1)
        self.assertFalse(first.done())
        self.assertEqual(self.model.calls, 0)
        second = asyncio.create_task(batcher.predict(named_request("second")))
        responses = await asyncio.wait_for(asyncio.gather(first, second), 2)
        self.assertEqual([status for status, _ in responses], [200, 200])
        self.assertEqual([list(body["answers"]) for _, body in responses], [["first"], ["second"]])
        self.assertEqual(self.model.batches, [["first", "second"]])

    async def test_collection_window_flushes_single_request(self):
        batcher = self.batcher(batch_wait_ms=10)
        status, result = await asyncio.wait_for(batcher.predict(named_request("alone")), 1)
        self.assertEqual(status, 200)
        self.assertEqual(list(result["answers"]), ["alone"])
        self.assertEqual(self.model.batches, [["alone"]])

    async def test_maximum_request_batch_size(self):
        batcher = self.batcher(max_batch_requests=2, batch_wait_ms=0)
        requests = [named_request(str(i)) for i in range(5)]
        results = await asyncio.gather(*(batcher.predict(payload) for payload in requests))
        self.assertTrue(all(status == 200 for status, _ in results))
        self.assertEqual(self.model.batches, [["0", "1"], ["2", "3"], ["4"]])
        self.assertEqual(self.model.peak_active, 1)

    async def test_queue_capacity_and_single_worker(self):
        self.model.block = True
        batcher = self.batcher(queue_capacity=1, max_batch_requests=1, batch_wait_ms=0)
        active = asyncio.create_task(batcher.predict(named_request("active")))
        await self.wait_for_model()
        queued = asyncio.create_task(batcher.predict(named_request("queued")))
        await wait_until(lambda: len(batcher.pending) == 1)
        status, detail = await batcher.predict(named_request("overflow"))
        self.assertEqual(status, 503)
        self.assertIn("full", detail)
        self.assertEqual(self.model.calls, 1)
        self.model.release.set()
        self.assertEqual([status for status, _ in await asyncio.gather(active, queued)], [200, 200])
        self.assertEqual(self.model.batches, [["active"], ["queued"]])
        self.assertEqual(self.model.peak_active, 1)

    async def test_queue_deadline_does_not_time_out_running_inference(self):
        self.model.block = True
        batcher = self.batcher(queue_capacity=1, max_batch_requests=1,
                               batch_wait_ms=0, queue_timeout_ms=20)
        active = asyncio.create_task(batcher.predict(named_request("active")))
        await self.wait_for_model()
        status, detail = await batcher.predict(named_request("expired"))
        self.assertEqual(status, 503)
        self.assertIn("expired", detail)
        self.assertFalse(active.done())
        self.assertEqual(len(batcher.pending), 0)
        self.model.release.set()
        self.assertEqual((await active)[0], 200)
        self.assertEqual(self.model.batches, [["active"]])

    async def test_cancellation_frees_queue_space_without_releasing_active_worker(self):
        self.model.block = True
        batcher = self.batcher(queue_capacity=1, max_batch_requests=1, batch_wait_ms=0)
        active = asyncio.create_task(batcher.predict(named_request("active")))
        await self.wait_for_model()
        queued = asyncio.create_task(batcher.predict(named_request("cancelled")))
        await wait_until(lambda: len(batcher.pending) == 1)
        queued.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await queued
        self.assertEqual(len(batcher.pending), 0)
        active.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await active
        replacement = asyncio.create_task(batcher.predict(named_request("replacement")))
        await wait_until(lambda: len(batcher.pending) == 1)
        self.assertEqual(self.model.calls, 1)
        self.model.release.set()
        self.assertEqual((await replacement)[0], 200)
        self.assertEqual(self.model.batches, [["active"], ["replacement"]])
        self.assertEqual(self.model.peak_active, 1)

    async def test_shutdown_rejects_waiting_requests_and_drains_active_inference(self):
        self.model.block = True
        batcher = self.batcher(max_batch_requests=1, batch_wait_ms=0)
        active = asyncio.create_task(batcher.predict(named_request("active")))
        await self.wait_for_model()
        queued = asyncio.create_task(batcher.predict(named_request("queued")))
        await wait_until(lambda: len(batcher.pending) == 1)
        closing = asyncio.create_task(batcher.close())
        self.assertEqual((await queued)[0], 503)
        self.assertFalse(closing.done())
        self.assertEqual((await batcher.predict(named_request("late")))[0], 503)
        self.model.release.set()
        self.assertEqual((await active)[0], 200)
        await closing
        self.assertTrue(batcher.worker.done())
        self.assertEqual(self.model.batches, [["active"]])

    async def test_shutdown_during_collection_does_not_start_inference(self):
        batcher = self.batcher(batch_wait_ms=500)
        waiting = asyncio.create_task(batcher.predict(named_request("queued")))
        await wait_until(lambda: len(batcher.pending) == 1)
        await batcher.close()
        self.assertEqual((await waiting)[0], 503)
        self.assertEqual(self.model.calls, 0)

    async def test_invalid_request_does_not_fail_its_batch_peers(self):
        self.model.prepare_errors["long"] = InputTooLongError(
            "overlength: decision exceeds 1024 tokens; no input was truncated")
        batcher = self.batcher(max_batch_requests=2, batch_wait_ms=500)
        long, valid = await asyncio.gather(batcher.predict(named_request("long")),
                                           batcher.predict(named_request("valid")))
        self.assertEqual(long[0], 422)
        self.assertIn("1024", long[1])
        self.assertEqual(valid[0], 200)
        self.assertEqual(list(valid[1]["answers"]), ["valid"])
        self.assertEqual(self.model.batches, [["valid"]])


@unittest.skipUnless(os.environ.get("KEV_TEST_API") == "1", "Set KEV_TEST_API=1 with API test dependencies installed")
class AsgiDisconnectTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.model = FakeModel()
        self.model.block = True
        self.app = create_app("/synthetic/final", TOKEN, model_loader=Mock(return_value=self.model),
                               queue_capacity=1, max_batch_requests=1, batch_wait_ms=0)
        await self.enterAsyncContext(self.app.router.lifespan_context(self.app))
        self.requests = []

        async def close_requests():
            self.model.release.set()
            for request in self.requests:
                request.task.cancel()
            await asyncio.gather(*(request.task for request in self.requests), return_exceptions=True)

        self.addAsyncCleanup(close_requests)

    def request(self, name):
        request = AsgiRequest(self.app, named_request(name))
        self.requests.append(request)
        return request

    async def wait_for_model(self):
        self.assertTrue(await asyncio.to_thread(self.model.started.wait, 2))

    async def test_http_disconnect_removes_queued_work_and_frees_capacity(self):
        active = self.request("active")
        await self.wait_for_model()
        abandoned = self.request("abandoned")
        await wait_until(lambda: len(self.app.state.batcher.pending) == 1)
        abandoned.disconnect()
        self.assertEqual(await abandoned.status(), 499)
        self.assertEqual(len(self.app.state.batcher.pending), 0)
        self.assertEqual(abandoned.receivers, 0)
        replacement = self.request("replacement")
        await wait_until(lambda: len(self.app.state.batcher.pending) == 1)
        self.assertEqual(self.model.calls, 1)
        self.model.release.set()
        self.assertEqual(await active.status(), 200)
        self.assertEqual(await replacement.status(), 200)
        self.assertEqual(self.model.batches, [["active"], ["replacement"]])
        self.assertEqual(self.model.peak_active, 1)
        for request in (active, replacement):
            self.assertEqual(request.receivers, 0)
            self.assertEqual(request.cancelled_receives, 1)

    async def test_http_disconnect_does_not_release_running_worker(self):
        active = self.request("active")
        await self.wait_for_model()
        active.disconnect()
        self.assertEqual(await active.status(), 499)
        self.assertEqual(self.model.active, 1)
        self.assertEqual(active.receivers, 0)
        replacement = self.request("replacement")
        await wait_until(lambda: len(self.app.state.batcher.pending) == 1)
        self.assertEqual(self.model.calls, 1)
        self.model.release.set()
        self.assertEqual(await replacement.status(), 200)
        self.assertEqual(self.model.batches, [["active"], ["replacement"]])
        self.assertEqual(self.model.peak_active, 1)

    async def test_handler_cancellation_cleans_up_disconnect_watcher(self):
        active = self.request("active")
        await self.wait_for_model()
        active.task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await active.task
        self.assertEqual(active.receivers, 0)
        self.assertEqual(active.cancelled_receives, 1)
        self.assertEqual(self.model.active, 1)

    async def test_queue_timeout_cleans_up_disconnect_watcher(self):
        self.app.state.batcher.queue_timeout = .02
        active = self.request("active")
        await self.wait_for_model()
        expired = self.request("expired")
        self.assertEqual(await expired.status(), 503)
        self.assertEqual(expired.receivers, 0)
        self.assertEqual(expired.cancelled_receives, 1)
        self.assertEqual(len(self.app.state.batcher.pending), 0)
        self.model.release.set()
        self.assertEqual(await active.status(), 200)
        self.assertEqual(self.model.batches, [["active"]])


@unittest.skipUnless(os.environ.get("KEV_TEST_API") == "1", "Set KEV_TEST_API=1 with API test dependencies installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient  # noqa: PLC0415
        self.model = FakeModel()
        self.loader = Mock(return_value=self.model)
        self.app = create_app("/synthetic/final", TOKEN, model_loader=self.loader)
        self.client = self.enterContext(TestClient(self.app))

    def test_startup_loads_once_and_serves_all_types(self):
        self.loader.assert_called_once_with(
            "/synthetic/final", pair_batch_size=DEFAULT_PAIR_BATCH_SIZE,
            max_batch_tokens=DEFAULT_MAX_BATCH_TOKENS,
            weight_dtype="float32", attn_implementation="sdpa", compile_model=False,
            compile_mode="default", pad_to_multiple_of=DEFAULT_PAD_MULTIPLE, group_by_length=True)
        self.assertEqual(self.client.get("/healthz").json(), {"status": "ready"})
        for state in (REQUEST["state"], {"content": "example"}):
            result = self.client.post("/predict", json={**REQUEST, "state": state}, headers=HEADERS)
            self.assertEqual(result.status_code, 200)
            answers = result.json()["answers"]
            self.assertEqual(answers["route"]["type"], "choice")
            self.assertEqual(answers["human_requested"]["noul"], .5)
            self.assertEqual(answers["frustration"]["score"], 1.)
        self.loader.assert_called_once()
        self.assertEqual(self.model.calls, 2)
        self.assertEqual(self.client.get("/docs").status_code, 404)

    def test_unauthorized_and_bad_inputs_never_invoke_model(self):
        self.assertEqual(self.client.post("/predict", content=b"invalid").status_code, 401)
        self.assertEqual(self.client.post("/predict", json=REQUEST,
                                        headers={"Authorization": "Bearer wrong"}).status_code, 401)
        self.assertEqual(self.client.post("/predict", content=b"{}",
                                        headers={"Authorization": "Bearer " + TOKEN}).status_code, 415)
        for body in (b"{", b"[]", b'{"state":Infinity,"questions":{}}'):
            self.assertEqual(self.client.post("/predict", content=body, headers=HEADERS).status_code, 422)
        self.assertEqual(self.client.post("/predict", json={**REQUEST, "questions": {}}, headers=HEADERS).status_code, 422)
        self.assertEqual(self.client.post("/predict", content=b"x" * (MAX_BODY_BYTES+1), headers=HEADERS).status_code, 413)
        # Generator sends chunked transfer with no Content-Length; actual bytes are still bounded.
        def chunks():
            yield b"x" * MAX_BODY_BYTES
            yield b"x"
        self.assertEqual(self.client.post("/predict", content=chunks(), headers=HEADERS).status_code, 413)
        self.assertEqual(self.model.calls, 0)

    def test_full_prediction_queue_does_not_block_health_or_overlap_inference(self):
        self.model.block = True
        self.app.state.batcher.queue_capacity = 1
        self.app.state.batcher.max_batch_requests = 1
        with ThreadPoolExecutor(max_workers=2) as pool:
            active = pool.submit(self.client.post, "/predict", json=REQUEST, headers=HEADERS)
            try:
                self.assertTrue(self.model.started.wait(2))
                self.assertEqual(self.client.get("/healthz").status_code, 200)
                queued = pool.submit(self.client.post, "/predict", json=REQUEST, headers=HEADERS)
                self.client.portal.call(wait_until, lambda: len(self.app.state.batcher.pending) == 1)
                busy = self.client.post("/predict", json=REQUEST, headers=HEADERS)
                self.assertEqual(busy.status_code, 503)
                self.assertIn("queue is full", busy.text)
                self.assertEqual(busy.headers["retry-after"], "1")
                self.assertEqual(self.model.calls, 1)
            finally:
                self.model.release.set()
            self.assertEqual(active.result(timeout=2).status_code, 200)
            self.assertEqual(queued.result(timeout=2).status_code, 200)
        self.assertEqual(self.model.peak_active, 1)
        self.assertEqual(self.client.post("/predict", json=REQUEST, headers=HEADERS).status_code, 200)

    def test_errors_keep_worker_usable_and_do_not_expose_internal_details(self):
        self.model.error = InputTooLongError("overlength: decision exceeds 1024 tokens; no input was truncated")
        response = self.client.post("/predict", json=REQUEST, headers=HEADERS)
        self.assertEqual(response.status_code, 422)
        self.assertIn("1024", response.text)
        for error in (RuntimeError("secret /private/model/path"), ValueError("secret nonfinite logits")):
            self.model.error = error
            with self.assertLogs("kev.api", level="ERROR") as logged:
                response = self.client.post("/predict", json=REQUEST, headers=HEADERS)
            self.assertEqual(response.status_code, 500)
            self.assertNotIn("secret", response.text + str(logged.output))
        self.model.error = None
        self.assertEqual(self.client.post("/predict", json=REQUEST, headers=HEADERS).status_code, 200)

    def test_shutdown_and_startup_failure(self):
        from fastapi.testclient import TestClient  # noqa: PLC0415
        loader = Mock(side_effect=RuntimeError("unavailable export"))
        app = create_app("/synthetic/missing", TOKEN, model_loader=loader)
        with self.assertRaisesRegex(RuntimeError, "unavailable export"), TestClient(app):
            self.fail("A failed startup must not serve requests")
        with self.assertRaises(ValueError):
            create_app("/synthetic/final", "short", model_loader=self.loader)
        with TestClient(create_app("/synthetic/final", TOKEN, model_loader=self.loader)) as client:
            model_app = client.app
            self.assertIsNotNone(model_app.state.model)
        self.assertIsNone(model_app.state.model)
        self.assertIsNone(model_app.state.batcher)

    def test_batch_configuration_must_be_bounded(self):
        for options in ({"pair_batch_size": True}, {"pair_batch_size": 1.5},
                        {"max_batch_tokens": 0}, {"max_batch_tokens": True}, {"max_batch_tokens": 1.5},
                        {"pad_to_multiple_of": 0}, {"pad_to_multiple_of": True}, {"pad_to_multiple_of": 1.5},
                        {"queue_capacity": 0}, {"queue_capacity": 1.5}, {"queue_capacity": True},
                        {"max_batch_requests": 0}, {"max_batch_requests": float("inf")},
                        {"batch_wait_ms": -1}, {"batch_wait_ms": float("nan")},
                        {"queue_timeout_ms": 0}, {"queue_timeout_ms": float("inf")},
                        {"queue_timeout_ms": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_app("/synthetic/final", TOKEN, model_loader=self.loader, **options)
