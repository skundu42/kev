"""Pure validation tests plus opt-in ASGI tests with an injected, offline fake model."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock

from kev.core import InputTooLongError, answer, parse_questions
from kev.serve import (MAX_BODY_BYTES, MAX_QUESTIONS, authorized, create_app,
                       decode_request, valid_api_key)

REQUEST = json.loads((Path(__file__).resolve().parents[1] / "examples/request.json").read_text())
TOKEN = "test-only-token-" + "x" * 32
HEADERS = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}


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
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.error = None

    def predict(self, state, questions):
        self.calls += 1
        if self.block:
            self.started.set()
            if not self.release.wait(5):
                raise RuntimeError("Test timed out")
        if self.error:
            raise self.error
        rows, metadata = parse_questions(state, questions)
        return {"answers": {qid: answer(row["kind"], row["candidates"], keys,
                                         [1 / len(keys)] * len(keys))
                            for row, (qid, keys) in zip(rows, metadata)}}


@unittest.skipUnless(os.environ.get("KEV_TEST_API") == "1", "Set KEV_TEST_API=1 with API test dependencies installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.model = FakeModel()
        self.loader = Mock(return_value=self.model)
        self.app = create_app("/synthetic/final", TOKEN, model_loader=self.loader)
        self.client = self.enterContext(TestClient(self.app))

    def test_startup_loads_once_and_serves_all_types(self):
        self.loader.assert_called_once_with("/synthetic/final", pair_batch_size=8)
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

    def test_busy_predictions_do_not_block_health_or_overlap(self):
        self.model.block = True
        with ThreadPoolExecutor(max_workers=1) as pool:
            active = pool.submit(self.client.post, "/predict", json=REQUEST, headers=HEADERS)
            try:
                self.assertTrue(self.model.started.wait(2))
                self.assertEqual(self.client.get("/healthz").status_code, 200)
                busy = self.client.post("/predict", json=REQUEST, headers=HEADERS)
                self.assertEqual(busy.status_code, 503)
                self.assertEqual(busy.headers["retry-after"], "1")
                self.assertEqual(self.model.calls, 1)
            finally:
                self.model.release.set()
            self.assertEqual(active.result(timeout=2).status_code, 200)
        self.assertEqual(self.client.post("/predict", json=REQUEST, headers=HEADERS).status_code, 200)

    def test_errors_release_lock_and_do_not_expose_internal_details(self):
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
        from fastapi.testclient import TestClient
        loader = Mock(side_effect=RuntimeError("unavailable export"))
        app = create_app("/synthetic/missing", TOKEN, model_loader=loader)
        with self.assertRaisesRegex(RuntimeError, "unavailable export"):
            with TestClient(app):
                self.fail("A failed startup must not serve requests")
        with self.assertRaises(ValueError):
            create_app("/synthetic/final", "short", model_loader=self.loader)
        with TestClient(create_app("/synthetic/final", TOKEN, model_loader=self.loader)) as client:
            model_app = client.app
            self.assertIsNotNone(model_app.state.model)
        self.assertIsNone(model_app.state.model)
