import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kev.benchmark import benchmark_model, calibration_metrics, configurations, load_workload, main, probability_delta


def row(index=0):
    return {"state": str(index), "instructions": "Choose", "candidates": ["No", "Yes"],
                "target": [0.0, 1.0], "kind": "choice"}


class BenchmarkTests(unittest.TestCase):
    def test_fixed_baseline_and_opt_in_backend_sweep(self):
        configs = configurations()
        self.assertEqual(len(configs), 13)
        self.assertEqual(configs[0]["pair_batch_size"], 8)
        self.assertIsNone(configs[0]["max_batch_tokens"])
        self.assertFalse(configs[0]["group_by_length"])
        self.assertEqual(configs[0]["pad_to_multiple_of"], 1)
        self.assertEqual(configs[0]["weight_dtype"], "float32")
        self.assertFalse(any(c["compile_model"] for c in configs))
        configs = configurations(compile_model=True, flash_attention=True)
        extras = [c for c in configs if c["compile_model"] or c["attn_implementation"] != "sdpa"]
        self.assertEqual(len(extras), 8)
        self.assertTrue(all(c["pair_batch_size"] == 32 for c in extras))
        self.assertTrue(all(c["weight_dtype"] == "bfloat16" for c in extras
                            if c["attn_implementation"] == "flash_attention_2"))
        self.assertEqual(len({c["name"] for c in configs}), len(configs))
        for values in ({"batch_sizes": []}, {"request_batch_sizes": [0]},
                       {"max_batch_tokens": 0}, {"pad_to_multiple_of": -1}):
            with self.assertRaises(ValueError):
                configurations(**values)

    def test_labelled_grouping_limit_and_request_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.jsonl"
            path.write_text("\n".join(json.dumps(row(i)) for i in range(12)))
            workload = load_workload(path, max_requests=2)
            self.assertEqual([len(request) for request in workload], [5, 5])
            self.assertEqual(workload[-1][-1]["state"], "9")
            self.assertEqual([len(r) for r in load_workload(path, max_requests=0)], [5, 5, 2])
            request = {"state": "example", "questions": {"check": {"type": "noul", "instructions": "Check"}}}
            path.write_text(json.dumps(request) + "\n" + json.dumps(request))
            requests = load_workload(path, request_input=True)
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0][0]["candidates"], ["No", "Yes"])
            self.assertIsNone(calibration_metrics(requests[0], [[.4, .6]]))
            with self.assertRaisesRegex(ValueError, "target"):
                load_workload(path)
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_workload(path)

    def test_probability_deltas_and_saved_temperature_metrics(self):
        result = probability_delta([[.1, .9], [.7, .3]], [[.2, .8], [.4, .6]])
        self.assertAlmostEqual(result["max_abs"], .3)
        self.assertAlmostEqual(result["mean_abs"], .2)
        self.assertAlmostEqual(result["mean_total_variation"], .2)
        self.assertEqual(result["argmax_changes"], 1)
        self.assertEqual(result["argmax_change_rate"], .5)
        self.assertEqual(calibration_metrics([row()], [[.2, .8]])["count"], 1)
        with self.assertRaises(ValueError):
            probability_delta([[.1, .9]], [[.1, .2, .7]])
        with self.assertRaises(ValueError):
            probability_delta([[.1, .9]], [])

    def test_warms_all_shapes_then_reports_combined_batch_timing(self):
        cuda = SimpleNamespace(synchronize=Mock(), reset_peak_memory_stats=Mock(),
                               max_memory_allocated=Mock(return_value=2**30),
                               max_memory_reserved=Mock(return_value=2 * 2**30))
        model = SimpleNamespace(device="test", score_rows=Mock(
            side_effect=lambda rows: [[float(r["state"]), 1.0] for r in rows]))
        ticks = iter(range(16))
        timing, logits = benchmark_model(model, [[row(i)] for i in range(3)], 2, cuda,
                                        warmup_passes=1, repeats=2, clock=lambda: next(ticks))
        self.assertEqual([len(call.args[0]) for call in model.score_rows.call_args_list], [2, 1] * 4)
        self.assertEqual(cuda.synchronize.call_count, 16)
        self.assertEqual(cuda.reset_peak_memory_stats.call_count, 2)
        self.assertEqual(timing["first_pass_ms"], 2000)
        self.assertEqual(timing["additional_warmup_ms"], 2000)
        self.assertEqual(timing["measured_seconds"], 4)
        self.assertEqual(timing["requests_per_second"], 1.5)
        self.assertEqual(timing["candidate_pairs_per_second"], 3)
        self.assertEqual(timing["steady_peak_allocated_gib"], 1)
        self.assertEqual(timing["combined_batch_latency_ms"], {"median": 1000, "p95": 1000})
        self.assertEqual(logits, [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        model.score_rows = Mock(return_value=[])
        with self.assertRaisesRegex(ValueError, "wrong number"):
            benchmark_model(model, [[row()]], 1, cuda)

    def test_cli_preserves_backend_failures_and_never_overwrites_existing_report(self):
        cuda = SimpleNamespace(synchronize=Mock(), reset_peak_memory_stats=Mock(),
                               max_memory_allocated=Mock(return_value=0), empty_cache=Mock(),
                               max_memory_reserved=Mock(return_value=0), get_device_name=Mock(return_value="test"))
        runtime = SimpleNamespace(cuda=cuda, compiler=SimpleNamespace(reset=Mock()),
                                  __version__="test", version=SimpleNamespace(cuda="test"))

        def load_model(path, **options):
            if options["weight_dtype"] == "bfloat16":
                raise RuntimeError("BF16 is unavailable")
            return SimpleNamespace(device="test", temperature=2.0, weight_dtype="float32",
                                   score_rows=lambda rows: [[0.0, 1.0] for _ in rows])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "kev_config.json").write_text(json.dumps({"max_candidates": 16, "temperature": 2.0}))
            (root / "test.jsonl").write_text(json.dumps(row()))
            arguments = ["--model", str(root), "--data", str(root / "test.jsonl"),
                         "--output", str(root / "report.json"), "--batch-sizes", "8",
                         "--request-batch-sizes", "1", "--warmup-passes", "0", "--repeats", "1"]
            with patch.dict(sys.modules, torch=runtime), patch("kev.benchmark.require_cuda", return_value="test"), \
                 patch("kev.benchmark.version", return_value="test"), \
                 patch("kev.benchmark.DecisionModel.from_pretrained", side_effect=load_model):
                self.assertEqual(main(arguments), 1)
                report = json.loads((root / "report.json").read_text())
                self.assertEqual([r["status"] for r in report["results"]], ["ok", "ok", "error"])
                self.assertEqual(report["results"][-1]["error"]["message"], "BF16 is unavailable")
                self.assertEqual(report["results"][0]["calibration_delta"]["log_loss"], 0)
                self.assertEqual(report["results"][0]["resolved_weight_dtype"], "float32")
                with self.assertRaises(SystemExit):
                    main(arguments)
                arguments[arguments.index("--output") + 1] = str(root / "failed.json")
                with patch("kev.benchmark.DecisionModel.from_pretrained", side_effect=RuntimeError("bad reference")):
                    self.assertEqual(main(arguments), 1)
                failed = json.loads((root / "failed.json").read_text())
                self.assertEqual([r["status"] for r in failed["results"]], ["error", "skipped", "skipped"])
