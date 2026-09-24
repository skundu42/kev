import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from kev.core import load_config, write_json
from kev.hub import (
    HUB_FILES, MODEL_FIELDS, PAYLOAD_FILES, SPLITS, build_bundle, pull, push,
    validate_dataset, validate_training_data, verify_bundle,
)


class HubTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "prepared"
        self.data.mkdir()
        self.config = {
            "model_name_or_path": "synthetic/model", "model_revision": "a" * 40,
            "max_length": 1024, "max_candidates": 16,
        }
        self.manifest = {
            "status": "complete", "config": self.config,
            "counts": dict.fromkeys(SPLITS, 1),
            "sources": [{"dataset": "synthetic/source", "revision": "b" * 40}],
        }
        write_json(self.data / "manifest.json", self.manifest)
        example = {
            "state": "A synthetic example.", "instructions": "Is this synthetic?",
            "candidates": ["No", "Yes"], "target": [0, 1], "kind": "noul",
            "input_ids": [[1, 2], [1, 3]], "attention_mask": [[1, 1], [1, 1]],
            "labels": [0, 1], "source": "synthetic/source", "task": "fixture",
        }
        for split in SPLITS:
            (self.data / f"{split}.jsonl").write_text(json.dumps(example) + "\n")
        self.repo = "synthetic/kev-data"
        self.revision = "c" * 40
        self.destination = self.root / "downloaded"

    def copy_snapshot(self, **kwargs):
        self.assertEqual(kwargs, {
            "repo_id": self.repo, "repo_type": "dataset", "revision": self.revision,
            "local_dir": str(self.destination) + ".download",
            "allow_patterns": list(HUB_FILES),
        })
        for name in kwargs["allow_patterns"]:
            shutil.copyfile(self.data / name, Path(kwargs["local_dir"]) / name)

    def test_completed_splits_and_compatible_tokenization_are_required(self):
        self.assertEqual(validate_dataset(self.data, self.config), self.manifest)
        # CPU preparation and GPU training may use different batch settings.
        validate_dataset(self.data, {**self.config, "per_device_train_batch_size": 2})
        for field in MODEL_FIELDS:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "differ"):
                validate_dataset(self.data, {**self.config, field: "different"})
        for status in ("preparing", "failed", None):
            write_json(self.data / "manifest.json", {**self.manifest, "status": status})
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "incomplete"):
                validate_dataset(self.data)
        for split in SPLITS:
            for count in (0, -1, True, "1", None):
                manifest = {**self.manifest, "counts": {**self.manifest["counts"], split: count}}
                write_json(self.data / "manifest.json", manifest)
                with self.subTest(split=split, count=count), self.assertRaises(ValueError):
                    validate_dataset(self.data)
        write_json(self.data / "manifest.json", self.manifest)
        for split in SPLITS:
            path = self.data / f"{split}.jsonl"
            contents = path.read_bytes()
            for action in (path.unlink, lambda: path.write_bytes(b"")):
                action()
                with self.subTest(split=split), self.assertRaises(ValueError):
                    validate_dataset(self.data)
                path.write_bytes(contents)

    def test_bundle_checks_counts_and_preserves_source_revisions(self):
        bundle = build_bundle(self.data)
        self.assertEqual(set(bundle["files"]), set(PAYLOAD_FILES))
        self.assertEqual(verify_bundle(self.data), self.manifest)
        card = (self.data / "README.md").read_text()
        self.assertIn("https://huggingface.co/datasets/synthetic/source", card)
        self.assertIn("b" * 40, card)
        with (self.data / "train.jsonl").open("a") as stream:
            stream.write('{}\n')
        with self.assertRaisesRegex(ValueError, "row count"):
            build_bundle(self.data)
        with self.assertRaisesRegex(ValueError, "Checksum"):
            verify_bundle(self.data)

    def test_training_checks_bundled_data_and_accepts_original_preparations(self):
        self.assertEqual(validate_training_data(self.data, self.config), self.manifest)
        build_bundle(self.data)
        self.assertEqual(validate_training_data(self.data, self.config), self.manifest)
        with self.assertRaisesRegex(ValueError, "differ"):
            validate_training_data(self.data, {**self.config, "max_length": 512})
        # Keep file size/row count identical: a metadata-only check misses this.
        path = self.data / "train.jsonl"
        path.write_text(path.read_text().replace("synthetic example", "different example"))
        with self.assertRaisesRegex(ValueError, "Checksum"):
            validate_training_data(self.data, self.config)

    def test_push_is_private_and_uploads_only_the_bundle(self):
        (self.data / "secret.txt").write_text("must not upload")
        (self.data / ".holdout-fingerprints.sqlite").write_text("scratch")
        original_receipt = {"repo_id": "original/data", "revision": "e" * 40}
        write_json(self.data / "hub.json", original_receipt)
        api = Mock()
        api.repo_info.return_value.private = True
        api.upload_folder.return_value.oid = self.revision
        sdk = types.SimpleNamespace(HfApi=Mock(return_value=api))
        with patch.dict(sys.modules, huggingface_hub=sdk), contextlib.redirect_stdout(io.StringIO()):
            result = push(self.data, self.repo)
        sdk.HfApi.assert_called_once_with()  # Let the SDK use the caller's saved auth/token.
        api.create_repo.assert_called_once_with(
            repo_id=self.repo, repo_type="dataset", private=True, exist_ok=True,
        )
        api.repo_info.assert_called_once_with(repo_id=self.repo, repo_type="dataset")
        uploaded = api.upload_folder.call_args.kwargs
        self.assertEqual(uploaded["repo_id"], self.repo)
        self.assertEqual(uploaded["repo_type"], "dataset")
        self.assertEqual(uploaded["folder_path"], str(self.data))
        self.assertEqual(uploaded["allow_patterns"], list(HUB_FILES))
        self.assertEqual(result, {"repo_id": self.repo, "revision": self.revision})
        self.assertEqual(load_config(self.data / "published.json"), result)
        self.assertEqual(load_config(self.data / "hub.json"), original_receipt)
        api.reset_mock()
        api.repo_info.return_value.private = False
        with patch.dict(sys.modules, huggingface_hub=sdk), self.assertRaisesRegex(ValueError, "private"):
            push(self.data, self.repo)
        api.upload_folder.assert_not_called()

    def test_pull_pins_revision_checks_payload_and_is_idempotent(self):
        build_bundle(self.data)
        snapshot = Mock(side_effect=self.copy_snapshot)
        sdk = types.SimpleNamespace(snapshot_download=snapshot)
        with patch.dict(sys.modules, huggingface_hub=sdk):
            for revision in ("main", "v1", "c" * 39, "g" * 40):
                with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "40-character"):
                    pull(self.destination, self.repo, revision)
            self.assertFalse(self.destination.exists())
            self.assertEqual(pull(self.destination, self.repo, self.revision), self.destination)
            self.assertEqual(verify_bundle(self.destination), self.manifest)
            self.assertFalse(self.destination.with_name("downloaded.download").exists())
            self.assertEqual(pull(self.destination, self.repo, self.revision), self.destination)
            snapshot.assert_called_once()
            with self.assertRaises(FileExistsError):
                pull(self.destination, self.repo, "d" * 40)
            (self.destination / "test.jsonl").write_text('{"tampered": true}\n')
            with self.assertRaisesRegex(ValueError, "Checksum"):
                pull(self.destination, self.repo, self.revision)

    def test_interrupted_transfer_resumes_and_never_exposes_partial_data(self):
        build_bundle(self.data)
        staging = self.destination.with_name("downloaded.download")

        def interrupted(**kwargs):
            shutil.copyfile(self.data / "train.jsonl", Path(kwargs["local_dir"]) / "train.jsonl")
            raise ConnectionError("synthetic interrupted transfer")

        snapshot = Mock(side_effect=interrupted)
        with patch.dict(sys.modules, huggingface_hub=types.SimpleNamespace(snapshot_download=snapshot)):
            with self.assertRaises(ConnectionError):
                pull(self.destination, self.repo, self.revision)
            self.assertFalse(self.destination.exists())
            self.assertTrue((staging / "train.jsonl").is_file())
            self.assertEqual(load_config(staging / ".kev-download.json"),
                             {"repo_id": self.repo, "revision": self.revision})
            snapshot.side_effect = self.copy_snapshot
            pull(self.destination, self.repo, self.revision)
        self.assertEqual(snapshot.call_count, 2)
        self.assertEqual(verify_bundle(self.destination), self.manifest)

    def test_tampered_download_is_rejected_before_promotion(self):
        build_bundle(self.data)

        def tampered(**kwargs):
            self.copy_snapshot(**kwargs)
            (Path(kwargs["local_dir"]) / "validation.jsonl").write_text('{}\n')

        with patch.dict(sys.modules, huggingface_hub=types.SimpleNamespace(snapshot_download=tampered)):
            with self.assertRaisesRegex(ValueError, "Checksum"):
                pull(self.destination, self.repo, self.revision)
        self.assertFalse(self.destination.exists())
        self.assertTrue(self.destination.with_name("downloaded.download").exists())

    def test_existing_or_unrelated_staging_data_is_preserved(self):
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            pull(self.destination, self.repo, self.revision)
        self.destination.rmdir()
        staging = self.destination.with_name("downloaded.download")
        staging.mkdir()
        original = staging / "keep.txt"
        original.write_text("preserve me")
        with self.assertRaises(FileExistsError):
            pull(self.destination, self.repo, self.revision)
        for identity in ({"repo_id": "other/data", "revision": self.revision},
                         {"repo_id": self.repo, "revision": "d" * 40}):
            write_json(staging / ".kev-download.json", identity)
            with self.assertRaises(FileExistsError):
                pull(self.destination, self.repo, self.revision)
        self.assertEqual(original.read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
