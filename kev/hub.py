"""Publish and fetch prepared decisions without Torch, CUDA, or Halo."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from .core import load_config, write_json


SPLITS = ("train", "validation", "calibration", "test")
DATA_FILES = tuple(f"{split}.jsonl" for split in SPLITS)
PAYLOAD_FILES = ("manifest.json", *DATA_FILES)
HUB_FILES = (*PAYLOAD_FILES, "bundle.json", "README.md")
MODEL_FIELDS = ("model_name_or_path", "model_revision", "max_length", "max_candidates")


def validate_dataset(data_dir, config=None):
    """Accept existing preparations, including those made before CPU support."""
    root = Path(data_dir)
    manifest = load_config(root / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Prepared data is incomplete; finish preparation before publishing or training.")
    if manifest.get("format_version", 1) != 1:
        raise ValueError("Unsupported prepared-data format version")
    settings = manifest.get("config", {})
    if any(key not in settings for key in MODEL_FIELDS):
        raise ValueError("Prepared manifest is missing model/tokenizer settings")
    if config is not None and any(settings[key] != config.get(key) for key in MODEL_FIELDS):
        raise ValueError("Prepared data's tokenizer/model revision or length/candidate limits differ from this config.")
    for split in SPLITS:
        count = manifest.get("counts", {}).get(split)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Manifest must record a positive count for {split}")
        path = root / f"{split}.jsonl"
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing, empty, or symlinked prepared split: {path}")
    if (root / "manifest.json").is_symlink():
        raise ValueError("Prepared manifest must not be a symlink")
    return manifest


def file_info(path):
    checksum = hashlib.sha256()
    size, lines = 0, 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
            size += len(block)
            lines += block.count(b"\n")
    return {"sha256": checksum.hexdigest(), "bytes": size, "lines": lines}


def build_bundle(data_dir):
    root = Path(data_dir)
    manifest = validate_dataset(root)
    files = {name: file_info(root / name) for name in PAYLOAD_FILES}
    for split in SPLITS:
        if files[f"{split}.jsonl"]["lines"] != manifest["counts"][split]:
            raise ValueError(f"{split} row count differs from the manifest; refusing partial data")
    bundle = {"format_version": 1, "files": files}
    write_json(root / "bundle.json", bundle)
    card = ["---", "configs:", "- config_name: default", "  data_files:"]
    for split in SPLITS:
        card.extend([f"  - split: {split}", f"    path: {split}.jsonl"])
    card.extend(["---", "# Kev prepared decisions", "",
                 "Tokenized decision examples for the Kev Halo trainer. See `manifest.json` for",
                 "exact source revisions, exclusions, sampling, split policy and retained counts.",
                 "`bundle.json` records SHA-256 checksums and byte/row counts for transfer validation.",
                 "Upstream license terms remain applicable; this mixture does not assign a new license.", "",
                 "## Sources", ""])
    for source in manifest.get("sources", []):
        card.append(f"- [{source['dataset']}](https://huggingface.co/datasets/{source['dataset']}) "
                    f"at `{source['revision']}`")
    (root / "README.md").write_text("\n".join(card) + "\n")
    return bundle


def verify_bundle(data_dir):
    root = Path(data_dir)
    manifest = validate_dataset(root)
    bundle = load_config(root / "bundle.json")
    if bundle.get("format_version") != 1 or set(bundle.get("files", {})) != set(PAYLOAD_FILES):
        raise ValueError("Invalid prepared-data bundle; required file list differs")
    for name in PAYLOAD_FILES:
        actual = file_info(root / name)
        if actual != bundle["files"][name]:
            raise ValueError(f"Checksum or size mismatch for {name}; dataset is incomplete or modified")
        if name in DATA_FILES and actual["lines"] != manifest["counts"][name[:-6]]:
            raise ValueError(f"Row count mismatch for {name}")
    return manifest


def push(data_dir, repo_id):
    root = Path(data_dir)
    build_bundle(root)
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id=repo_id, repo_type="dataset").private:
        raise ValueError("Choose a private dataset repository; this command does not publish the mixture publicly.")
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(root),
        allow_patterns=list(HUB_FILES), commit_message="Publish prepared Kev decisions and provenance",
    )
    result = {"repo_id": repo_id, "revision": commit.oid}
    # Publishing must not change the download provenance of an active run.
    write_json(root / "published.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def pull(data_dir, repo_id, revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use the exact 40-character commit revision printed by push-data, not a branch or tag.")
    root = Path(data_dir)
    identity = {"repo_id": repo_id, "revision": revision}
    if root.exists():
        if (root / "hub.json").is_file() and load_config(root / "hub.json") == identity:
            verify_bundle(root)
            return root
        raise FileExistsError(f"Data already exists: {root}. Choose a new KEV_DATA_DIR; existing data is preserved.")
    # Retry interrupted transfers in the same sibling directory, never expose
    # partial files at the path that training consumes.
    staging = root.with_name(root.name + ".download")
    marker = staging / ".kev-download.json"
    if staging.exists():
        if not marker.is_file() or load_config(marker) != identity:
            raise FileExistsError(f"A different download uses {staging}; choose another KEV_DATA_DIR.")
    else:
        staging.mkdir(parents=True)
        write_json(marker, identity)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                      local_dir=str(staging), allow_patterns=list(HUB_FILES))
    verify_bundle(staging)
    write_json(staging / "hub.json", identity)
    staging.rename(root)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("push", "pull", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--data-dir", required=True)
        if name in {"push", "pull"}:
            command.add_argument("--repo-id", required=True)
        if name == "pull":
            command.add_argument("--revision", required=True)
        if name == "validate":
            command.add_argument("--config")
    args = parser.parse_args()
    if args.command == "push":
        push(args.data_dir, args.repo_id)
    elif args.command == "pull":
        print(pull(args.data_dir, args.repo_id, args.revision))
    else:
        config = load_config(args.config) if args.config else None
        manifest = validate_dataset(args.data_dir, config)
        print(json.dumps({"status": "complete", "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()
