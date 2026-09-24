"""Prepare pinned datasets on the GPU pod; importing this module is offline-safe."""

import argparse
from collections import Counter, defaultdict
import heapq
import itertools
import json
from pathlib import Path
import sqlite3

from .core import load_config, require_cuda, tokenize_row, validate_row, write_json
from .data import (SPLIT_PRIORITY, SkipRow, adapt_row, content_key, digest,
                   split_for)


class HashSample:
    """Bounded, order-independent sample of the lowest seeded content hashes."""

    def __init__(self, limit, seed):
        self.limit, self.seed = limit, seed
        self.heap, self.ids = [], set()
        self.seen = 0

    def add(self, row):
        self.seen += 1
        if row["id"] in self.ids:
            return
        priority = int(digest(f"{self.seed}:{row['id']}"), 16)
        item = (-priority, row["id"], row)
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, item)
            self.ids.add(row["id"])
        elif item > self.heap[0]:
            removed = heapq.heapreplace(self.heap, item)
            self.ids.remove(removed[1])
            self.ids.add(row["id"])

    def rows(self):
        return [item[2] for item in sorted(self.heap, reverse=True)]


def positive_int(config, key):
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def prepare(config, output_dir):
    require_cuda()  # This precedes every Hugging Face import or network operation.
    from datasets import get_dataset_config_names, load_dataset
    from transformers import AutoTokenizer

    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be an empty directory: {output}; choose a new output directory.")
    max_train = positive_int(config, "max_train_per_source")
    max_task = positive_int(config, "max_train_per_task")
    max_eval = positive_int(config, "max_eval_per_source")
    max_length = positive_int(config, "max_length")
    max_candidates = positive_int(config, "max_candidates")
    if max_length > 7999 or max_candidates < 2:
        raise ValueError("max_length must be at most 7999 and max_candidates at least 2")
    scan_limit = config.get("scan_limit_per_split")
    if scan_limit is not None and (isinstance(scan_limit, bool) or not isinstance(scan_limit, int) or scan_limit < 1):
        raise ValueError("scan_limit_per_split must be null or a positive integer (smoke runs only)")
    seed = config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    output.mkdir(parents=True, exist_ok=True)
    data_config = load_config(Path(__file__).resolve().parents[1] / "configs" / "data.json")
    specs = data_config["sources"]
    resolved = []
    for original in specs:
        spec = dict(original)
        spec["upstream_url"] = f"https://huggingface.co/datasets/{spec['dataset']}"
        spec["license_terms_reference"] = spec["upstream_url"]
        if spec["configs"] == "discover":
            available = sorted(get_dataset_config_names(spec["dataset"], revision=spec["revision"]))
            chosen = config.get("bigbench_configs")
            if chosen is not None and (not isinstance(chosen, list) or not chosen or set(chosen) - set(available)):
                raise ValueError(f"Unknown or empty bigbench_configs; available configs: {available}")
            spec["configs"] = sorted(chosen) if chosen is not None else available
            if not spec["configs"]:
                raise RuntimeError(f"No configurations discovered for {spec['dataset']} at its pinned revision")
        resolved.append(spec)
    manifest = {
        "status": "preparing", "config": config, "sources": resolved,
        "seed": seed, "scan_limit_per_split": scan_limit,
        "smoke_prefix_scan": scan_limit is not None,
        "partition_policy": "Native test retained; native validation: 50/50 validation/calibration with test, otherwise 50/25/25 validation/calibration/test.",
        "dedup_policy": "Normalized state across tasks/sources; related content groups remain in one partition. Holdouts win, with test > calibration > validation > train.",
        "sampling_policy": "Lowest seeded row hashes per source/output partition, then aggregate task caps, then tokenization. Caps are maxima and are not backfilled after later filtering.",
        "limitations": "Smoke scans use only a prefix per native split; no claim of full-corpus leakage checking. Exact/group dedup does not detect paraphrases.",
        "max_eval_per_source_applies_to": "each output partition",
        "instruct_allowlist": data_config["instruct_allowlist"],
    }
    write_json(output / "manifest.json", manifest)
    stats = {spec["dataset"]: Counter() for spec in resolved}
    scan_stats = []
    samples = {}
    for spec in resolved:
        for split in SPLIT_PRIORITY:
            samples[spec["dataset"], split] = HashSample(max_train if split == "train" else max_eval, seed)
    database_path = output / ".holdout-fingerprints.sqlite"
    db = sqlite3.connect(database_path)
    db.execute("CREATE TABLE groups (fingerprint TEXT PRIMARY KEY, priority INTEGER NOT NULL)")
    db.execute("CREATE TABLE content (fingerprint TEXT PRIMARY KEY)")
    try:
        # All scanned holdouts reserve their fingerprints before any training rows
        # are sampled, including holdouts subsequently lost to caps/length limits.
        for phase in ("holdout", "train"):
            for spec in resolved:
                source = spec["dataset"]
                has_test = "test" in spec["splits"].values()
                for subset in spec["configs"]:
                    for original_split, role in spec["splits"].items():
                        if (role == "train") != (phase == "train"):
                            continue
                        print(f"Scanning {source} / {subset} / {original_split}", flush=True)
                        scanned = 0
                        try:
                            stream = load_dataset(source, subset, split=original_split,
                                                  revision=spec["revision"], streaming=True)
                            rows = iter(stream)
                            if scan_limit is not None:
                                rows = itertools.islice(rows, scan_limit)
                            for raw in rows:
                                scanned += 1
                                stats[source]["raw_scanned"] += 1
                                try:
                                    row = adapt_row(spec, subset, original_split, raw)
                                    if len(row["candidates"]) > max_candidates:
                                        raise SkipRow("over_candidate_limit")
                                    validate_row(row, max_candidates)
                                except SkipRow as error:
                                    stats[source][f"dropped:{error}"] += 1
                                    continue
                                split = split_for(role, row["group_id"], has_test, seed)
                                fingerprint = content_key(row)
                                if phase == "holdout":
                                    db.execute("INSERT INTO groups VALUES (?, ?) ON CONFLICT(fingerprint) DO UPDATE SET priority=max(priority, excluded.priority)",
                                               (row["group_id"], SPLIT_PRIORITY[split]))
                                    db.execute("INSERT OR IGNORE INTO content VALUES (?)", (fingerprint,))
                                elif (db.execute("SELECT 1 FROM groups WHERE fingerprint=?", (row["group_id"],)).fetchone()
                                      or db.execute("SELECT 1 FROM content WHERE fingerprint=?", (fingerprint,)).fetchone()):
                                    stats[source]["dropped:holdout_overlap"] += 1
                                    continue
                                stats[source][f"eligible:{split}"] += 1
                                samples[source, split].add(row)
                            db.commit()
                        except Exception as error:
                            raise RuntimeError(f"Preparation failed for {source}, config={subset}, split={original_split}, revision={spec['revision']}: {error}") from error
                        scan_stats.append({"source": source, "config": subset, "split": original_split,
                                           "scanned": scanned,
                                           "scan_limit_reached": scan_limit is not None and scanned == scan_limit})
        tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], revision=config["model_revision"], trust_remote_code=False)
        seen_content = set()
        counts = Counter()
        written_train_sources = set()
        aggregate = {spec["dataset"]: spec.get("aggregate", False) for spec in resolved}
        retained_tasks = defaultdict(Counter)
        # Emit highest-priority partitions first so cross-source exact duplicates
        # cannot end up in both a training and an evaluation file.
        for split in sorted(SPLIT_PRIORITY, key=SPLIT_PRIORITY.get, reverse=True):
            destination = output / f"{split}.jsonl.tmp"
            with destination.open("w") as handle:
                for spec in resolved:
                    source = spec["dataset"]
                    pool = samples[source, split]
                    stats[source][f"sampled:{split}"] = len(pool.heap)
                    stats[source][f"dropped:sampling_or_repetition:{split}"] = pool.seen - len(pool.heap)
                    task_counts = Counter()
                    for row in pool.rows():
                        owner = db.execute("SELECT priority FROM groups WHERE fingerprint=?", (row["group_id"],)).fetchone()
                        if owner and owner[0] > SPLIT_PRIORITY[split]:
                            stats[source]["dropped:group_partition_conflict"] += 1
                            continue
                        fingerprint = content_key(row)
                        if fingerprint in seen_content:
                            stats[source]["dropped:duplicate_content"] += 1
                            continue
                        if split == "train" and aggregate[source] and task_counts[row["task"]] >= max_task:
                            stats[source]["dropped:task_cap"] += 1
                            continue
                        try:
                            encoded = tokenize_row(row, tokenizer, max_length, max_candidates)
                        except ValueError as error:
                            if str(error).startswith("overlength:"):
                                stats[source][f"dropped:overlength:{split}"] += 1
                                continue
                            raise
                        seen_content.add(fingerprint)
                        task_counts[row["task"]] += 1
                        counts[split] += 1
                        stats[source][f"written:{split}"] += 1
                        retained_tasks[f"{source}:{split}"][row["task"]] += 1
                        if split == "train":
                            written_train_sources.add(source)
                        handle.write(json.dumps({**row, **encoded}, ensure_ascii=False, allow_nan=False) + "\n")
        missing = sorted({spec["dataset"] for spec in resolved} - written_train_sources)
        if missing:
            raise RuntimeError("No usable training rows for: " + ", ".join(missing)
                               + ". Inspect manifest rejection counts; increase scan_limit_per_split for smoke, candidate/length limits, or source caps.")
        if any(counts[split] == 0 for split in SPLIT_PRIORITY):
            raise RuntimeError(f"Empty output partition: {dict(counts)}. Increase smoke scan/source limits.")
        for split in SPLIT_PRIORITY:
            (output / f"{split}.jsonl.tmp").replace(output / f"{split}.jsonl")
        manifest.update(status="complete", counts=dict(counts), tokenizer={"name": config["model_name_or_path"], "revision": config["model_revision"]})
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest.update(source_counts={key: dict(value) for key, value in stats.items()}, scans=scan_stats)
        if "retained_tasks" in locals():
            manifest["retained_tasks"] = {key: dict(value) for key, value in retained_tasks.items()}
        write_json(output / "manifest.json", manifest)
        db.close()
        database_path.unlink(missing_ok=True)
    print(json.dumps({"output_dir": str(output), "counts": dict(counts)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    prepare(load_config(args.config), args.output_dir)


if __name__ == "__main__":
    main()
