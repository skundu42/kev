"""Pure-Python adapters. Source labels keep their original meaning."""

import hashlib
import json
import math
import re
import unicodedata


NLI = ["entailment", "neutral", "contradiction"]
RATINGS = ["1 star", "2 star", "3 stars", "4 stars", "5 stars"]
SENTIMENT = ["negative", "neutral", "positive"]
INSTRUCT_TASKS = {
    "yelp_review_full/yelp_review_full": (RATINGS, "score", "How many stars did the reviewer give?"),
    "tweet_eval/sentiment": (SENTIMENT, "score", "Rate the sentiment of the text."),
}
SPLIT_PRIORITY = {"train": 0, "validation": 1, "calibration": 2, "test": 3}


class SkipRow(ValueError):
    """A known, counted reason a source example is unsuitable."""


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_key(row):
    # Deliberately ignores task prompts/labels so ANLI and its DocNLI recast match.
    return digest(normalize(row["state"]))


def split_for(role, group_id, has_test, seed=42):
    if role in ("train", "test"):
        return role
    if role != "validation":
        raise ValueError(f"Unsupported source split role: {role}")
    fraction = int(digest(f"{seed}:{group_id}")[:16], 16) / 2**64
    if fraction < 0.5:
        return "validation"
    return "calibration" if has_test or fraction < 0.75 else "test"


def excluded_task(task):
    """Prefer native sources over known duplicate aggregate task families."""
    task = task.casefold().replace("_", "-")
    names = ("mnli", "multi-nli", "anli", "boolq", "hellaswag", "defeasible-nli",
             "fol-nli", "doc-nli", "bigbench", "zero-shot-label-nli", "yelp-review-full")
    return (any(name in task.split("/") for name in names)
            or task == "tweet-eval/sentiment"
            or task.startswith(("boolq-natural-perturbations", "robust-nli", "breaking-nli")))


def text(raw, key):
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise SkipRow(f"blank_or_invalid_{key}")
    return value.strip()


def one_hot(label, candidates):
    if isinstance(label, str) and label in candidates:
        label = candidates.index(label)
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(candidates):
        raise SkipRow("invalid_label")
    return [float(index == label) for index in range(len(candidates))]


def adapt_row(spec, config, original_split, raw):
    """Convert one native example; missing schema fields fail rather than disappear."""
    source = spec["dataset"]
    adapter = spec["adapter"]
    task = f"{source}/{config}"
    kind = "choice"
    if adapter in ("nli", "zero_shot", "doc_nli"):
        premise, hypothesis = text(raw, "premise"), text(raw, "hypothesis")
        state = f"Premise: {premise}\nHypothesis: {hypothesis}"
        group = premise
        instructions = "What is the relationship between the premise and hypothesis?"
        candidates = NLI
        key = "labels" if adapter == "zero_shot" else "label"
        if adapter == "zero_shot":
            task = text(raw, "task")
            if excluded_task(task):
                raise SkipRow("overlapping_task_family")
        if adapter == "doc_nli":
            candidates = ["entailment", "not_entailment"]
            instructions = "Does the premise entail the hypothesis?"
        target = one_hot(raw[key], candidates)
    elif adapter == "defeasible":
        hypothesis, update = text(raw, "Hypothesis"), text(raw, "Update")
        premise = "" if config == "social" else text(raw, "Premise")
        group = f"Premise: {premise}\nHypothesis: {hypothesis}"
        state = (f"Premise: {premise}\n" if premise else "") + f"Hypothesis: {hypothesis}\nUpdate: {update}"
        instructions = "Does the update make the hypothesis more or less likely?"
        candidates = ["strengthener", "weakener"]
        target = one_hot(raw["UpdateType"], candidates)
    elif adapter == "hellaswag":
        state = text(raw, "ctx")
        candidates = raw["endings"]
        instructions = "Choose the most plausible continuation of the passage."
        group = str(raw.get("source_id") or state)
        label = raw["label"]
        if not isinstance(label, str) or label not in {"0", "1", "2", "3"} or len(candidates) != 4:
            raise SkipRow("unlabeled_or_invalid_hellaswag")
        target = one_hot(int(label), candidates)
    elif adapter == "boolq":
        passage, question = text(raw, "passage"), text(raw, "question")
        state = f"Passage: {passage}\nQuestion: {question}"
        group = passage
        instructions = "Answer the question using the passage."
        candidates, kind = ["No", "Yes"], "noul"
        if not isinstance(raw["answer"], bool):
            raise SkipRow("invalid_boolq_answer")
        target = one_hot(int(raw["answer"]), candidates)
    elif adapter == "instruct":
        task = text(raw, "task")
        if task not in INSTRUCT_TASKS:
            raise SkipRow("outside_instruct_allowlist")
        candidates, kind, instructions = INSTRUCT_TASKS[task]
        header, separator, state = text(raw, "inputs").partition("\n")
        prefix = "With no explanation, label the following with either "
        if not separator or not header.startswith(prefix) or not header.endswith("."):
            raise SkipRow("unsupported_instruct_format")
        options = re.findall(r'"([^"\n]+)"', header[len(prefix):])
        answer = text(raw, "targets")
        # The frozen renderer adds exactly one period, even after punctuation.
        if not answer.endswith("."):
            raise SkipRow("unsupported_instruct_target")
        answer = answer[:-1]
        if (len(options) < 2 or len(set(options)) != len(options)
                or not set(options).issubset(candidates) or answer not in options):
            raise SkipRow("invalid_instruct_options")
        state = state.strip()
        if not state:
            raise SkipRow("blank_instruct_state")
        group = state
        target = one_hot(answer, candidates)
    elif adapter == "bigbench":
        state = text(raw, "inputs")
        group = state
        instructions = "Choose the best answer to the question."
        candidates, scores = raw["multiple_choice_targets"], raw["multiple_choice_scores"]
        if not candidates:
            raise SkipRow("open_ended_bigbench")
        if len(candidates) != len(scores) or len(scores) < 2:
            raise SkipRow("invalid_bigbench_scores")
        if any(isinstance(score, bool) or not isinstance(score, (int, float))
               or not math.isfinite(score) or score < 0 for score in scores):
            raise SkipRow("invalid_bigbench_scores")
        if sum(scores) <= 0 or len(set(scores)) == 1:
            raise SkipRow("uninformative_bigbench_scores")
        target = [float(score / sum(scores)) for score in scores]
    else:
        raise ValueError(f"Unknown adapter {adapter!r}")
    if not isinstance(candidates, list) or any(not isinstance(c, str) or not c.strip() for c in candidates):
        raise SkipRow("invalid_candidates")
    candidates = [c.strip() for c in candidates]
    if len(set(candidates)) != len(candidates):
        raise SkipRow("duplicate_candidates")
    row = dict(state=state, instructions=instructions, candidates=candidates,
               target=target, kind=kind, source=source, task=task,
               original_split=original_split, group_id=digest(normalize(group)))
    row["id"] = digest(json.dumps(row, sort_keys=True, ensure_ascii=False))
    return row
