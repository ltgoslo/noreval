"""Build noreval_metadata.json from the `metadata` blocks of the task configs."""

import argparse
import json
import sys
from pathlib import Path

from lm_eval.tasks import TaskManager


ROOT = Path(__file__).resolve().parent.parent
PERCENT_METRICS = {"bleu", "chrf", "rougeL_max"}


def build(tasks_dir):
    index = TaskManager(include_path=str(tasks_dir), include_defaults=False).task_index
    variants = sorted(n for n, e in index.items() if e.kind.name == "TASK")
    tags = {n: e.tags for n, e in index.items() if e.kind.name == "TAG"}
    tags.update({n: set(e.cfg["task"]) for n, e in index.items() if e.kind.name == "GROUP"})

    tasks = {}
    for variant in variants:
        own_tags = [t for t, members in tags.items() if variant in members]
        task = max(own_tags, key=lambda t: len(tags[t])) if own_tags else variant
        metadata = index[variant].cfg["metadata"]
        language = metadata["language"]
        fields = {
            "category": metadata["category"],
            "language": language if isinstance(language, list) else [language],
            "primary_metric": metadata["primary_metric"],
            "random_baseline": metadata["random_baseline"],
            "max_score": 100.0 if metadata["primary_metric"] in PERCENT_METRICS else 1.0,
            "version": metadata["version"],
        }
        entry = tasks.setdefault(task, {**fields, "variants": []})
        for key, value in fields.items():
            if entry[key] != value:
                sys.exit(f"{variant}: {key} is {value!r}, but {entry[key]!r} in other variants of {task}")
        entry["variants"].append(variant)
    return {task: tasks[task] for task in sorted(tasks)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tasks", default=ROOT / "tasks", type=Path, help="Directory with the task configs")
    parser.add_argument("--output", default=ROOT / "noreval_metadata.json", type=Path, help="Path of the JSON file")
    args = parser.parse_args()

    tasks = build(args.tasks)
    args.output.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(tasks)} tasks, {sum(len(t['variants']) for t in tasks.values())} variants -> {args.output}")
