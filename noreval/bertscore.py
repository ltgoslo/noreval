import os
import json
import numpy as np
import argparse
from typing import Dict
from evaluate import load


TASK2FIELDS = {
    "ask_gec_nb": {"source": "source", "target": "correction"},
    "tatoeba_nob_nno_nb": {"source": "sourceString", "target": "targetString"},
    "tatoeba_nob_eng_nb": {"source": "targetString", "target": "sourceString"},
    "tatoeba_nno_nob_nn": {"source": "targetString", "target": "sourceString"},
    "tatoeba_nno_eng_nn": {"source": "targetString", "target": "sourceString"},
    "tatoeba_eng_nno_nn": {"source": "sourceString", "target": "targetString"},
    "tatoeba_eng_nob_nb": {"source": "sourceString", "target": "targetString"},
    "schibsted_vg_nb": {"source": "article_text", "target": "front_title"},
}


def read_examples(
    fpath: str,
    task_name: str,
    prediction_key: str = "filtered_resps",
    task2fields: Dict = TASK2FIELDS,
):
    with open(fpath, "r", encoding="utf-8") as f:
        examples = json.load(f)
    sources, targets, predictions = [], [], []
    source_key, target_key = (
        task2fields[task_name]["source"],
        task2fields[task_name]["target"],
    )
    for i, example in enumerate(examples):
        source = example["doc"][source_key]
        sources.append(source)
        target = example["doc"][target_key]
        targets.append(target)
        prediction = example[prediction_key][0]
        if prediction_key == "resps":
            prediction = prediction[0]
        predictions.append(prediction.strip())
        if i == 0:
            print(
                f"Source key: {source_key}\n\nTarget key: {target_key}\n\nPrediction key: {prediction_key}\n\n"
            )
            print(
                f"Source: {source}\n\nTarget: {target}\n\nPrediction: {prediction.strip()}",
                flush=True,
            )
    return sources, targets, predictions


def save_results(fpath: str, obj: dict):
    with open(fpath, "w+", encoding="utf-8") as out:
        json.dump(obj, out, indent=3)


def evaluate(
    fpath: str,
    out_fpath: str,
    task_name: str,
    prediction_key: str,
    batch_size: int = 64,
    task2fields: Dict = TASK2FIELDS,
):
    tmp_name = fpath.replace(".jsonl", "").replace("/", "-")
    os.makedirs("tmp", exist_ok=True)

    sources, targets, predictions = read_examples(
        fpath=fpath,
        task_name=task_name,
        prediction_key=prediction_key,
        task2fields=task2fields,
    )

    bertscore = load("bertscore")
    bertscore_f1 = bertscore.compute(
        predictions=predictions,
        references=targets,
        model_type="bert-base-multilingual-cased",
        num_layers=9,
        batch_size=batch_size,
    )["f1"]
    bertscore_f1_avg = round(np.mean(bertscore_f1) * 100, 3)

    print(f"Prediction fpath: {fpath}\n\nBERTScore F1: {bertscore_f1_avg}", flush=True)
    print(f"Saving to: {out_fpath}", flush=True)

    save_results(obj={"bertscore_f1_avg": bertscore_f1_avg}, fpath=out_fpath)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fpath",
        type=str,
        help="path to a model output file in the lm-evaluation-harness format.",
        required=True,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="name of the task according to the lm-evaluation-harness configs.",
        required=True,
    )
    parser.add_argument(
        "--prediction_key",
        type=str,
        help="name of the datafield for predictions: usually 'resps' or 'filtered_resps'.",
        default="filtered_resps",
        required=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size for BERTScore.",
        required=False,
        default=64,
    )
    parser.add_argument(
        "--out_fdir",
        type=str,
        help="path to an output directory for saving the results.",
        required=True,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    fpath = args.fpath
    prompt_number = fpath.split("prompt-")[-1].replace(".jsonl", "")
    print(prompt_number, flush=True)
    print(f"Out: {args.out_fdir}", flush=True)
    out_fpath = os.path.join(args.out_fdir, f"prompt_{prompt_number}_bertscore_f1.json")
    evaluate(
        fpath=fpath,
        out_fpath=out_fpath,
        task_name=args.task_name,
        prediction_key=args.prediction_key,
        batch_size=args.batch_size,
        task2fields=TASK2FIELDS,
    )


if __name__ == "__main__":
    main()
