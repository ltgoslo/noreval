import datasets
from evaluate import load
from functools import partial
from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask


def _squad_metric(predictions, references):
    squad_metric = load("squad")
    return squad_metric.compute(predictions=predictions, references=references)


def _squad_agg(key, items):
    predictions, references = zip(*items)
    return _squad_metric(predictions=predictions, references=references).get(key, 0)


def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
        doc["title"] = doc["context"].strip().split("\n")[0].strip()
        doc["passage"] = "\n".join(doc["context"].strip().split("\n")[1:]).strip()
        doc["question"] = " ".join(doc["question"].strip().split())
        doc["answer"] = doc["answers"]["text"][0]
        return doc

    return dataset.map(_helper)


class NorQuAD(ConfigurableTask):
    DATASET_PATH = "ltg/norquad"
    VERSION = 1.0
    DATASET_NAME = None

    def __init__(self):
        super().__init__(config={"metadata": {"version": self.VERSION}})

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def construct_requests(self, doc, ctx, **kwargs):
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(
                ctx,
                {
                    "until": ["\n"],
                    "max_new_tokens": 32,
                    "do_sample": False,
                    "num_beams": 1,
                },
            ),
            idx=0,
            **kwargs,
        )

    def doc_to_target(self, doc):
        return " " + doc["answers"]["text"][0]

    def process_results(self, doc, continuation):
        predictions = {
            "id": doc["id"],
            "prediction_text": continuation[0].strip(),
        }
        references = {
            "id": doc["id"],
            "answers": doc["answers"],
        }
        return {
            "exact_match": (
                predictions,
                references,
            ),
            "f1": (
                predictions,
                references,
            ),
        }

    def aggregation(self):
        return {
            "exact_match": partial(_squad_agg, "exact_match"),
            "f1": partial(_squad_agg, "f1"),
        }

    def higher_is_better(self):
        return {"exact_match": True, "f1": True}


class P0NB(NorQuAD):
    def __init__(self):
        super().__init__()

    def doc_to_text(self, doc):
        title = doc["context"].strip().split("\n")[0].strip()
        passage = "\n".join(doc["context"].strip().split("\n")[1:]).strip()
        question = " ".join(doc["question"].strip().split())
        prompt = f"Tittel: {title}\n\nTekst: {passage}\n\nSpørsmål: {question}\n\nSvar:"
        return prompt


class P1NB(NorQuAD):
    def __init__(self):
        super().__init__()

    def doc_to_text(self, doc):
        title = doc["context"].strip().split("\n")[0].strip()
        passage = "\n".join(doc["context"].strip().split("\n")[1:]).strip()
        question = " ".join(doc["question"].strip().split())
        prompt = f'Tittel: {title}\n\nTekst: {passage}\n\nGitt teksten over, hva er svaret på følgende spørsmål? "{question}"\n\nSvar:'
        return prompt


class P2NB(NorQuAD):
    def __init__(self):
        super().__init__()

    def doc_to_text(self, doc):
        title = doc["context"].strip().split("\n")[0].strip()
        passage = "\n".join(doc["context"].strip().split("\n")[1:]).strip()
        question = " ".join(doc["question"].strip().split())
        prompt = f"Tittel: {title}\n\nTekst: {passage}\n\nSvar på følgende: {question}\n\nSvar:"
        return prompt


class P3NB(NorQuAD):
    def __init__(self):
        super().__init__()

    def doc_to_text(self, doc):
        title = doc["context"].strip().split("\n")[0].strip()
        passage = "\n".join(doc["context"].strip().split("\n")[1:]).strip()
        question = " ".join(doc["question"].strip().split())
        prompt = f'Tittel: {title}\n\nTekst: {passage}\n\nHvordan kan man svare på spørsmålet "{question}", gitt teksten over?\n\nSvar:'
        return prompt


class P4NB(NorQuAD):
    def __init__(self):
        super().__init__()

    def doc_to_text(self, doc):
        title = doc["context"].strip().split("\n")[0].strip()
        passage = "\n".join(doc["context"].strip().split("\n")[1:]).strip()
        question = " ".join(doc["question"].strip().split())
        prompt = f'Tittel: {title}\n\nTekst:{passage}\n\nGitt teksten over, besvar følgende spørsmål: "{question}"\n\nSvar:'
        return prompt
