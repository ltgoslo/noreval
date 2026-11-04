# 🇳🇴 NorEval

NorEval is a multi-task Norwegian language understanding and generation evaluation benchmark. 

### 🔥 Updates
* **`03.11.2026`**: 📌 We release [NorEval v1.1.1](https://github.com/ltgoslo/noreval/releases/tag/v1.1.1), which pins every dataset to a fixed version for reproducible evaluation and fixes the Tatoeba English ↔ Bokmål/Nynorsk tasks.
* **`17.10.2026`**: 🆕 We release [NorEval v1.1](https://github.com/ltgoslo/noreval/releases/tag/v1.1) with new tasks (SLIDE, NoCoLA, MultiBLiMP, NorOpenBookQA without facts, Bokmål ↔ Nynorsk and Bokmål ↔ Northern Sámi translation) and updated task configs.
* **`15.10.2026`**: 📊 Our interactive dashboard is available [here](https://ltgoslo.github.io/llm-dashboard/noreval/).
* **`16.05.2025`**: 🎉 Our paper is accepted to [ACL 2025 Findings](https://2025.aclweb.org/program/find_papers/).
* **`07.05.2025`**: 🫰 NorEval is now integrated into [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/noreval).
* **`10.04.2025`**: 📕 Our pre-print is available on [arXiv](https://arxiv.org/abs/2504.07749).
* **`09.04.2025`**: 🚀 We release NorEval v1.0, including our [annotation guidelines](./guidelines/) and novel datasets ([NorRewrite-Instruct](https://huggingface.co/datasets/ltg/norrewrite-instruct) & [NorSummarize-Instruct](https://huggingface.co/datasets/ltg/norsummarize-instruct)).

### 📖 Contents

- [Overview](#-overview)
- [Dashboard](#-dashboard)
- [Tasks](#️-tasks)
- [How to use](#‍-how-to-use)
- [Annotation guidelines and tools](#-annotation-guidelines-and-tools)
- [Citation](#-citation)

### 😎 Overview

NorEval is a multi-task Norwegian language understanding and generation evaluation benchmark that combines existing peer-reviewed datasets with new datasets manually created from scratch ([NCB](https://huggingface.co/datasets/hcfa/ncb), [NorRewrite-Instruct](https://huggingface.co/datasets/ltg/norrewrite-instruct), [NorSummarize-Instruct](https://huggingface.co/datasets/ltg/norsummarize-instruct) for Norwegian Bokmål, and [NorIdiom](https://huggingface.co/datasets/Sprakbanken/Norwegian_idioms) for Norwegian Bokmål and Nynorsk). NorEval covers diverse task categories: sentiment analysis, Norwegian language knowledge, Norwegian-specific \& world knowledge, machine reading comprehension, commonsense reasoning, machine translation, text summarization, instruction following, and truthfulness. Our main evaluation principles are:

- 🌐 **Linguistic diversity**: support for both of the official written standards of Norwegian: Bokmål and Nynorsk (the minority variant).
- 📊 **Task diversity**: coverage of various least addressed tasks for Norwegian. In particular, only three out of 24 NorEval datasets are included in existing Norwegian benchmarks to date: [NorBench](https://aclanthology.org/2023.nodalida-1.61/), [NLEBench](https://aclanthology.org/2024.emnlp-main.317/), [ScandEval](https://aclanthology.org/2023.nodalida-1.20/), and [SEB](https://proceedings.neurips.cc/paper_files/paper/2024/file/4746bb91bd073ec7eef930d5775122ba-Paper-Datasets_and_Benchmarks_Track.pdf).
- 🧠 **Data quality**: focus on only peer-reviewed human-created datasets to ensure reliable evaluation in the context of the Norwegian language, culture, and values.
- 📏 **Prompt sensitivity**: evaluation across multiple human-written prompts to account for the prompt sensitivity.
- 👩🏻‍🔬 **Standardized evaluation**: integration of NorEval into LM Evaluation Harness for flexible and reproducible evaluation.


### 📊 Dashboard

Explore the results of benchmarking Norwegian and multilingual models on NorEval in our [interactive dashboard](https://ltgoslo.github.io/llm-dashboard/noreval/). The dashboard lets you compare models on individual tasks or category averages across 0-shot, 1-shot, and 5-shot regimes, choose how the scores are aggregated over prompts and normalized, and filter models by size or by whether they are fully open.

[![NorEval dashboard](images/dashboard.png)](https://ltgoslo.github.io/llm-dashboard/noreval/)

### 🗃️ Tasks

|Name  |Bokmål | Nynorsk  | Northern Sámi | Task formulation  | Task category |
|:---|:---|:---|:---|:---|:---|
|[NoReC Sentence](https://huggingface.co/datasets/ltg/norec_sentence) |`norec_sentence`  | — | — |Text classification| Language understanding |
|[NoReC Document](https://huggingface.co/datasets/ltg/norec_document) |`norec_document`  | — | — |Text classification| Language understanding |
|[SLIDE](https://huggingface.co/datasets/ltg/slide) |`slide`  | `slide` | — |Text classification| Linguistic knowledge |
|[NCB](https://huggingface.co/datasets/hcfa/ncb) |`ncb`| — | — |Text classification| Linguistic knowledge   |
|[NoCoLA](https://huggingface.co/datasets/ltg/nocola) |`nocola`| — | — |Text classification| Linguistic knowledge   |
|[MultiBLiMP](https://huggingface.co/datasets/jumelet/multiblimp) | — | — |`noreval_multiblimp`|Text classification| Linguistic knowledge |
|[NorIdiom](https://huggingface.co/datasets/Sprakbanken/Norwegian_idioms) |`noridiom_nob`  | `noridiom_nno`  | — |Text classification| Linguistic knowledge |
|[NorBelebele](https://huggingface.co/datasets/ltg/norbelebele) |`norbelebele`| —| — |Text classification| Language understanding |
|[NRK-Quiz-QA](https://huggingface.co/datasets/ltg/nrk_quiz_qa) |`nrk_quiz_qa_nob`| `nrk_quiz_qa_nno`| — |Text classification| World knowledge and reasoning |
|[NorOpenBookQA](https://huggingface.co/datasets/ltg/noropenbookqa) |`noropenbookqa_nob`| `noropenbookqa_nno` | — |Text classification| Language understanding |
|[NorOpenBookQA (without facts)](https://huggingface.co/datasets/ltg/noropenbookqa) |`noropenbookqa_no_fact_nob`| `noropenbookqa_no_fact_nno` | — |Text classification| World knowledge and reasoning |
|[NorCommonsenseQA](https://huggingface.co/datasets/ltg/norcommonsenseqa) |`norcommonsenseqa_nob`| `norcommonsenseqa_nno` | — |Text classification|World knowledge and reasoning  |
|[NorTruthfulQA Multiple choice](https://huggingface.co/datasets/ltg/nortruthfulqa_mc) |`nortruthfulqa_mc_nob`| `nortruthfulqa_mc_nno` | — |Text classification | World knowledge and reasoning |
|[NorTruthfulQA Generation](https://huggingface.co/datasets/ltg/nortruthfulqa_gen) |`nortruthfulqa_gen_nob`| `nortruthfulqa_gen_nno` | — | Text generation|World knowledge and reasoning |
|[NorQuAD](https://huggingface.co/datasets/ltg/norquad) |`norquad`| —  | — |Text generation| Language understanding |
|[ASK-GEC](https://huggingface.co/datasets/ltg/ask-gec) |`ask_gec`| — | — |Text generation| Linguistic knowledge |
|[NorSumm](https://huggingface.co/datasets/SamiaT/NorSumm)  |`norsumm_nob` | `norsumm_nno`  | — |Text generation| Generation and summarization |
|[NorSumm Translation (Bokmål ↔ Nynorsk)](https://huggingface.co/datasets/ltg/norsumm-nob-nno-translation) | `norsumm_nob_nno_translation`| `norsumm_nno_nob_translation`  | — |Text generation|Generation and summarization |
|[Tatoeba (English → Bokmål/Nynorsk)](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt) | `tatoeba_eng_nob`| `tatoeba_eng_nno`  | — |Text generation|Machine translation |
|[Tatoeba (Bokmål/Nynorsk → English)](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt) | `tatoeba_nob_eng`| `tatoeba_nno_eng`  | — |Text generation|Machine translation |
|[Tatoeba (Bokmål ↔ Northern Sámi)](https://huggingface.co/datasets/ltg/saami-tatoeba) | `tatoeba_sme_nob` | — | `tatoeba_nob_sme`|Text generation|Machine translation |
|[NorRewrite-Instruct](https://huggingface.co/datasets/ltg/norrewrite-instruct) |`norrewrite_instruct`  | — | — |Text generation|Generation and summarization|
|[NorSummarize-Instruct](https://huggingface.co/datasets/ltg/norsummarize-instruct) |`norsummarize_instruct` | — | — |Text generation|Generation and summarization|

<details>
<summary><b>Table description</b></summary>

* **Name**: a dataset name with a HuggingFace link.
* **Bokmål**: the LM Evaluation Harness task name for the Norwegian Bokmål dataset. `slide`, which covers Bokmål, Nynorsk, Danish, and Swedish, is also listed here.
* **Nynorsk**: the LM Evaluation Harness task name for the Norwegian Nynorsk dataset, if available.
* **Northern Sámi**: the LM Evaluation Harness task name for the Northern Sámi dataset, if available. Translation tasks between Bokmål and Northern Sámi are listed here.
* **Task type**: the task type.
* **Task category**: the task category.

</details>

### 👨🏻‍💻 How to use

The tasks are plain [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) configs — no fork of the harness is needed. Install the harness, clone this repository, and point the harness to it with `--include_path`:

```bash
pip install lm-eval  # or lm-eval[vllm] for the vLLM backend
git clone https://github.com/ltgoslo/noreval.git
```

Example: zero-shot evaluation on NorQuAD across all five prompts:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=norallm/normistral-7b-warm \
  --include_path noreval \
  --tasks norquad \
  --output results/norquad/0-shot/ \
  --show_config \
  --write_out \
  --batch_size auto \
  --num_fewshot 0
```

* The task names in the table above run all prompt variants of a dataset at once; append a prompt index to run a single variant (e.g. `norquad_p2`).
* Tasks loaded via `--include_path` take precedence over any same-named NorEval tasks bundled with the harness, so this repository is the source of truth for the task definitions.


### 🧰 Annotation guidelines and tools

* [`guidelines/`](./guidelines/): the annotation guidelines for the human baselines, prompt writing, and the creation of NorRewrite-Instruct and NorSummarize-Instruct.
* [`tools/`](./tools/): example task groups for aggregating results by category and a script for processing and plotting the results.


### 📝 Citation

```bibtex
@inproceedings{mikhailov-etal-2025-noreval,
    title = "{N}or{E}val: A {N}orwegian Language Understanding and Generation Evaluation Benchmark",
    author = "Mikhailov, Vladislav  and
      Enstad, Tita  and
      Samuel, David  and
      Farseth{\r{a}}s, Hans Christian  and
      Kutuzov, Andrey  and
      Velldal, Erik  and
      {\O}vrelid, Lilja",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.181/",
    doi = "10.18653/v1/2025.findings-acl.181",
    pages = "3495--3541",
    ISBN = "979-8-89176-256-5",
}
```
