# 🇳🇴 NorEval

NorEval is a multi-task Norwegian language understanding and generation evaluation benchmark. 

## 🔥 Updates
* **`16.05.2025`**: 🎉 Our paper is accepted to [ACL 2025 Findings](https://2025.aclweb.org/program/find_papers/).
* **`07.05.2025`**: 🫰 NorEval is now integrated into [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/noreval).
* **`10.04.2025`**: 📕 Our pre-print is available on [arXiv](https://arxiv.org/abs/2504.07749).
* **`09.04.2025`**: 🚀 We release NorEval, including our [annotation guidelines](./guidelines/) and novel datasets ([NorRewrite-Instruct](https://huggingface.co/datasets/ltg/norrewrite-instruct) & [NorSummarize-Instruct](https://huggingface.co/datasets/ltg/norsummarize-instruct)).

## 📖 Contents

- [Overview](#😎-overview)
- [Tasks](#🗃️-tasks)
- [Installation and Usage](#👨🏻‍💻-installation-and-usage)
- [Cite Us](#📝-cite-us)

## 😎 Overview

![noreval](noreval.jpg)

**Overview of the NorEval design.**  😼 denotes datasets used in [NorBench](https://aclanthology.org/2023.nodalida-1.61/), [NLEBench](https://aclanthology.org/2024.emnlp-main.317/), [ScandEval](https://aclanthology.org/2023.nodalida-1.20/), and [SEB](https://proceedings.neurips.cc/paper_files/paper/2024/file/4746bb91bd073ec7eef930d5775122ba-Paper-Datasets_and_Benchmarks_Track.pdf), 🚀 represents datasets that have not been used in the existing Norwegian benchmarks, and 😎 denotes our novel datasets introduced as part of NorEval. EN=English; BM=Norwegian Bokmål; NN=Norwegian Nynorsk.

🇳🇴 NorEval combines 19 existing peer-reviewed datasets with five datasets created from scratch ([NCB](https://huggingface.co/datasets/hcfa/ncb), [NorRewrite-Instruct](https://huggingface.co/datasets/ltg/norrewrite-instruct), [NorSummarize-Instruct](https://huggingface.co/datasets/ltg/norsummarize-instruct) for Norwegian Bokmål, and [NorIdiom](https://huggingface.co/datasets/Sprakbanken/Norwegian_idioms) for Norwegian Bokmål and Nynorsk). NorEval covers nine diverse task categories, including sentiment analysis, Norwegian language knowledge, Norwegian-specific \& world knowledge, machine reading comprehension, commonsense reasoning, machine translation, text summarization, instruction following, and truthfulness. Our main evaluation principles are:

- 🌐 **Linguistic diversity**: support for both of the official written standards of Norwegian: Bokmål and Nynorsk (the minority variant).
- 📊 **Task diversity**: coverage of various least addressed tasks for Norwegian. In particular, only three out of 24 NorEval datasets are included in existing Norwegian benchmarks to date: [NorBench](https://aclanthology.org/2023.nodalida-1.61/), [NLEBench](https://aclanthology.org/2024.emnlp-main.317/), [ScandEval](https://aclanthology.org/2023.nodalida-1.20/), and [SEB](https://proceedings.neurips.cc/paper_files/paper/2024/file/4746bb91bd073ec7eef930d5775122ba-Paper-Datasets_and_Benchmarks_Track.pdf).
- 🧠 **Data quality**: focus on only peer-reviewed human-created datasets to ensure reliable evaluation in the context of the Norwegian language, culture, and values.
- 📏 **Prompt sensitivity**: evaluation across 100+ human-written prompts to account for the prompt sensitivity.
- 👩🏻‍🔬 **Standardized evaluation**: integration of NorEval into [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) for flexible and reproducible evaluation.

# 🗃️ Tasks

We group our datasets into text classification, sentence ranking, sentence completion, multiple-choice question answering, generative question answering, and sequence-to-sequence generation tasks. We refer the reader to our paper for more details and describe our tasks below.

|Name  |Bokmål | Nynorsk  |*k*-shot | Task type  | Task category |
|:---|:---|:---|:---|:---|:---|
|[NoReC Sentence](https://huggingface.co/datasets/ltg/norec_sentence) |```norec_sentence```  | ❌ |✅ |Text classification| Sentiment analysis |
|[NoReC Document](https://huggingface.co/datasets/ltg/norec_document) |```norec_document```  | ❌ |✅ |Text classification| Sentiment analysis | 
|[NCB](https://huggingface.co/datasets/hcfa/ncb) |```ncb```| ❌ | ❌ |Sentence ranking| Norwegian language knowledge   |
|[NorIdiom](https://huggingface.co/datasets/Sprakbanken/Norwegian_idioms) |```noridiom_nob```  | ```noridiom_nno```  | ❌ |Sentence completion| Norwegian language knowledge  |
|[Belebele](https://huggingface.co/datasets/facebook/belebele) |```norbelebele```| ❌|❌ |Multiple-choice question answering| Machine reading comprehension |
|[NRK-Quiz-QA](https://huggingface.co/datasets/ltg/nrk_quiz_qa) |```nrk_quiz_qa_nob```| ```nrk_quiz_qa_nno```| ❌   |Multiple-choice question answering| Norwegian-specific & world knowledge | 
|[NorOpenBookQA](https://huggingface.co/datasets/ltg/noropenbookqa) |```noropenbookqa_nob```| ```noropenbookqa_nno``` |✅  |Multiple-choice question answering| Norwegian-specific & world knowledge |
|[NorCommonsenseQA](https://huggingface.co/datasets/ltg/norcommonsenseqa) |```norcommonsenseqa_nob```| ```norcommonsenseqa_nno``` |❌   |Multiple-choice question answering|Commonsense reasoning  |
|[NorTruthfulQA Multiple choice](https://huggingface.co/datasets/ltg/nortruthfulqa_mc) |```nortruthfulqa_mc_nob```| ```nortruthfulqa_mc_nno``` |❌   |Multiple-choice question answering |Truthfulness | 
|[NorQuAD](https://huggingface.co/datasets/ltg/norquad) |```norquad```| ❌  | ✅  |Generative question answering |Machine reading comprehension | 
|[NorTruthfulQA Generation](https://huggingface.co/datasets/ltg/nortruthfulqa_gen) |```nortruthfulqa_gen_nob```| ```nortruthfulqa_gen_nno``` | ❌   | Generative question answering|Truthfulness |
|[ASK-GEC](https://huggingface.co/datasets/ltg/ask-gec) |```ask_gec```| ❌ |✅ |Sequence-to-sequence generation|Norwegian language knowledge | 
|[NorSumm](https://huggingface.co/datasets/SamiaT/NorSumm)  |```norsumm_nob``` | ```norsumm_nno```  |✅ |Sequence-to-sequence generation|Text summarization |
|[Tatoeba (English → Bokmål/Nynorsk)](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt) | ```tatoeba_eng_nob```| ```tatoeba_eng_nno```  |✅  |Sequence-to-sequence generation|Machine translation | 
|[Tatoeba (Bokmål/Nynorsk → English)](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt) | ```tatoeba_nob_eng```| ```tatoeba_nno_eng```  |✅  |Sequence-to-sequence generation|Machine translation |
|[NorRewrite-Instruct](https://huggingface.co/datasets/ltg/norrewrite-instruct) |```norrewrite_instruct```  |❌ |❌ |Sequence-to-sequence generation|Instruction following|
|[NorSummarize-Instruct](https://huggingface.co/datasets/ltg/norsummarize-instruct) |```norsummarize_instruct``` |❌ |❌ |Sequence-to-sequence generation|Instruction following|

<details>
<summary><b>Table description</b></summary>

* **Name**: a dataset name with a HuggingFace link.
* **Bokmål**: the LM Evaluation Harness task name for the Norwegian Bokmål dataset.
* **Nynorsk**: the LM Evaluation Harness task name for the Norwegian Nynorsk dataset, if available.
* **k-shot**: the support for *k*-shot evaluation regimes with *k* > 0. We follow the original datasets' design and focus mainly on the zero-shot evaluation by default. 
  * ✅ means that the user can run the evaluation in both zero-shot and *k*-shot regimes.
  * ❌ denotes that only the zero-shot evaluation regime is available due to the lack of the training or validation set to sample the demonstration examples from. Technically, *k*-shot evaluation on the test set is possible using sampling without replacement, given that the model is not proprietary and not accessed via an API.
* **Task type**: the task type.
* **Task category**: the task category.

</details>

## 👨🏻‍💻 Installation and Usage

Install LM Evaluation Harness as described [here](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install).

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### Examples

Detailed guidelines on how to use LM Evaluation Harness can be found [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md).

<details>
<summary><b>Example 1: Zero-shot evaluation on NorQuAD across five prompts.</b></summary>

```bash
lm_eval \
  --model hf \
  --model_args pretrained=norallm/normistral-7b-warm \
  --tasks norquad \
  --output results/norquad/0-shot/\
  --log_samples \
  --show_config \
  --write_out \
  --batch_size auto \
  --num_fewshot 0
```

</details>


<details>

<summary><b>Example 2: One-shot evaluation on NorQuAD across five prompts.</b></summary>

```bash
lm_eval \
  --model hf \
  --model_args pretrained=norallm/normistral-7b-warm \
  --tasks norquad \
  --output results/norquad/0-shot/ \
  --log_samples \
  --show_config \
  --write_out \
  --batch_size auto \
  --num_fewshot 1
```

</details>



<details>
<summary><b>Example 3: Zero-shot evaluation on NorQuAD using one prompt of interest.</b></summary>

All prompts are numbered from `0` to `6`, and the corresponding configuration files for all supported prompts can be found [in the task directories](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/noreval).

```bash
lm_eval \
  --model hf \
  --model_args pretrained=norallm/normistral-7b-warm \
  --tasks norquad_p0 \
  --output results/norquad_p0/0-shot/ \
  --log_samples \
  --show_config \
  --write_out \
  --batch_size auto \
  --num_fewshot 0
```

</details>

<details>
<summary><b>Example 4: Zero-shot evaluation on task groups.</b></summary>

Consider an example of conducting an evaluation on a task category of interest, e.g., Norwegian-specific & world knowledge. LM Evaluation Harness allows to group tasks as shown below; please find more details [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md#group-configuration).

**Step 1: Create a configuration file**

Create a configuration file containing the name of the group and corresponding tasks and save it in the `lm_eval/tasks/noreval` folder.

```bash
group: norwegian_specific_and_world_knowledge_tasks_nob
task:
  - nrk_quiz_qa_nob
  - noropenbookqa_nob
aggregate_metric_list:
  - metric: acc
    weight_by_size: True
```

**Step 2: Run the evaluation**

Here, we are specifying the name of our created group as ```tasks``` and pass the `include_path` argument to ensure our group is registered:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=norallm/normistral-7b-warm \
  --tasks norwegian_specific_and_world_knowledge_tasks_nob \
  --include_path ./lm_eval/tasks/noreval/ \
  --output results/norwegian_specific_and_world_knowledge_tasks_nob/0-shot/ \
  --log_samples \
  --show_config \
  --write_out \
  --batch_size auto \
  --num_fewshot 0
```

</details>

<details>
<summary><b>Example 5: Zero-shot evaluation on ASK-GEC, which requires computation of the performance metric using a separate script.</b></summary>

Here, we use the `--predict_only` argument and compute the performance metrics as described below.

**Step 1: Generate the predictions**

```bash
lm_eval \
  --model hf \
  --model_args pretrained=AI-Sweden-Models/Llama-3-8B \
  --tasks ask_gec \
  --output results/ask_gec/0-shot/ \
  --log_samples \
  --show_config \
  --write_out \
  --predict_only \
  --batch_size auto \
  --num_fewshot 0
```

**Step 2: Evaluate the predictions with ERRANT**

* Please refer to the installation instructions [here](https://github.com/chrisjbryant/errant/tree/main).
* Run the following:
    ```bash
    python3 lm_eval/tasks/noreval/ask_gec/errant.py --fpath results/ask_gec/0-shot/AI-Sweden-Models__Llama-3-8B/samples_ask_gec_p0_2025-01-28T01-08-13.454441.jsonl --out_fdir results/ask_gec/0-shot/AI-Sweden-Models__Llama-3-8B/
    ```
* The results will be saved as `results/ask_gec/0-shot/AI-Sweden-Models__Llama-3-8B/samples_ask_gec_p0_2025-01-28T01-08-13.454441_errant.json`

</details>

<details>
<summary><b>Comment: BERTScore.</b></summary>

In our paper, we compute BERTScore for most sequence-to-sequence generation tasks *offline* after running the evaluation with the ```--predict_only``` argument for efficiency.

</details>

# 📝 Cite Us

```
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
    abstract = "This paper introduces NorEval, a new and comprehensive evaluation suite for large-scale standardized benchmarking of Norwegian generative language models (LMs). NorEval consists of 24 high-quality human-created datasets {--} of which five are created from scratch. In contrast to existing benchmarks for Norwegian, NorEval covers a broad spectrum of task categories targeting Norwegian language understanding and generation, establishes human baselines, and focuses on both of the official written standards of the Norwegian language: Bokm{\r{a}}l and Nynorsk. All our datasets and a collection of over 100 human-created prompts are integrated into LM Evaluation Harness, ensuring flexible and reproducible evaluation. We describe the NorEval design and present the results of benchmarking 19 open-source pretrained and instruction-tuned LMs for Norwegian in various scenarios. Our benchmark, evaluation framework, and annotation materials are publicly available."
}
```
