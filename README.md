# Noreval

## Codebase
The evaluation codebase relies on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) framework codebase (version 0.4.2). In particular, each task in this evaluation suite is integrated into the framework with the help of ```.yaml``` configuration files, which allow for high task customizability.


## Tasks
Noreval includes various tasks, which are divided the (i) text classification, (ii) question answering, (iii) ranking sentence pairs, and (iv) text generation groups.

Each table below has the following columns:
* **Task name (default)**: a task name for a configuration file with *one* default prompt. Even if the task has both Bokmål and Nynorsk dataset versions, the default one is set to Bokmål.
* **Bokmål / Nynorsk (N prompts)**: task names for configuration files with *N* prompts for the Bokmål and Nynorsk dataset versions, respectively.
  * *N* is in range between 4 and 6. Tasks in the "ranking sentence pairs" group do not utilize any prompts; the evaluation design includes selecting the most probable sentence.
  * ❌ means that the task does not have a dataset for a given written standard.
* **0-shot / few-shot**: support for the zero-shot and few-shot evaluation setups, respectively.
  * ✅ means that one can run evaluation in a given setup.
  * ❌ denotes that a given setup is not supported due to the lack of the training or validation set to sample the demonstration examples from.
* **Task category**: task formulation or task category.
* **HuggingFace**: a link to the dataset on HuggingFace.

### Note

#### (1) HF datasets from the ```mimir-project``` repository are currently not available; to be released later.

#### (2) The ```.yaml``` configuration files are created according to version 0.4.2 of the ```lm-evaluation-harness``` framework. These are in the process of being updated to a more recent version. However, you can still use 0.4.2 for empirical evaluation experiments. 

<details open>
    <summary>Text classification</summary>

|Task name (default)  |Bokmål/Nynorsk (N prompts)   |0-shot / few-shot   |Task category  | HuggingFace   |
|:---|:---|:---|:---|:---|
|```norec_sentence``` |```norec_sentence_nb``` / ❌ |✅ / ✅  |Sentiment analysis | [ltg/norec_sentence](https://huggingface.co/datasets/ltg/norec_sentence)  |
|```norec_document``` |```norec_document_nb``` / ❌ |✅ / ✅  |Sentiment analysis | [ltg/norec_document](https://huggingface.co/datasets/ltg/norec_document)  |
</details>

<details open>
    <summary>Question answering</summary>

|Task name (default)  |Bokmål / Nynorsk (N prompts)   |0-shot / few-shot   |Task category  | HuggingFace   |
|:---|:---|:---|:---|:---|
|```norquad``` |```norquad_nb``` / ❌  |✅ / ✅  |Reading comprehension | [ltg/norquad](https://huggingface.co/datasets/ltg/norquad)  |
|```belebele_nob_Latn``` |```norbelebele_nb``` / ❌|✅ / ❌   |Reading comprehension | [facebook/belebele](https://huggingface.co/datasets/facebook/belebele)  |
|```nrk``` |```nrk_nb``` / ```nrk_nn``` |✅ / ❌   |World knowledge | [mimir-project/nrk](https://huggingface.co/datasets/mimir-project/nrk)  |
|```noropenbookqa``` |```noropenbookqa_nb``` / ```noropenbookqa_nn```  |✅ / ✅  |World knowledge | [mimir-project/noropenbookqa](https://huggingface.co/datasets/mimir-project/noropenbookqa)  |
|❌ (use ```fact``` as part of the input) |```noropenbookqa_nb_use_fact``` / ```noropenbookqa_nn_use_fact```  |✅ / ✅  |World knowledge | [mimir-project/noropenbookqa](https://huggingface.co/datasets/mimir-project/noropenbookqa)  |
|```norcommonsenseqa``` |```norcommonsenseqa_nb``` / ```norcommonsenseqa_nn```  |✅ / ❌   |Commonsense reasoning  | [mimir-project/norcommonsenseqa](https://huggingface.co/datasets/mimir-project/norcommonsenseqa)  |
|```nortruthfulqa_mc``` |```nortruthfulqa_mc_nb``` / ```nortruthfulqa_mc_nn```  |✅ / ❌   |Fairness & truthfulness | [mimir-project/nortruthfulqa_mc](https://huggingface.co/datasets/mimir-project/nortruthfulqa_mc)  |
</details>

<details open>
    <summary>Ranking sentence pairs</summary>

|Task name (default)  |Bokmål/Nynorsk (N prompts)   |0-shot / few-shot   |Task category  | HuggingFace   |
|:---|:---|:---|:---|:---|
|```mimir_bias``` |❌ / ❌ |✅ / ❌  |Fairness & truthfulness | [mimir-project/mimir-bias](https://huggingface.co/datasets/mimir-project/mimir-bias)  |
|```ncb``` |❌ / ❌ |✅ / ❌ |Norwegian language: grammar, punctuation, and idioms | [hcfa/ncb](https://huggingface.co/datasets/hcfa/ncb)  |
</details>

<details open>
    <summary>Text generation</summary>

|Task name (default)  |Bokmål/Nynorsk (N prompts)   |0-shot / few-shot   |Task category  | HuggingFace   |
|:---|:---|:---|:---|:---|
|```noridiom``` |```noridiom_nb``` / ```noridiom_nn``` |✅ / ❌  |Norwegian language: grammar, punctuation, and idioms | [mimir-project/noridiom](https://huggingface.co/datasets/mimir-project/noridiom)  |
|```ask_gec``` |```ask_gec_nb``` / ❌ |✅ / ✅  |Norwegian language: grammar, punctuation, and idioms | [ltg/ask-gec](https://huggingface.co/datasets/ltg/ask-gec)  |
|```norsumm``` |```norsumm_nb``` / ```norsumm_nn``` |✅ / ❌ |Text summarization | [mimir-project/norsumm](https://huggingface.co/datasets/mimir-project/norsumm)  |
|```nortruthfulqa_gen``` |```nortruthfulqa_gen_nb``` / ❌  |✅ / ❌   |Fairness & truthfulness | [mimir-project/nortruthfulqa_gen](https://huggingface.co/datasets/mimir-project/nortruthfulqa_gen)  |
|```tatoeba_eng_nno``` (English → Nynorsk) |❌ / ```tatoeba_eng_nno_nn``` |✅ / ✅  |Machine translation | [Helsinki-NLP/tatoeba_mt](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt)  |
|```tatoeba_nno_eng``` (Nynorsk → English) |❌ / ```tatoeba_eng_nno_nn``` |✅ / ✅  |Machine translation | [Helsinki-NLP/tatoeba_mt](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt)  |
|```tatoeba_eng_nob``` (English → Bokmål) | ```tatoeba_eng_nob_nb``` / ❌  |✅ / ✅  |Machine translation | [Helsinki-NLP/tatoeba_mt](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt)  |
|```tatoeba_nob_eng``` (Bokmål → English) | ```tatoeba_nob_eng_nb``` / ❌  |✅ / ✅  |Machine translation | [Helsinki-NLP/tatoeba_mt](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt)  |
|```tatoeba_nob_nno``` (Bokmål → Nynorsk) | ```tatoeba_nob_nno_nb``` / ❌  |✅ / ❌ |Machine translation | [Helsinki-NLP/tatoeba_mt](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt)  |
|```tatoeba_nno_nob``` (Nynorsk → Bokmål) | ❌ / ```tatoeba_nno_nob_nn```  |✅ / ❌ |Machine translation | [Helsinki-NLP/tatoeba_mt](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt)  |

</details>

Please find below the links to the relevant framework documentation:
* [the task guide: how the .yaml configuration files are organized.](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)
* [the new task guide: how to integrate your task into the framework.](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md)

## Installation

1. Install one of the latest ```lm-evaluation-harness``` versions:

```bash
pip install --quiet https://github.com/EleutherAI/lm-evaluation-harness/archive/refs/tags/v0.4.2.tar.gz
```

2. Log in to your HuggingFace account. You can get your access token [here](https://huggingface.co/settings/tokens).

```bash
pip install --quiet "huggingface_hub[cli]"
huggingface-cli login --token <YOUR TOKEN>
```

## Usage

The original guidelines on the ```lm-evaluation-harness``` framework interface can be found [here.](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)

### Must-have arguments
The high-level framework usage requires the following arguments:
* `--model_args`: the model type; in our case, it is ```pretrained={model_name}```, where ```model_name``` refers to the model name on HuggingFace (e.g., `pretrained=norallm/normistral-7b-warm`).
* `--tasks`: the name(s) of evaluation tasks (e.g., `norcommonsenseqa_nb` or `noropenbookqa_nb,noropenbookqa_nb_use_fact`).
* `--include_path`: a path to custom configuration files in the `.yaml` format (in our case, it is ```noreval```). this is used to add the noreval tasks to the framework's task registry as available tasks.
* `--log_samples`: allows to save the model inputs and outputs in a directory specified with the help of the `--output` argument.
* `--output`: a path where high-level results will be saved. if one provides `--log_samples`, both model predictions and results will be saved in the specified directory.
* `--write_out`: a complementary function, which prints out the format of the prompts and outputs.
* `--show_config`: a complementary function, which prints out the configuration file.
* `--batch_size`: the batch size. `"auto"` allows to automatically select the largest batch size that will fit in memory, speeding up evaluation.
  * **NB**: depending on the cluster, `"auto"` can still fail due to the out of memory error. the behavior can be controlled with the `--max_batch_size` and `--batch_size auto:N` arguemnts, where `N` stands for the number of times to re-select the maximum batch size during evaluation.
* `--num_fewshot`: the number of demonstrations used in the model input.
* `--limit`: selects first N examples and runs the evaluation on this subset. can be used for debugging or testing purposes.
* `--predict_only`: allows to *not* compute the performance metrics but *only* save the predictions. should be used together with `--log_samples`.

### Examples

In general, one needs to specify the following high-level arguments to conduct an evaluation run:
* `--tasks` (the task name can be found in the tables above).
* `--model_args` (any model on HuggingFace).
* `--batch_size` (`"auto"`; one can test the largest batch size using the complementary arguments mentioned above, such as `--limit`, `--max_batch_size`, and `--batch_size auto:N`).
* `--include_path` (always ```./noreval/```).
* `--num_fewshot` (the supported k-shot setup for a given task can be found in the tables above).

1. Running zero-shot evaluation of the ```norallm/normistral-7b-warm``` model on the ```norquad``` task using a default prompt.

```bash
lm_eval \
  --model hf \
  --model_args pretrained=norallm/normistral-7b-warm \
  --tasks norquad \
  --include_path ./noreval/ \
  --output results/norquad/0-shot/normistral-7b-warm/ \
  --log_samples \
  --show_config \
  --write_out \
  --batch_size auto \
  --num_fewshot 0
```

2. Running 1-shot evaluation of the ```norallm/normistral-7b-warm``` on the ```norquad_nb``` task, which involves testing the model on a set of 5 Norwegian Bokmål prompts.

```bash
lm_eval \
  --model hf \
  --model_args pretrained=norallm/normistral-7b-warm \
  --tasks norquad_nb \
  --include_path ./noreval/ \
  --output results/norquad/0-shot/normistral-7b-warm/ \
  --log_samples \
  --show_config \
  --write_out \
  --batch_size auto \
  --num_fewshot 1
```

3. Running 0-shot evaluation of the ```norallm/normistral-7b-warm``` on the ```ask_gec``` task, which requires computation of the performance metric using a separate script. Here, we use the `--predict_only` argument and compute the performance metrics as described in the next subsection.

```bash
lm_eval \
  --model hf \
  --model_args pretrained=norallm/normistral-7b-warm \
  --tasks ask_gec \
  --include_path ./noreval/ \
  --output results/ask_gec/0-shot/normistral-7b-warm/ \
  --log_samples \
  --show_config \
  --write_out \
  --predict_only \
  --batch_size auto \
  --num_fewshot 0
```

## Comments on performance metrics and inference

### Inference

```lm-evaluation-harness``` supports ```accelerate``` and ```vllm``` to speed up the evaluation. Please refer to the framework documentation for usage examples.


### Performance metrics

1. ```BERTScore``` (used in the text summarization, machine translation, and headline generation tasks)
* Unfortunately, there is an unresolved [bug](https://github.com/EleutherAI/lm-evaluation-harness/issues/1302) related to calculation of the BERTSCore. The current version of the mimir evaluation configuration files follows the proposed workaround; however, it slows the evaluation, since it loads the ```bertscore``` metric during evaluating each batch or prediction-reference pair.
* **Solution**:
  * One can discard computation of the ```BERTScore``` metric from the ```.yaml``` configuration file and use the `--log_samples` argument when conducting the evaluation run. 
  * Then, one can use ```noreval/bertscore.py``` to compute the metric score for a given file with the saved predictions, which significantly reduces the computation costs.
  * *Example*:
    ```bash
    python3 noreval/bertscore.py --fpath results/schibsted_vg/0-shot/normistral-7b-warm/predictions.jsonl --out_fdir results/schibsted_vg/0-shot/normistral-7b-warm/ --task_name schibsted_vg_nb --batch_size 128
    ```
2. ```ERRANT``` (used in the grammar error correction task)
* This metric is calculated using a separate evaluation script, which can be found in ```noreval/ask_gec/errant.py```.
* Please refer to the installation instructions [here](https://github.com/chrisjbryant/errant/tree/main).
* *Example*:
    ```bash
    python3 ask_gec/errant.py --fpath results/ask_gec/0-shot/normistral-7b-warm/predictions.jsonl --out_fdir results/ask_gec/0-shot/normistral-7b-warm/
    ```
