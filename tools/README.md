## Additional tools for NorEval

### Task groups

See the `groups/` directory. The NorEval tasks are grouped by their nature and by their evaluation metrics (to ease aggregation).
Other groupings are, of course, possible.

### Processing and visualizing results

The `process_results.py` takes as an input a directory with the output of `lm_eval_harness` and produces informative plots on comparative models performance in the chosen task groups.
### Task metadata

`build_metadata.py` writes `noreval_metadata.json` in the repository root from the `metadata` blocks of the task configs: the category, language(s), primary metric, random baseline and maximum score of each task, and the names of its variants. It requires `lm-eval`; rerun it after changing the task configs:

```bash
python tools/build_metadata.py
```
