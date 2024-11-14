# Controllable Context Sensitivity and the Knob Behind It
Paper Link: https://arxiv.org/abs/2411.07404

## Getting started
First, clone the repo:
```
git clone git@github.com:kdu4108/context-vs-prior-finetuning.git
```

Create an environment and install dependencies via virtualenv/pip:
```
python3.10 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Running experiments from the paper
Here are the steps to regenerate the experiments in the paper. The key steps are (1) accessing or regenerating the dataset and (2) running the main entry point, `main.py`.

### Generating the dataset
#### BaseFakepedia and MultihopFakepedia
Run all cells in `preprocessing/preprocess_fakepedia_train_and_dev.ipynb`.

#### Arithmetic
Run `python preprocessing/generate_arithmetic.py`.

### Running a single experiment
The main entry point to run a single experiment is `main.py`. The most important arguments to this script are:
* `DATASET_NAME` (positional argument, determines which dataset to run the experiment on. Must exactly match the name of a dataset defined in `preprocessing/datasets.py`).
* `--SUBSPLIT` (the subsplit of the dataset to use)
* `--MODEL_ID` (the model name in huggingface)
* `--CONTEXT_WEIGHT_FORMAT` (the context weight format to use for training examples, e.g., `float` or `instruction`.)
* `--EVALS` (the list of evals to run the model on. Must be a List of Dicts containing at least `dataset_name`, `subsplit`, `k_demonstrations`, and `context_weight_format`. Optionally also include `do_steering`.)
* `--PROJECTION-PATH` (path to a saved projection for training)

The remaining arguments (visible via `python main.py --help`) are either dataset-specific (e.g., specify the `--QUERY_ID` if running an experiment with `DATASET_NAME="YagoECQ"`), allow for control over other experiment details (e.g., which query types to use, the model's batch size for inference, how to sample entities, etc.), or steering-specific hyperparameters (e.g., steering values).

An example command to finetune a model and evaluate on the same dataset is:
```
python main.py BaseFakepedia -M meta-llama/Meta-Llama-3.1-8B-Instruct -S 3 -TS 2048 -TSS 1000 -P -CWF float -O -EV '{"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "float", "do_steering": False}'
```

### Running the full suite of experiments
If you have access to a slurm cluster, you can kick off the full suite of experiments via
```
python slurm/batch_submit_main.py
```
