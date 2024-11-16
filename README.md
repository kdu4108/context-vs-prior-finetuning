# Controllable Context Sensitivity and the Knob Behind It
Paper Link: https://arxiv.org/abs/2411.07404

## Getting started
First, clone the repo:
```
git clone git@github.com:kdu4108/context-vs-prior-finetuning.git
```

Create an environment and install dependencies via virtualenv/pip:
```
python3.11 -m venv env
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
Run the following command to submit all experiments for a given model (`llama`, `mistral`, `gemma`).
```
python generate_run_scripts.py --model-id llama --add-default --add-steering --add-oos-datasets --add-training && bash run_scripts.sh
```

## Interpretability Analysis

The scripts for the interpretability analysis are in the `analysis` directory. 
It is mainly based on the [`nnsight`](http://nnsight.net) and [`nnpatch`](https://github.com/jkminder/nnpatch) libraries.

Check the notebooks `notebooks/analysis_llama.ipynb`, `notebooks/analysis_mistral.ipynb` and `notebooks/analysis_gemma.ipynb` for the analysis in section 5 and 6 of the paper. To regenerate the plots, first generate the orthogonal projections using the notebooks mentioned, run all experiments and then run the `analysis/plots_das.ipynb` notebook.

You can also use the existing projections, which are hosted on huggingface. 
- [Meta-Llama-3.1-8B-Instruct-L16](https://huggingface.co/jkminder/CTXPRIOR-Projection-Meta-Llama-3.1-8B-Instruct-L16) Projection for the Meta-Llama-3.1 family of models. Layer 16. Recommended steering values: `prior=6`, `context=-6`.
- [gemma-2-9b-it-L27](https://huggingface.co/jkminder/CTXPRIOR-Projection-gemma-2-9b-it-L27) Projection for the gemma-2-9b-it family of models. Layer 27. Recommended steering values: `prior=-100`, `context=150`.
- [Mistral-7B-Instruct-v0.3-L16](https://huggingface.co/jkminder/CTXPRIOR-Projection-Mistral-7B-Instruct-v0.3-L16) Projection for the Mistral family of models. Layer 16. Recommended steering values: `prior=5`, `context=-5`.

Check the [`analysis/demo_steering.ipynb`](analysis/demo_steering.ipynb) notebook for a demo of how to use the steering hook.

If you want the basis vector of the subspace, use the following snippet to get it:

```python
proj = LowRankOrthogonalProjection.from_pretrained("jkminder/CTXPRIOR-Projection-Meta-Llama-3.1-8B-Instruct-L16")
u = proj.weight
u.shape # [4096, 1]
```



