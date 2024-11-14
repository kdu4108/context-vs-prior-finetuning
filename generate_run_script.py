import argparse

def get_ds(ds, k, cwf, steer=False):
    if steer:
        cwf = "none"
    if ds == "BaseFakepedia":
        return f'{{"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": {k}, "context_weight_format": "{cwf}", "do_steering": "{steer}"}}'
    elif ds == "MultihopFakepedia":
        return f'{{"dataset_name": "MultihopFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": {k}, "context_weight_format": "{cwf}", "do_steering": "{steer}"}}'
    elif ds == "Arithmetic":
        return f'{{"dataset_name": "Arithmetic", "subsplit": "d2ub9", "k_demonstrations": {k}, "context_weight_format": "{cwf}", "do_steering": "{steer}"}}'
    else:
        raise ValueError(f"Dataset {ds} not supported")

def get_base_script(train_dataset, in_domain, instruct_model, base_model, seed, bs, ebs, ga, projection_path, prior_value, context_value, steering_layer, add_training=True, add_default=True, add_steering=True, add_ood_datasets=True, add_instruct_generalisation=True): 
    eval_datasets = list(set(["BaseFakepedia", "MultihopFakepedia", "Arithmetic"]).difference(set([train_dataset]))) 
    out = f"""
#!/bin/bash

set -x -e

pip install circuitsvis python-dotenv --no-deps
cd /dlabscratch1/jminder/repositories/context-vs-prior-finetuning/
INSTRUCT_MODEL={instruct_model}
BASE_MODEL={base_model}
SEED={seed}
BS={bs}
GA={ga}
PROJECTION_PATH={projection_path}
PRIOR_VALUE={prior_value}
CONTEXT_VALUE={context_value}
STEERING_LAYER={steering_layer}
ID={'-ID' if in_domain else ''}
EBS={ebs}

BASE_ARGS="-S ${{SEED}} -TS 2048 -EBS ${{EBS}} -TSS 1000 -P -BS ${{BS}} -GA ${{GA}} ${{ID}} -PP ${{PROJECTION_PATH}} -SPV ${{PRIOR_VALUE}} -SCV ${{CONTEXT_VALUE}} -SL ${{STEERING_LAYER}}"
### NORMAL ###"""
    if add_training:
        out += f"""
## Trained Models
# Instruct
python main.py {train_dataset} -M ${{INSTRUCT_MODEL}} ${{BASE_ARGS}} -CWF instruction -O -FC \
-EV '[{get_ds(train_dataset, 0, "instruction", steer=False)}]'
python main.py {train_dataset} -M ${{INSTRUCT_MODEL}} ${{BASE_ARGS}} -CWF float -O -FC \
-EV '[{get_ds(train_dataset, 0, "float", steer=False)}]'

# Base
python main.py {train_dataset} -M ${{BASE_MODEL}} ${{BASE_ARGS}} -CWF instruction -O -FC \
-EV '[{get_ds(train_dataset, 0, "instruction", steer=False)}]'
python main.py {train_dataset} -M ${{BASE_MODEL}} ${{BASE_ARGS}} -CWF float -O -FC \
-EV '[{get_ds(train_dataset, 0, "float", steer=False)}]'
"""
    if add_default:
        out += f"""
## Default Models
python main.py {train_dataset} -NT -M ${{INSTRUCT_MODEL}} ${{BASE_ARGS}} -O -FC \
-EV '[{get_ds(train_dataset, 0, "float", steer=False)},{get_ds(train_dataset, 10, "float", steer=False)},{get_ds(train_dataset, 0, "instruction", steer=False)},{get_ds(train_dataset, 10, "instruction", steer=False)}]'

python main.py {train_dataset} -NT -M ${{BASE_MODEL}} ${{BASE_ARGS}} -O -FC \
-EV '[{get_ds(train_dataset, 0, "float", steer=False)},{get_ds(train_dataset, 10, "float", steer=False)},{get_ds(train_dataset, 0, "instruction", steer=False)},{get_ds(train_dataset, 10, "instruction", steer=False)}]'
"""
    if add_steering:
        out += f"""
### STEERING ###
## Trained Models
python main.py {train_dataset} -M ${{INSTRUCT_MODEL}} ${{BASE_ARGS}} -CWF instruction \
-EV '[{get_ds(train_dataset, 0, "instruction", steer=True)}]'
python main.py {train_dataset} -M ${{INSTRUCT_MODEL}} ${{BASE_ARGS}} -CWF float \
-EV '[{get_ds(train_dataset, 0, "float", steer=True)}]'

python main.py {train_dataset} -M ${{BASE_MODEL}} ${{BASE_ARGS}} -CWF instruction \
-EV '[{get_ds(train_dataset, 0, "instruction", steer=True)}]'
python main.py {train_dataset} -M ${{BASE_MODEL}} ${{BASE_ARGS}} -CWF float \
-EV '[{get_ds(train_dataset, 0, "float", steer=True)}]'

## Default Models
python main.py {train_dataset} -NT -M ${{INSTRUCT_MODEL}} ${{BASE_ARGS}} \
-EV '[{get_ds(train_dataset, 0, "float", steer=True)},{get_ds(train_dataset, 10, "float", steer=True)},{get_ds(train_dataset, 0, "instruction", steer=True)},{get_ds(train_dataset, 10, "instruction", steer=True)}]'

python main.py {train_dataset} -NT -M ${{BASE_MODEL}} ${{BASE_ARGS}} \
-EV '[{get_ds(train_dataset, 0, "float", steer=True)},{get_ds(train_dataset, 10, "float", steer=True)},{get_ds(train_dataset, 0, "instruction", steer=True)},{get_ds(train_dataset, 10, "instruction", steer=True)}]'
"""
    if add_ood_datasets:
        out += f"""
### OOD DATASETS ###
## Trained Models
python main.py {train_dataset} -M ${{INSTRUCT_MODEL}} ${{BASE_ARGS}} -CWF instruction \
-EV '[{",".join([get_ds(ds, 0, "instruction", steer=False) for ds in eval_datasets])}]'

python main.py {train_dataset} -NT -M ${{INSTRUCT_MODEL}} ${{BASE_ARGS}} \
-EV '[{",".join([get_ds(ds, 0, "instruction", steer=False) for ds in eval_datasets])}]'

python main.py {train_dataset} -NT -M ${{INSTRUCT_MODEL}} ${{BASE_ARGS}} \
-EV '[{",".join([get_ds(ds, 10, "instruction", steer=False) for ds in eval_datasets])}]'

"""
    if add_instruct_generalisation:
        out += f"""
### GENERALIZATION ACROSS Instruction Formats ###

python main.py {train_dataset} -M ${{INSTRUCT_MODEL}} ${{BASE_ARGS}} -CWF instruction \
-EV '[{get_ds(train_dataset, 0, "float", steer=True)}]'

python main.py {train_dataset} -M ${{INSTRUCT_MODEL}} ${{BASE_ARGS}} -CWF float \
-EV '[{get_ds(train_dataset, 0, "instruction", steer=True)}]'
        """
    return out


def generalisation_script(dataset, model, seed):
    return f"""

"""

CONFIGS = {
    "llama": {
        "instruct_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "base_model": "meta-llama/Meta-Llama-3.1-8B",
        "bs": 8,
        "ga": 2,
        "projection_path": "jkminder/CTXPRIOR-Projection-Meta-Llama-3.1-8B-Instruct-L16",
        "prior_value": 6,
        "context_value": -6,
        "steering_layer": 16,
        "ebs": 8,
    },
    "mistral": {
        "instruct_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "base_model": "mistralai/Mistral-7B-v0.3",
        "bs": 8,
        "ga": 2,
        "projection_path": "jkminder/CTXPRIOR-Projection-Mistral-7B-Instruct-v0.3-L16",
        "prior_value": 5.0,
        "context_value": -5.0,
        "steering_layer": 16,
        "ebs": 8,
    },
    "gemma": {
        "instruct_model": "google/gemma-2-9b-it",
        "base_model": "google/gemma-2-9b",
        "bs": 2,
        "ebs": 4,
        "ga": 8,
        "projection_path": "jkminder/CTXPRIOR-Projection-gemma-2-9b-it-L27",
        "prior_value": -100.0,
        "context_value": 150.0,
        "steering_layer": 27,
    },
}


def generate_run_script(args):
    add_training = args.add_training if hasattr(args, 'add_training') else True
    add_default = args.add_default if hasattr(args, 'add_default') else True
    add_steering = args.add_steering if hasattr(args, 'add_steering') else True
    add_ood_datasets = args.add_ood_datasets if hasattr(args, 'add_ood_datasets') else True
    
    if not any([add_training, add_default, add_steering, add_ood_datasets]):
        add_training = add_default = add_steering = add_ood_datasets = True
    
    return get_base_script(args.dataset_name, seed=args.seed, in_domain=args.in_domain, 
                           add_training=add_training, add_default=add_default, 
                           add_steering=add_steering, add_ood_datasets=add_ood_datasets, 
                           **CONFIGS[args.model_id])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True, choices=["llama", "mistral", "gemma"])
    parser.add_argument("--dataset_name", type=str, default="BaseFakepedia")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--add-instruct-generalisation", action="store_true")
    parser.add_argument("--add-default", action="store_true")
    parser.add_argument("--add-steering", action="store_true")
    parser.add_argument("--add-ood-datasets", action="store_true")
    parser.add_argument("--add-training", action="store_true")
    parser.add_argument("--in-domain", action="store_true")
    parser.add_argument("--outfile", type=str, default="run_script.sh")
    args = parser.parse_args()
    script = generate_run_script(args)
    with open(args.outfile, "w") as f:
        f.write(script)
