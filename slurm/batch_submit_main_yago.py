import subprocess
import json
from typing import List, Dict


RUN_LOCALLY = False
dataset_names = ["Yago"]
subsplit_names = [
    "nodup_relpid",
    # "nodup_relpid_obj",
    # "nodup_relpid_subj",
    # "nodup_s_or_rel_or_obj",
    # "base",
]
seeds = [0]
train_sizes = [640]
no_train_statuses = [False]
peft_modules = [
    json.dumps(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], separators=(",", ":")),
    json.dumps(["gate_proj", "up_proj", "down_proj"], separators=(",", ":")),
    json.dumps(["q_proj", "k_proj", "v_proj", "o_proj"], separators=(",", ":")),
    json.dumps(["q_proj", "k_proj", "v_proj"], separators=(",", ":")),
    json.dumps(["q_proj", "k_proj"], separators=(",", ":")),
    json.dumps(["o_proj", "v_proj"], separators=(",", ":")),
    json.dumps(["o_proj"], separators=(",", ":")),
    json.dumps(["v_proj"], separators=(",", ":")),
]

if RUN_LOCALLY:
    model_id_and_bs_and_ga_and_quantize_and_peft_tuples = [
        ("unsloth/mistral-7b-v0.2-bnb-4bit", 4, 4, "4bit", True),
        # ("unsloth/gemma-2b-bnb-4bit", 4, 4, "4bit", True),
        # ("unsloth/gemma-7b-bnb-4bit", 4, 4, "4bit", True),
    ]
else:
    model_id_and_bs_and_ga_and_quantize_and_peft_tuples = [
        # ("unsloth/mistral-7b-v0.2-bnb-4bit", 4, 4, "4bit", True),
        # # ("unsloth/mistral-7b-instruct-v0.2-bnb-4bit", 4, 4, "4bit", True),
        # # ("unsloth/llama-2-7b-bnb-4bit", 4, 4, "4bit", True),
        ("unsloth/llama-2-7b-chat-bnb-4bit", 4, 4, "4bit", True),
        # ("unsloth/llama-3-8b-bnb-4bit", 4, 4, "4bit", True),
        # ("unsloth/llama-3-8b-Instruct-bnb-4bit", 4, 4, "4bit", True),
        # # ("unsloth/gemma-2b-bnb-4bit", 4, 4, "4bit", True),
        # ("unsloth/gemma-7b-bnb-4bit", 4, 4, "4bit", True),
        # # ("unsloth/gemma-2b-it-bnb-4bit", 4, 4, "4bit", True),
        # ("unsloth/gemma-7b-it-bnb-4bit", 4, 4, "4bit", True),
    ]

overwrite = False

for ds in dataset_names:
    for sp in subsplit_names:
        for seed in seeds:
            for ts in train_sizes:
                for nts in no_train_statuses:
                    for pm in peft_modules:
                        for model_id, bs, ga, quantize, peft in model_id_and_bs_and_ga_and_quantize_and_peft_tuples:
                            if RUN_LOCALLY:
                                subprocess.run(
                                    [
                                        "python",
                                        "main.py",
                                        f"{ds}",
                                        "-S",
                                        f"{seed}",
                                        "-TS",
                                        f"{ts}",
                                        "-M",
                                        f"{model_id}",
                                        "-LM",
                                        f"{pm}",
                                        "-BS",
                                        f"{bs}",
                                        "-GA",
                                        f"{ga}",
                                        "-SP",
                                        f"{sp}",
                                    ]
                                    + (["-NT"] if nts else [])
                                    + (["-P"] if peft else [])
                                    + (["-F"] if quantize == "4bit" else [])
                                    + (["-E"] if quantize == "8bit" else [])
                                    + (["-O"] if overwrite else [])
                                )
                            else:
                                cmd = (
                                    [
                                        "sbatch",
                                        "slurm/submit_main.cluster",
                                        f"{ds}",
                                        f"{seed}",
                                        f"{model_id}",
                                        f"{ts}",
                                        f"{bs}",
                                        f"{ga}",
                                        f"{pm}",
                                        f"{sp}",
                                    ]
                                    + (["-NT"] if nts else [])
                                    + (["-P"] if peft else [])
                                    + (["-F"] if quantize == "4bit" else [])
                                    + (["-E"] if quantize == "8bit" else [])
                                    + (["-O"] if overwrite else [])
                                )
                                print(cmd)
                                subprocess.check_call(cmd)
