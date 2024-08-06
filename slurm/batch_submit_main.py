import subprocess
import json
from typing import List, Dict


RUN_LOCALLY = False
dataset_names = ["BaseFakepedia"]  # , "MultihopFakepedia", "YagoLlama2"]
# evals = json.dumps(
#     [
#         "BaseFakepedia",
#         "MultihopFakepedia",
#         "YagoLlama2"
#     ],
#     separators=(",", ":")
# )
evals = json.dumps(
    [
        {
            "dataset_name": "BaseFakepedia",
            "k_demonstrations": 20,
            "context_weight_format": "float",
        },
        {
            "dataset_name": "BaseFakepedia",
            "k_demonstrations": 20,
            "context_weight_format": "instruction",
        },
        # {
        #     "dataset_name": "MultihopFakepedia",
        #     "k_demonstrations": 20,
        #     "context_weight_format": "float",
        # },
        # {
        #     "dataset_name": "MultihopFakepedia",
        #     "k_demonstrations": 20,
        #     "context_weight_format": "instruction",
        # },
        # {
        #     "dataset_name": "YagoLlama2",
        #     "k_demonstrations": 20,
        #     "context_weight_format": "float",
        # },
        # {
        #     "dataset_name": "YagoLlama2",
        #     "k_demonstrations": 20,
        #     "context_weight_format": "instruction",
        # },
    ],
    separators=(",", ":"),
)
print(evals)
subsplit_names = [
    "nodup_relpid",
    # "nodup_relpid_obj",
    # "nodup_relpid_subj",
    # "nodup_s_or_rel_or_obj",
    # "base",
]
seeds = [0]
train_sizes = [1200]
# no_train_statuses = [False]
no_train_statuses = [True]
peft_modules = [
    json.dumps(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], separators=(",", ":")),
    # json.dumps(["gate_proj", "up_proj", "down_proj"], separators=(",", ":")),
    # json.dumps(["q_proj", "k_proj", "v_proj", "o_proj"], separators=(",", ":")),
    # json.dumps(["q_proj", "k_proj", "v_proj"], separators=(",", ":")),
    # json.dumps(["q_proj", "k_proj"], separators=(",", ":")),
    # json.dumps(["o_proj", "v_proj"], separators=(",", ":")),
    # # json.dumps(["o_proj"], separators=(",", ":")),
    # # json.dumps(["v_proj"], separators=(",", ":")),
    # # json.dumps(["q_proj"], separators=(",", ":")),
    # # json.dumps(["k_proj"], separators=(",", ":")),
]
context_weight_formats = ["float", "instruction"]

if RUN_LOCALLY:
    model_id_and_bs_and_ga_and_quantize_and_peft_tuples = [
        ("unsloth/mistral-7b-v0.2-bnb-4bit", 4, 4, "4bit", True),
        # ("unsloth/gemma-2b-bnb-4bit", 4, 4, "4bit", True),
        # ("unsloth/gemma-7b-bnb-4bit", 4, 4, "4bit", True),
    ]
else:
    # FEWSHOT
    model_id_and_bs_and_ga_and_quantize_and_peft_tuples = [
        ("unsloth/mistral-7b-v0.2-bnb-4bit", 4, 4, "4bit", False),
        # ("unsloth/mistral-7b-instruct-v0.2-bnb-4bit", 4, 4, "4bit", True),
        # ("unsloth/llama-2-7b-bnb-4bit", 4, 4, "4bit", True),
        ("unsloth/llama-2-7b-chat-bnb-4bit", 4, 4, "4bit", False),
        # ("unsloth/llama-3-8b-bnb-4bit", 4, 4, "4bit", True),
        ("unsloth/llama-3-8b-Instruct-bnb-4bit", 4, 4, "4bit", False),
        # ("unsloth/gemma-2b-bnb-4bit", 4, 4, "4bit", True),
        ("unsloth/gemma-7b-bnb-4bit", 4, 4, "4bit", False),
        # ("unsloth/gemma-2b-it-bnb-4bit", 4, 4, "4bit", True),
        # ("unsloth/gemma-7b-it-bnb-4bit", 4, 4, "4bit", True),
    ]
    # FINETUNE
    # model_id_and_bs_and_ga_and_quantize_and_peft_tuples = [
    #     ("unsloth/mistral-7b-v0.2-bnb-4bit", 4, 4, "4bit", True),
    #     # ("unsloth/mistral-7b-instruct-v0.2-bnb-4bit", 4, 4, "4bit", True),
    #     # ("unsloth/llama-2-7b-bnb-4bit", 4, 4, "4bit", True),
    #     ("unsloth/llama-2-7b-chat-bnb-4bit", 4, 4, "4bit", True),
    #     # ("unsloth/llama-3-8b-bnb-4bit", 4, 4, "4bit", True),
    #     ("unsloth/llama-3-8b-Instruct-bnb-4bit", 4, 4, "4bit", True),
    #     # ("unsloth/gemma-2b-bnb-4bit", 4, 4, "4bit", True),
    #     ("unsloth/gemma-7b-bnb-4bit", 4, 4, "4bit", True),
    #     # ("unsloth/gemma-2b-it-bnb-4bit", 4, 4, "4bit", True),
    #     # ("unsloth/gemma-7b-it-bnb-4bit", 4, 4, "4bit", True),
    # ]

overwrite = False

job_count = 0
for ds in dataset_names:
    for sp in subsplit_names:
        for seed in seeds:
            for ts in train_sizes:
                for nts in no_train_statuses:
                    for pm in peft_modules:
                        for cwf in context_weight_formats:
                            for model_id, bs, ga, quantize, peft in model_id_and_bs_and_ga_and_quantize_and_peft_tuples:
                                job_count += 1
                                print(job_count)
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
                                            "-EV",
                                            f"{evals}",
                                            "-CWF",
                                            f"{cwf}",
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
                                            f"{evals}",
                                            f"{sp}",
                                            f"{cwf}",
                                        ]
                                        + (["-NT"] if nts else [])
                                        + (["-P"] if peft else [])
                                        + (["-F"] if quantize == "4bit" else [])
                                        + (["-E"] if quantize == "8bit" else [])
                                        + (["-O"] if overwrite else [])
                                    )
                                    print(cmd)
                                    subprocess.check_call(cmd)
