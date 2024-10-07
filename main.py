# python main.py BaseFakepedia -M meta-llama/Meta-Llama-3.1-8B-Instruct -S 3 -TS 2048 -TSS 1000 -P -BS 8 -GA 2 -CWF float -O
import argparse
from dotenv import load_dotenv
import gc
import json
import os
import random
import sys
from tqdm import tqdm
from typing import Optional, List, Union, Dict, Tuple, Set
import yaml

import numpy as np
import pandas as pd
import torch
import wandb

from transformers import TrainingArguments
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

from model_utils.utils import (
    construct_paths_and_dataset_kwargs,
    construct_artifact_name,
    create_pscore_format_func,
    format_prompts,
    MODEL_ID_TO_TEMPLATES_DICT,
    evaluate_model,
    evaluate_model_queries_only,
    evaluate_model_pscores,
    load_model_and_tokenizer,
    get_raw_data_dir,
    compute_metrics,
    compute_metrics_only_og_correct,
    construct_test_results_dir,
    EvalConfig,
    sample_few_shot_examples,
)

from preprocessing.dataset import Arithmetic, BaseFakepedia, MultihopFakepedia, ContextQueryDataset, Yago, YagoLlama2


load_dotenv()
hf_token = os.environ.get("HF_TOKEN")


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for training a model with context weights.")
    parser.add_argument("DATASET_NAME", type=str, help="Name of the dataset class")
    parser.add_argument(
        "-SP",
        "--SUBSPLIT",
        type=str,
        default="nodup_relpid",
        choices=[
            "nodup_relpid",
            "nodup_relpid_obj",
            "nodup_relpid_subj",
            "nodup_s_or_rel_or_obj",
            "base",
        ],
        help="Name of the dataset subsplit to use.",
    )
    # Options: nodup_relpid, nodup_relpid_obj, nodup_relpid_subj, nodup_s_or_rel_or_obj, base
    parser.add_argument("-S", "--SEED", type=int, default=0, help="Random seed")
    parser.add_argument(
        "-M",
        "--MODEL_ID",
        type=str,
        # default="unsloth/gemma-2b-bnb-4bit",
        default="unsloth/gemma-7b-bnb-4bit",
        help="Name of the model to use from huggingface",
    )
    parser.add_argument("-P", "--PEFT", action="store_true", help="Whether to train with PEFT")
    parser.add_argument(
        "-LM",
        "--LORA_MODULES",
        type=json.loads,
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Which modules to train with LoRA",
    )
    parser.add_argument("-TS", "--TRAIN_SIZE", type=int, default=320, help="Number of train examples")
    parser.add_argument("-TSS", "--TEST_SIZE", type=int, default=100, help="Number of test examples")
    parser.add_argument("-F", "--LOAD_IN_4BIT", action="store_true", help="Whether to load in 4 bit")
    parser.add_argument("-E", "--LOAD_IN_8BIT", action="store_true", help="Whether to load in 8 bit")
    parser.add_argument("-BS", "--BATCH_SIZE", type=int, default=4, help="Batch size for training (per device)")
    parser.add_argument("-EBS", "--EVAL_BATCH_SIZE", type=int, default=8, help="Batch size for evaluation (per device)")
    parser.add_argument("-GA", "--GRAD_ACCUM", type=int, default=4, help="Number of steps for gradient accumulation")
    parser.add_argument("-MSL", "--MAX_SEQ_LENGTH", type=int, default=2048, help="Maximum sequence length for training")
    parser.add_argument(
        "-CWE",
        "--CONTEXT_WEIGHTS_END",
        action="store_true",
        help="Whether to have the context weight flag at the very end of the prompt",
    )
    parser.add_argument(
        "-CWF",
        "--CONTEXT_WEIGHT_FORMAT",
        type=str,
        default="instruction",
        choices=[
            "float",
            "instruction",
        ],
        help="Name of the format of specifying the context weights.",
    )
    parser.add_argument(
        "-EV",
        "--EXTRA_EVALS",
        type=json.loads,
        # default=[],
        default=[
            {
                "dataset_name": "Arithmetic",
                "subsplit": "base",
                "k_demonstrations": 0,
                "context_weight_format": "instruction",
            }
        ],
        help="Datasets on which to run evals. Expected format: a List of Dicts containing {'dataset_name': str, 'k_demonstrations': int, 'context_weight_format': str}",
    )
    parser.add_argument(
        "-NT",
        "--NO-TRAIN",
        action="store_true",
        help="Whether to train the model",
    )
    parser.add_argument(
        "-NE",
        "--NO-EVAL",
        action="store_true",
        help="Whether to evaluate on test set",
    )
    parser.add_argument(
        "-PE",
        "--DO-PSCORE-EVAL",
        action="store_false",
        help="Whether to evaluate on test set with pscores",
    )
    parser.add_argument(
        "-ID",
        "--ICL-IN-DOMAIN",
        action="store_true",
        help="Whether to evaluate on test set with pscores",
    )
    parser.add_argument(
        "-O",
        "--OVERWRITE",
        action="store_true",
        help="Whether to overwrite existing results and retrain model",
    )
    return parser.parse_args()


def main():
    args = get_args()
    DATASET_NAME = args.DATASET_NAME
    SUBSPLIT = args.SUBSPLIT
    SEED = args.SEED
    TRAIN_SIZE = args.TRAIN_SIZE
    TEST_SIZE = args.TEST_SIZE
    MODEL_ID = args.MODEL_ID
    PEFT = args.PEFT
    LORA_MODULES = args.LORA_MODULES
    LOAD_IN_4BIT = args.LOAD_IN_4BIT
    LOAD_IN_8BIT = args.LOAD_IN_8BIT
    EXTRA_EVALS = args.EXTRA_EVALS
    ICL_IN_DOMAIN = args.ICL_IN_DOMAIN
    NO_EVAL = args.NO_EVAL
    DO_PSCORE_EVAL = args.DO_PSCORE_EVAL
    NO_TRAIN = args.NO_TRAIN
    OVERWRITE = args.OVERWRITE
    CONTEXT_WEIGHT_AT_END = args.CONTEXT_WEIGHTS_END
    CONTEXT_WEIGHT_FORMAT = args.CONTEXT_WEIGHT_FORMAT

    # Model parameters
    BATCH_SZ = args.BATCH_SIZE
    EVAL_BATCH_SZ = args.EVAL_BATCH_SIZE
    GRAD_ACCUM = args.GRAD_ACCUM
    MAX_SEQ_LENGTH = args.MAX_SEQ_LENGTH

    # wandb stuff
    PROJECT_NAME = "sftcontext"
    GROUP_NAME = None
    TAGS = []
    LOG_DATASETS = False

    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Construct paths from run parameters and construct DATASET_KWARGS_IDENTIFIABLE
    (
        data_dir,
        input_dir,
        model_dir,
        results_dir,
        val_results_path,
        data_id,
        model_id,
        DATASET_KWARGS_IDENTIFIABLE,
        MODEL_KWARGS_IDENTIFIABLE,
    ) = construct_paths_and_dataset_kwargs(
        DATASET_NAME=DATASET_NAME,
        SUBSPLIT=SUBSPLIT,
        SEED=SEED,
        TRAIN_SIZE=TRAIN_SIZE,
        MODEL_ID=MODEL_ID,
        PEFT=PEFT,
        LORA_MODULES=LORA_MODULES,
        LOAD_IN_4BIT=LOAD_IN_4BIT,
        LOAD_IN_8BIT=LOAD_IN_8BIT,
        BATCH_SZ=BATCH_SZ,
        GRAD_ACCUM=GRAD_ACCUM,
        NO_TRAIN=NO_TRAIN,
        CONTEXT_WEIGHT_AT_END=CONTEXT_WEIGHT_AT_END,
        CONTEXT_WEIGHT_FORMAT=CONTEXT_WEIGHT_FORMAT,
        OVERWRITE=OVERWRITE,
        verbose=True,
    )
    # # GPU stuff
    # device = "auto"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")

    # wandb stuff
    params_to_log = {k: v for k, v in locals().items() if k.isupper()}

    run = wandb.init(
        project=PROJECT_NAME,
        group=GROUP_NAME,
        config=params_to_log,
        tags=TAGS,
        mode="disabled",
    )
    print(dict(wandb.config))

    dataset: ContextQueryDataset = getattr(sys.modules[__name__], DATASET_NAME)(**DATASET_KWARGS_IDENTIFIABLE)

    # After loading/preprocessing your dataset, log it as an artifact to W&B
    print(f"Saving datasets and run config to {input_dir}.")
    os.makedirs(input_dir, exist_ok=True)
    dataset.train_data.to_csv(os.path.join(input_dir, "train.csv"))
    dataset.val_data.to_csv(os.path.join(input_dir, "val.csv"))
    dataset.test_data.to_csv(os.path.join(input_dir, "test.csv"))
    with open(os.path.join(input_dir, "config.yml"), "w") as yaml_file:
        yaml.dump({**DATASET_KWARGS_IDENTIFIABLE, **MODEL_KWARGS_IDENTIFIABLE}, yaml_file, default_flow_style=False)

    # Load prompt template for chosen model
    train_mode = not NO_TRAIN

    # Check if local model
    if os.path.exists(MODEL_ID):
        model_id = os.path.basename(MODEL_ID)
    else:
        model_id = MODEL_ID

    prompt_template_dict, response_template = MODEL_ID_TO_TEMPLATES_DICT[model_id]
    peft_config = (
        LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=LORA_MODULES,
            lora_dropout=0.00,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if PEFT
        else None
    )

    # Load the model
    if not OVERWRITE and (
        os.path.isfile(os.path.join(model_dir, "config.json"))
        or os.path.isfile(os.path.join(model_dir, "adapter_config.json"))
    ):
        # Model has already been trained
        print(f"Model already saved at {model_dir}, attempting to load.")
        model, tokenizer = load_model_and_tokenizer(
            model_id=model_dir,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
            peft_config=peft_config,
            train_mode=train_mode,
            attn_implementation="eager"
            if "gemma" in model_id.lower()
            else "sdpa",  # it is recommended to use eager for gemma models
        )
        print(f"Loaded pretrained model from {model_dir}")
    else:
        print(f"Loading model {MODEL_ID} from huggingface.")
        # Cannot load model with PeftConfig if in training mode
        model, tokenizer = load_model_and_tokenizer(
            model_id=MODEL_ID,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
            peft_config=peft_config,
            train_mode=train_mode,
            attn_implementation="eager"
            if "gemma" in model_id.lower()
            else "sdpa",  # it is recommended to use eager for gemma models
        )
        if NO_TRAIN:
            print("Skipping training loop.")
        else:
            # SFT Train
            if response_template.startswith("\n"):
                # https://huggingface.co/docs/trl/v0.7.2/en/sft_trainer#using-tokenids-directly-for-responsetemplate
                # adding a \n to the start of the response template will result in a different tokenization for the first token (otherwise the first token is tokenized differently): Edit JM: Not true, same tokenization, we need to remove <eot_id> and \n though (so 2 now)
                response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[
                    2:
                ]  # to remove \n and somehow <eot_id>, JM: I don't understand why this is necessary, but it is for Llama3
            else:
                response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
            collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
            trainer = SFTTrainer(
                model=model,
                # tokenizer = tokenizer,
                data_collator=collator,
                formatting_func=lambda x: format_prompts(
                    x,
                    eos_token=tokenizer.eos_token,
                    prompt_template_dict=prompt_template_dict,
                    demonstrations_context_weight_format=None,
                    query_context_weight_format=CONTEXT_WEIGHT_FORMAT,
                    context_weight_at_end=CONTEXT_WEIGHT_AT_END,
                    demonstrations_df=pd.DataFrame(),
                    do_eval=False,
                ),
                train_dataset=dataset.train_data,
                # eval_dataset=dataset.val_data.select(100),
                max_seq_length=MAX_SEQ_LENGTH,
                dataset_num_proc=2,
                packing=False,  # Can make training 5x faster for short sequences.
                args=TrainingArguments(
                    output_dir=model_dir,
                    gradient_checkpointing=False,
                    per_device_train_batch_size=BATCH_SZ,
                    gradient_accumulation_steps=GRAD_ACCUM,
                    warmup_steps=5,
                    # max_steps=10,
                    num_train_epochs=1,
                    save_strategy="no",
                    # save_steps=10,
                    learning_rate=2e-4,
                    fp16=not torch.cuda.is_bf16_supported(),
                    bf16=torch.cuda.is_bf16_supported(),
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="linear",
                    seed=SEED,
                ),
                # peft_config=peft_config if PEFT else None, # for some reason this uses more memory and OOMs when the existing way does not? (but also appears to be faster?)
            )

            gc.collect()
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()

            print("Preparing to train model.")
            trainer_stats = trainer.train()
            print("Trainer stats:", trainer_stats)
            trainer.save_model(model_dir)
            print(f"Model saved to {model_dir}")

    # Evaluate
    if not NO_EVAL:
        # Set padding_side to left for all evals
        tokenizer.padding_side = "left"

        # Construct full list of eval configs
        # evals: List[EvalConfig] = [
        #     EvalConfig(
        #         dataset_name=DATASET_NAME,
        #         k_demonstrations=0,
        #         context_weight_format=CONTEXT_WEIGHT_FORMAT,
        #     )
        # ] + [EvalConfig(**eval) for eval in EXTRA_EVALS]
        evals: List[EvalConfig] = [EvalConfig(**eval) for eval in EXTRA_EVALS]
        # print(evals)
        for eval_name, eval_subsplit, eval_k_demonstrations, eval_ctx_weight_format in evals:
            print(
                f"Evaluating model on test split of {eval_name} using {eval_k_demonstrations} few shot examples from {DATASET_NAME} and with context weight format of `{eval_ctx_weight_format}`."
            )
            ds_class: ContextQueryDataset = getattr(sys.modules[__name__], eval_name)()

            # Collect data for few shot example demonstrations
            few_shot_examples_path = os.path.join(
                get_raw_data_dir(
                    dataset_name=eval_name if ICL_IN_DOMAIN else DATASET_NAME,
                    subsplit=eval_subsplit if ICL_IN_DOMAIN else SUBSPLIT,
                ),
                "train.csv",
            )
            few_shot_examples_df = pd.read_csv(few_shot_examples_path)
            few_shot_examples_sampled_df = sample_few_shot_examples(
                few_shot_examples_df, k=eval_k_demonstrations, seed=SEED
            )
            test_dataset_path = os.path.join(
                get_raw_data_dir(dataset_name=eval_name, subsplit=eval_subsplit), "test.csv"
            )
            # import pdb; pdb.set_trace()
            test_dataset = pd.read_csv(test_dataset_path, dtype={"answer": str, "prior_answer": str, "ctx_answer": str})
            test_dataset = Dataset.from_pandas(test_dataset)
            test_dataset = test_dataset.map(
                lambda examples: {
                    "text": format_prompts(
                        examples=examples,
                        eos_token=tokenizer.eos_token,
                        prompt_template_dict=prompt_template_dict,
                        demonstrations_context_weight_format=CONTEXT_WEIGHT_FORMAT,
                        query_context_weight_format=eval_ctx_weight_format,
                        context_weight_at_end=CONTEXT_WEIGHT_AT_END,
                        demonstrations_df=few_shot_examples_sampled_df,
                        do_eval=True,
                    ),
                    "labels": examples["answer"],
                },
                batched=True,
            )
            subsampled_test_dataset = test_dataset.select(range(min(TEST_SIZE, len(test_dataset))))
            eval_results = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                dataset=subsampled_test_dataset,
                batch_sz=EVAL_BATCH_SZ,
                is_response_correct_func=ds_class.is_response_correct,
                # batch_sz=8 if eval_k_demonstrations == 0 else 1,
            )
            query_to_is_correct, query_to_prediction = evaluate_model_queries_only(
                model=model,
                tokenizer=tokenizer,
                dataset=subsampled_test_dataset,
                is_response_correct_func=ds_class.is_response_correct,
            )
            eval_results = eval_results.map(
                lambda row: {
                    "query_only_prediction": query_to_prediction[row["query"]],
                    "query_only_is_correct": query_to_is_correct[row["query"]],
                }
            )
            if DO_PSCORE_EVAL:
                pscore_format_func = create_pscore_format_func(
                    prompt_template_dict=prompt_template_dict,
                    eos_token=tokenizer.eos_token,
                    demonstrations_df=few_shot_examples_sampled_df,
                    demonstrations_context_weight_format=CONTEXT_WEIGHT_FORMAT,
                    query_context_weight_format=eval_ctx_weight_format,
                    context_weight_at_end=CONTEXT_WEIGHT_AT_END,
                )
                p_score_results = evaluate_model_pscores(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=subsampled_test_dataset,
                    format_func=pscore_format_func,
                    batch_sz=4 if eval_k_demonstrations == 0 else 2,
                )

            eval_metrics = compute_metrics(eval_results.to_pandas())
            query_only_eval_metrics = compute_metrics_only_og_correct(eval_results.to_pandas())

            # Save results
            test_results_dir = construct_test_results_dir(
                results_dir,
                eval_name=eval_name,
                subsplit=eval_subsplit,
                k_demonstrations=eval_k_demonstrations,
                context_weight_format=eval_ctx_weight_format,
            )
            os.makedirs(test_results_dir, exist_ok=True)

            test_results_path = os.path.join(test_results_dir, "test.csv")
            test_metrics_path = os.path.join(test_results_dir, "metrics.json")
            test_metrics_query_only_path = os.path.join(test_results_dir, "metrics_query_only.json")
            test_results_pscore_path = os.path.join(test_results_dir, "test_pscore.csv")

            if eval_k_demonstrations > 0:
                few_shot_examples_sampled_df.to_csv(
                    os.path.join(test_results_dir, "few_shot_examples.csv"), index=False
                )

            print(f"Saving eval results to {test_results_path}")
            eval_results.to_csv(test_results_path, index=False)
            if DO_PSCORE_EVAL:
                p_score_results.to_csv(test_results_pscore_path, index=False)

            with open(test_metrics_path, "w", encoding="utf-8") as fp:
                json.dump(eval_metrics, fp, ensure_ascii=False, indent=4, sort_keys=True)
            with open(test_metrics_query_only_path, "w", encoding="utf-8") as fp:
                json.dump(query_only_eval_metrics, fp, ensure_ascii=False, indent=4, sort_keys=True)

    # After loading/preprocessing your dataset, log it as an artifact to W&B
    if LOG_DATASETS:
        print(f"Logging results to w&b run {wandb.run}.")
        artifact_name = construct_artifact_name(data_id, SEED, model_id)
        artifact = wandb.Artifact(name=artifact_name, type="results")
        artifact.add_dir(local_path=results_dir)
        run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
