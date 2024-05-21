import argparse
from dotenv import load_dotenv
import gc
import random
import os
import sys
from tqdm import tqdm
from typing import Optional, List, Union, Dict, Tuple
import yaml

import numpy as np
import pandas as pd
import torch
import wandb

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from utils import construct_paths_and_dataset_kwargs, construct_artifact_name, format_prompts, PROMPTS_DICT
from preprocessing.dataset import BaseFakepedia, ContextQueryDataset


load_dotenv()
hf_token = os.environ.get("HF_TOKEN")


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for training a model with context weights.")
    parser.add_argument("DATASET_NAME", type=str, help="Name of the dataset class")
    parser.add_argument("-S", "--SEED", type=int, default=0, help="Random seed")
    parser.add_argument(
        "-M",
        "--MODEL_ID",
        type=str,
        default="unsloth/gemma-2b-bnb-4bit",
        help="Name of the model to use from huggingface",
    )
    parser.add_argument("-F", "--LOAD_IN_4BIT", action="store_true", help="Whether to load in 4 bit")
    parser.add_argument("-E", "--LOAD_IN_8BIT", action="store_true", help="Whether to load in 8 bit")
    parser.add_argument("-BS", "--BATCH_SIZE", type=int, default=32, help="Batch size for training")
    parser.add_argument("-MSL", "--MAX_SEQ_LENGTH", type=int, default=2048, help="Maximum sequence length for training")
    parser.add_argument(
        "-O",
        "--OVERWRITE",
        action="store_true",
        help="Whether to overwrite existing results and recompute susceptibility scores",
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    model_id: str, load_in_4bit: bool, load_in_8bit: bool, dtype: Optional[str] = "auto", device: str = "auto"
):
    """
    Load the model and tokenizer from huggingface.
    Args:
        model_id: str
        load_in_4bit: bool -  whether to use 4bit quantization to reduce memory usage.
            # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
            fourbit_models = [
                "unsloth/mistral-7b-bnb-4bit",
                "unsloth/mistral-7b-v0.2-bnb-4bit", # New Mistral 32K base model
                "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
                "unsloth/llama-2-7b-bnb-4bit",
                "unsloth/llama-2-13b-bnb-4bit",
                "unsloth/codellama-34b-bnb-4bit",
                "unsloth/tinyllama-bnb-4bit",
                "unsloth/gemma-7b-bnb-4bit", # New Google 6 trillion tokens model 2.5x faster!
                "unsloth/gemma-2b-bnb-4bit",
            ] # More models at https://huggingface.co/unsloth
        load_in_8bit: bool
        dtype: torch.dtype - default to None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        device: str - default to auto
    """
    if load_in_4bit and load_in_8bit:
        raise ValueError("Cannot load in both 4bit and 8bit.")

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif load_in_8bit:
        # TODO(kdu): untested
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # "microsoft/Phi-3-mini-4k-instruct",
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=dtype,
    )
    print(f"Loaded model on device {model.device} with dtype {model.dtype}.")

    tokenizer = prepare_tokenizer(model)

    torch.cuda.empty_cache()
    gc.collect()

    return model, tokenizer


def prepare_tokenizer(model):
    tokenizer = AutoTokenizer.from_pretrained(
        model.config._name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    tokenizer.add_special_tokens({"pad_token": "<|PAD|>"})
    tokenizer.pad_token = "<|PAD|>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|PAD|>")
    tokenizer.padding_side = "right"  # for kbit training apparently you need to pad on the right
    model.resize_token_embeddings(len(tokenizer))
    print(tokenizer)
    return tokenizer


def prepare_peft_model(model, **lora_config_kwargs):
    model.gradient_checkpointing_disable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # "fc1", "fc2",
        # "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.00,
        bias="none",
        task_type="CAUSAL_LM",
        **lora_config_kwargs,
    )  # TODO: replace with lora_config_kwargs and add to argparse

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def evaluate_model(
    model, tokenizer, dataset: Dataset, max_new_tokens: int = 30, batch_sz: int = 4, device: str = "auto"
):
    """
    Given a dataset with columns ["text", "labels"], generate answers and evaluate model accuracy against those labels.
    """
    pass


def main():
    args = get_args()
    DATASET_NAME = args.DATASET_NAME
    SEED = args.SEED
    MODEL_ID = args.MODEL_ID
    LOAD_IN_4BIT = args.LOAD_IN_4BIT
    LOAD_IN_8BIT = args.LOAD_IN_8BIT
    OVERWRITE = args.OVERWRITE

    # Model parameters
    BATCH_SZ = args.BATCH_SIZE
    MAX_SEQ_LENGTH = args.MAX_SEQ_LENGTH

    # wandb stuff
    PROJECT_NAME = "sftcontext"
    GROUP_NAME = None
    TAGS = []
    LOG_DATASETS = True

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
        SEED=SEED,
        MODEL_ID=MODEL_ID,
        LOAD_IN_4BIT=LOAD_IN_4BIT,
        LOAD_IN_8BIT=LOAD_IN_8BIT,
        BATCH_SZ=BATCH_SZ,
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
        mode="online",
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

    # Load the model
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, LOAD_IN_4BIT, LOAD_IN_8BIT)
    model = prepare_peft_model(model)

    # SFT Train
    prompt, response_template = PROMPTS_DICT[MODEL_ID]
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=model,
        # tokenizer = tokenizer,
        data_collator=collator,
        formatting_func=lambda x: format_prompts(
            x, eos_token=tokenizer.eos_token, prompt_template=prompt, do_eval=False
        ),
        train_dataset=dataset.train_data,
        # eval_dataset = dataset_valid,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            gradient_checkpointing=False,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=10,  # increase this.... this is a tiny number of steps that i used just for debugging.
            # num_train_epochs = 1,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=SEED,
            output_dir="outputs",
        ),
    )

    gc.collect()
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()

    trainer_stats = trainer.train()
    print(trainer_stats)
    trainer.save_model(os.path.join(model_dir, "model"))

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
