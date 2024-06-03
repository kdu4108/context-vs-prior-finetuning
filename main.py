import argparse
from dotenv import load_dotenv
import gc
import json
import os
import random
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
from peft import AutoPeftModelForCausalLM, prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftConfig

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
    parser.add_argument("-F", "--LOAD_IN_4BIT", action="store_true", help="Whether to load in 4 bit")
    parser.add_argument("-E", "--LOAD_IN_8BIT", action="store_true", help="Whether to load in 8 bit")
    parser.add_argument("-BS", "--BATCH_SIZE", type=int, default=4, help="Batch size for training (per device)")
    parser.add_argument("-GA", "--GRAD_ACCUM", type=int, default=4, help="Number of steps for gradient accumulation")
    parser.add_argument("-MSL", "--MAX_SEQ_LENGTH", type=int, default=2048, help="Maximum sequence length for training")
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
        "-O",
        "--OVERWRITE",
        action="store_true",
        help="Whether to overwrite existing results and retrain model",
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    model_id: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    train_mode: bool = True,
    peft_config: Optional[PeftConfig] = None,
    dtype: Optional[str] = "auto",
    device: str = "auto",
    attn_implementation: str = "sdpa",
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

    if peft_config is not None:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_id,
                is_trainable=train_mode,
                config=peft_config,
                quantization_config=bnb_config,
                device_map=device,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
            )
            tokenizer = prepare_tokenizer(model)
        except ValueError:
            print("Failed to load model with AutoPeftModelForCausalLM, now attempting with AutoModelForCausalLM.")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=device,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
            )
            tokenizer = prepare_tokenizer(model)
            if train_mode:
                # If we are not training the model, we do not want to load it in peft mode
                model = prepare_peft_model(model, peft_config=peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        tokenizer = prepare_tokenizer(model)
    print(f"Loaded model on device {model.device} with dtype {model.dtype}.")

    torch.cuda.empty_cache()
    gc.collect()

    return model, tokenizer


def prepare_tokenizer(model, add_pad_token=False):
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    # if "mistral" in model.config._name_or_path.lower():
    #     tokenizer.add_special_tokens({"pad_token": "<|PAD|>"})
    #     tokenizer.pad_token = "<|PAD|>"
    #     tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|PAD|>")
    #     model.resize_token_embeddings(len(tokenizer))
    #     # tokenizer.pad_token = tokenizer.eos_token
    #     # print("Setting pad token to EOS")

    # if add_pad_token:
    #     tokenizer.add_special_tokens({"pad_token": "<|PAD|>"})
    #     tokenizer.pad_token = "<|PAD|>"
    #     tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|PAD|>")
    #     model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "right"  # for kbit training apparently you need to pad on the right
    return tokenizer


def prepare_peft_model(
    model, peft_config, target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"], **lora_config_kwargs
):
    """
    Args:
        target modules - subset of ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2", "gate_proj", "up_proj", "down_proj"]
    """
    model.gradient_checkpointing_disable()
    model = prepare_model_for_kbit_training(model)  # model becomes float32 instead of bfloat16
    # peft_config = LoraConfig(
    #     r=64,
    #     lora_alpha=16,
    #     target_modules=target_modules,
    #     lora_dropout=0.00,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     **lora_config_kwargs,
    # )  # TODO: replace with lora_config_kwargs and add to argparse

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def is_response_correct(response: str, label: str) -> bool:
    return response.startswith(label)


def evaluate_model(
    model,
    tokenizer,
    dataset: Dataset,
    max_new_tokens: int = 30,
    batch_sz: int = 4,
    device: str = "auto",
):
    """
    Given a dataset with columns ["text", "labels"], generate answers and evaluate model accuracy against those labels.
    1. Generate predictions from text
    2. Extract answer, compare to labels, and return accuracy
    """
    tokenizer.padding_side = "left"
    encoded_dataset = dataset.map(
        lambda examples: tokenizer(examples["text"], padding=True, return_tensors="pt"),
        batched=True,
    ).select_columns(["input_ids", "attention_mask", "labels"])
    encoded_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"], device="cuda"
    )  # required for loading correctly into dataloader
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_sz)
    predictions, labels, is_correct_all = [], [], []
    num_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            init_seq_len = batch["input_ids"].shape[1]
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )
            responses_only = outputs[:, init_seq_len:]
            decoded_responses = tokenizer.batch_decode(responses_only)
            decoded_responses = [r.strip() for r in decoded_responses]
            is_correct = [
                is_response_correct(response, label) for response, label in zip(decoded_responses, batch["labels"])
            ]

            num_correct += sum(is_correct)
            total += len(batch["labels"])
            predictions += decoded_responses
            is_correct_all += is_correct
            labels += batch["labels"]

            print(f"Average accuracy at batch {i}: {num_correct/total} ({num_correct}/{total}).")

    dataset = dataset.map(
        lambda examples: {
            "predictions": predictions,
            "is_correct": is_correct_all,
        },
        batched=True,
    )
    metrics = {"acc": num_correct / total}

    return dataset, metrics


def main():
    args = get_args()
    DATASET_NAME = args.DATASET_NAME
    SEED = args.SEED
    TRAIN_SIZE = args.TRAIN_SIZE
    MODEL_ID = args.MODEL_ID
    PEFT = args.PEFT
    LORA_MODULES = args.LORA_MODULES
    LOAD_IN_4BIT = args.LOAD_IN_4BIT
    LOAD_IN_8BIT = args.LOAD_IN_8BIT
    NO_EVAL = args.NO_EVAL
    NO_TRAIN = args.NO_TRAIN
    OVERWRITE = args.OVERWRITE

    # Model parameters
    BATCH_SZ = args.BATCH_SIZE
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

    # Mech itnerp:
    # (4096, 128, 32)
    # # reshape lora_A @ lora_B layer to (hs, attn_hs, num_heads),
    # then get the norm across the 0 and 1 direction to get the norm for each of the (num_heads,)
    # this norm across 0 and 1 direction is asking: for the W_q for each head, how much did that change?
    # this reshaping operation will be implementation/model-family specific
    # maybe transformerlens can already handle this for us?

    # Way to test our reshaping is: maybe try using einsum and skip the reshaping to make sure we get the ssame result
    # here is how they reshape: https://github.com/huggingface/transformers/blob/9837a25481e1e381753119c1676289e8358d91af/src/transformers/models/llama/modeling_llama.py#L331
    # Can also sanity check against Meta's LLM Transparency tool
    # possibly can use this as a reference https://colab.research.google.com/drive/1bZkkJd8pAVnSN23svyszZ3f4WrnYKN_3?usp=sharing#scrollTo=wNOKYkvom4R_, also this https://arena3-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch
    # Load prompt template for chosen model
    train_mode = not NO_TRAIN
    prompt, response_template = PROMPTS_DICT[MODEL_ID]
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
        )
        if NO_TRAIN:
            print("Skipping training loop.")
        else:
            # if PEFT:
            #     model = prepare_peft_model(model, target_modules=LORA_MODULES)

            # SFT Train
            collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
            trainer = SFTTrainer(
                model=model,
                # tokenizer = tokenizer,
                data_collator=collator,
                formatting_func=lambda x: format_prompts(
                    x, eos_token=tokenizer.eos_token, prompt_template=prompt, do_eval=False
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
                    save_steps=10,
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

    # Evaluate
    if not NO_EVAL:
        test_dataset = dataset.test_data.map(
            lambda examples: {
                "text": format_prompts(
                    examples=examples, eos_token=tokenizer.eos_token, prompt_template=prompt, do_eval=True
                ),
                "labels": examples["answer"],
            },
            batched=True,
        )
        eval_results, eval_metrics = evaluate_model(
            model=model, tokenizer=tokenizer, dataset=test_dataset.select(range(100))
        )

        # Save results
        eval_results.to_csv(val_results_path, index=False)
        with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as fp:
            json.dump(eval_metrics, fp, ensure_ascii=False, indent=4, sort_keys=True)

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
