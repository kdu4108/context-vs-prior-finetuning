import os
import gc
import hashlib
import subprocess as sp
from typing import Optional, List, Union, Dict, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments


#################
# MODEL LOADING #
#################
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
    tokenizer.pad_token = tokenizer.eos_token
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

def merge_save_peft(peft_model, tokenizer, path):
    """ Merge the peft model and save to path."""

    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    tokenizer.padding_side = "left"

    return merged_model, tokenizer


##############
# EVALUATION #
##############
def is_response_correct(response: str, label: str) -> bool:
    return label in response


def compute_mr(df) -> Tuple[float, float, float]:
    """
    Given a df with columns `predictions`,  `prior_answer`, and `ctx_answer`, return a tuple containing (MR, % of other answers).
    MR = (# prior) / (# prior + # context)
    CR = (# ctx) / (# prior + # context)
    % other answers = (# other) / (# prior + # context + # other)

    Note that MR and CR are not inverses
    """
    num_prior_answers = df.apply(lambda row: is_response_correct(row["predictions"], row["prior_answer"]), axis=1).sum()
    num_ctx_answers = df.apply(lambda row: is_response_correct(row["predictions"], row["ctx_answer"]), axis=1).sum()
    num_other_answers = len(df) - (num_ctx_answers + num_prior_answers)
    return num_prior_answers / (num_prior_answers + num_ctx_answers), num_other_answers / len(df)


def compute_metrics(df):
    ctx_pref_df = df[df["weight_context"] == 1.0]
    prior_pref_df = df[df["weight_context"] == 0.0]

    context_acc = ctx_pref_df["is_correct"].mean()
    prior_acc = prior_pref_df["is_correct"].mean()

    context_mr, context_other = compute_mr(ctx_pref_df)
    prior_mr, prior_other = compute_mr(prior_pref_df)

    overall_mr, overall_other = compute_mr(df)

    metrics = {
        "acc": df["is_correct"].mean(),  # Overall accuracy
        "context_acc": context_acc,  # accuracy across the examples that SHOULD follow the context
        "prior_acc": prior_acc,  # accuracy across the examples that SHOULD follow the prior
        "context_mr": context_mr,  # MR across the examples that SHOULD follow the context (we want this to be low)
        "prior_mr": prior_mr,  # MR across the examples that SHOULD follow the context (we want this to be high)
        "overall_mr": overall_mr,  # MR across all examples (we want this to be 50%)
        "context_pct_other": context_other,  # percent of examples featured a non-context or prior answer across examples that SHOULD follow the context (lower better)
        "prior_pct_other": prior_other,  # percent of examples that featured a non-context or prior answer across examples that SHOULD follow the prior (lower better)
        "overall_pct_other": overall_other,  # percent of examples that featured a non-context or prior answer across all examples (lower better)
        "query_only_acc": df["query_only_is_correct"].mean(),
    }

    return metrics


def compute_metrics_only_og_correct(df):
    metrics = compute_metrics(df[df["query_only_is_correct"] == True])  # noqa
    del metrics["query_only_acc"]
    return metrics


def evaluate_model_queries_only(
    model,
    tokenizer,
    dataset: Dataset,
    max_new_tokens: int = 30,
    batch_sz: int = 8,  # "auto",
    device: str = "auto",
):
    """
    Given a dataset with columns ["query", "prior_answer"], generate answers and evaluate model accuracy against those labels.
    1. Generate predictions from text
    2. Extract answer, compare to labels, and return accuracy
    """
    # Free gpu memory
    gc.collect()
    torch.cuda.empty_cache()
    if batch_sz == "auto":
        batch_sz = int(2 * int(sum(get_gpu_memory()) / 1000))
        print(f"Setting batch size to {batch_sz} for eval.")

    tokenizer.padding_side = "left"
    queries_only_dataset = Dataset.from_pandas(
        dataset.to_pandas()[["query", "prior_answer"]].drop_duplicates(), preserve_index=False
    )
    queries_only_dataset = queries_only_dataset.rename_column(
        "prior_answer", "labels"
    )  # need to make the labels column

    encoded_dataset = queries_only_dataset.map(
        lambda examples: tokenizer(examples["query"], padding=True, return_tensors="pt"),
        batched=True,
        batch_size=batch_sz,
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

            print(f"Average accuracy at batch {i} (query-only): {num_correct/total} ({num_correct}/{total}).")

    queries_only_dataset = queries_only_dataset.map(
        lambda examples: {
            "predictions": predictions,
            "is_correct": is_correct_all,
        },
        batched=True,  # need to set this so that it sets the predictions column to be one element per row from the list
        batch_size=len(
            queries_only_dataset
        ),  # need to set this so that it doesn't have shape mismatch errors in the length of the column.
    )
    queries_only_df = queries_only_dataset.to_pandas()
    query_to_is_correct = dict(zip(queries_only_df["query"], queries_only_df["is_correct"]))
    query_to_prediction = dict(zip(queries_only_df["query"], queries_only_df["predictions"]))

    return query_to_is_correct, query_to_prediction


def evaluate_model(
    model,
    tokenizer,
    dataset: Dataset,
    max_new_tokens: int = 30,
    batch_sz: int = 8,  # "auto",
    device: str = "auto",
):
    """
    Given a dataset with columns ["text", "labels"], generate answers and evaluate model accuracy against those labels.
    1. Generate predictions from text
    2. Extract answer, compare to labels, and return accuracy
    """
    # Free gpu memory
    gc.collect()
    torch.cuda.empty_cache()

    if batch_sz == "auto":
        batch_sz = int(2 * int(sum(get_gpu_memory()) / 1000))
        print(f"Setting batch size to {batch_sz} for eval.")
    tokenizer.padding_side = "left"
    encoded_dataset = dataset.map(
        lambda examples: tokenizer(examples["text"], padding=True, return_tensors="pt", add_special_tokens=False),
        batched=True,
        batch_size=batch_sz,
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
        batched=True,  # need to set this so that it sets the predictions column to be one element per row from the list
        batch_size=len(
            dataset
        ),  # need to set this so that it doesn't have shape mismatch errors in the length of the column.
    )

    return dataset


#########################
# EXPERIMENT MANAGEMENT #
#########################
def get_raw_data_dir(dataset_name: str, subsplit: str):
    return os.path.join(
        "data",
        dataset_name,
        "splits",
        subsplit,
    )


def construct_paths_and_dataset_kwargs(
    DATASET_NAME: str,
    SUBSPLIT: str,
    SEED: int,
    TRAIN_SIZE: int,
    MODEL_ID: str,
    PEFT: bool,
    LORA_MODULES: List[str],
    LOAD_IN_4BIT: bool,
    LOAD_IN_8BIT: bool,
    BATCH_SZ: int,
    GRAD_ACCUM: int,
    NO_TRAIN: bool,
    CONTEXT_WEIGHT_AT_END: bool,
    CONTEXT_WEIGHT_FORMAT: str,
    OVERWRITE: bool = False,
    verbose: bool = False,
):
    DATASET_KWARGS_IDENTIFIABLE = dict(
        seed=SEED,
        train_size=TRAIN_SIZE,
        # context_weight_format=CONTEXT_WEIGHT_FORMAT,
    )
    MODEL_KWARGS_IDENTIFIABLE = dict(
        PEFT=PEFT,
        LORA_MODULES=LORA_MODULES,
        LOAD_IN_4BIT=LOAD_IN_4BIT,
        LOAD_IN_8BIT=LOAD_IN_8BIT,
        BATCH_SZ=BATCH_SZ,
        GRAD_ACCUM=GRAD_ACCUM,
        NO_TRAIN=NO_TRAIN,
    )

    # Paths
    # Construct dataset and data ids
    data_id = f"{DATASET_NAME}_{SUBSPLIT}"
    data_id += f"-ts{TRAIN_SIZE}" if TRAIN_SIZE is not None else ""

    data_dir = os.path.join(
        "data",
        DATASET_NAME,
        data_id,
        f"{SEED}",
    )
    input_dir = os.path.join(data_dir, "inputs")

    # train_path = os.path.join(input_dir, "train.csv")
    # val_path = os.path.join(input_dir, "val.csv")
    # test_path = os.path.join(input_dir, "test.csv")

    raw_data_dir = get_raw_data_dir(
        dataset_name=DATASET_NAME,
        subsplit=SUBSPLIT,
    )
    train_path = os.path.join(raw_data_dir, "train.csv")
    val_path = os.path.join(raw_data_dir, "val.csv")
    test_path = os.path.join(raw_data_dir, "test.csv")

    DATASET_KWARGS_IDENTIFIABLE = {
        **DATASET_KWARGS_IDENTIFIABLE,
        **dict(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
        ),
    }

    # Check if model id is a path
    if os.path.exists(MODEL_ID):
        # parse only the model id
        MODEL_ID = os.path.basename(MODEL_ID)
        
    # Construct model id
    model_id = MODEL_ID
    model_id += f"-peft{'_'.join(LORA_MODULES)}" if MODEL_KWARGS_IDENTIFIABLE["PEFT"] else ""
    model_id += "-4bit" if MODEL_KWARGS_IDENTIFIABLE["LOAD_IN_4BIT"] else ""
    model_id += "-8bit" if MODEL_KWARGS_IDENTIFIABLE["LOAD_IN_8BIT"] else ""
    model_id += f"-bs{MODEL_KWARGS_IDENTIFIABLE['BATCH_SZ']}"
    model_id += f"-ga{MODEL_KWARGS_IDENTIFIABLE['GRAD_ACCUM']}" if MODEL_KWARGS_IDENTIFIABLE["GRAD_ACCUM"] != 1 else ""
    model_id += "-NT" if NO_TRAIN else ""
    model_id += "-cwe" if CONTEXT_WEIGHT_AT_END else ""
    model_id += f"-cwf_{CONTEXT_WEIGHT_FORMAT}"

    model_parent_dir = os.path.join(data_dir, "models", model_id)
    model_dir = os.path.join(model_parent_dir, "model")

    # Results path
    results_dir = os.path.join(model_parent_dir, "results")
    val_results_path = os.path.join(results_dir, "val.csv")

    if verbose:
        print(f"Data dir: {data_dir}")
        print(f"Model dir: {model_dir}")
        print(f"Results dir: {results_dir}")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(model_parent_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    return (
        data_dir,
        input_dir,
        model_dir,
        results_dir,
        val_results_path,
        data_id,
        model_id,
        DATASET_KWARGS_IDENTIFIABLE,
        MODEL_KWARGS_IDENTIFIABLE,
    )


def construct_artifact_name(data_id, SEED, model_id, prefix=""):
    artifact_name = f"{data_id}-{SEED}-{model_id}".replace("/", ".")
    artifact_name = prefix + hashlib.sha256(artifact_name.encode()).hexdigest()[:8]
    return artifact_name


def get_gpu_memory() -> List[int]:
    # From https://stackoverflow.com/a/59571639
    # Returns list of MB of free GPU memory per gpu
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


###################
# DATA PROCESSING #
###################
def format_prompts(
    examples: Union[Dataset, dict],
    eos_token: str,
    prompt_template_dict: str,
    context_weight_format: str,
    context_weight_at_end: bool = False,
    demonstrations_df: pd.DataFrame = pd.DataFrame(),
    do_eval: bool = False,
) -> List[str]:
    """
    Construct a prompt for each example in examples using the prompt_template.

    Args:
        examples - a dataset containing columns ["context", "query", "weight_context", "answer"],
        eos_token - the eos token required to signify the end of the prompt.
        prompt_template - the prompt template for which to fill out with examples
        do_eval - whether to construct the prompt for evaluation mode (True) or training mode (False). For eval mode, the answer is not included in the prompt.
        context_weight_at_end - whether to include the context weight at the end of the context.

    Returns:
        a list of prompts that combines the instruction, formatted input, and expected answer for each example.
    """
    return [
        construct_query_with_demonstrations(
            prompt_template_dict=prompt_template_dict,
            eos_token=eos_token,
            demonstrations_df=demonstrations_df,
            val_context=context,
            val_query=query,
            val_answer=answer,
            context_weight=context_weight,
            context_weight_format=context_weight_format,
            context_weight_at_end=context_weight_at_end,
            do_eval=do_eval,
        )
        for (context, query, answer, context_weight) in zip(
            examples["context"], examples["query"], examples["answer"], examples["weight_context"]
        )
    ]

    # instructions = len(examples["context"]) * ["Answer the following query considering the provided context."]
    # inputs = [
    #     f"Context: {context}\nContext weight: {weight:.2f}\nQuery: {query}" if not context_weight_at_end else f"Context: {context}\nQuery: {query}\nContext weight: {weight:.2f}"
    #     for context, weight, query in zip(examples["context"], examples["weight_context"], examples["query"])
    # ]

    # # Must add EOS_TOKEN during training, otherwise your generation will go on forever!
    # # NOTE: this assumes that eos_token is the end of the answer and there's nothing else in the prompt template after the answer.
    # outputs = [answer + eos_token if not do_eval else "" for answer in examples["answer"]]

    # texts = [
    #     formatted_demonstrations + prompt_template_dict.format(instruction, inp, output)
    #     for instruction, inp, output in zip(instructions, inputs, outputs)
    # ]

    # return texts


QUERY_TEMPLATE_FLOAT = """Context: {context}
Context weight: {weight:.2f}
Query: {query}"""

QUERY_TEMPLATE_FLOAT_CTX_W_END = """Context: {context}
Query: {query}
Context weight: {weight:.2f}"""

QUERY_TEMPLATE_STR = """Context: {context}
Instruction: {weight}
Query: {query}"""

QUERY_TEMPLATE_STR_CTX_W_END = """Context: {context}
Query: {query}
Instruction: {weight}"""


# LLAMA3 INSTRUCT
LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE = (
    {
        "SYSTEM": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
        "ROUND": "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{}",
        "END_OF_ROUND": "<|eot_id|>",
    },
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
)  # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

# MISTRAL INSTRUCT
MISTRAL_INSTRUCT_PROMPT_TEMPLATE_DICT, MISTRAL_INSTRUCT_RESPONSE_TEMPLATE = (
    {
        "SYSTEM": """<s>[INST] {} \n""",
        "ROUND": """{}[/INST]{}""",
        "END_OF_ROUND": """</s>[INST]""",
    },
    "[/INST]",
)  # https://www.promptingguide.ai/models/mistral-7b#chat-template-for-mistral-7b-instruct

# LLAMA2 CHAT
LLAMA2_PROMPT_TEMPLATE_DICT, LLAMA2_RESPONSE_TEMPLATE = (
    {
        "SYSTEM": """<s>[INST] <<SYS>> {} <</SYS>> \n""",
        "ROUND": """{}[/INST]{}""",
        "END_OF_ROUND": """[INST]""",
    },
    "[/INST]",
)  # https://developer.ibm.com/tutorials/awb-prompt-engineering-llama-2/

# GEMMA
GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE = (
    {
        "SYSTEM": """<start_of_turn>user\n{}""",
        "ROUND": """{}<end_of_turn>\n<start_of_turn>model\n{}""",
        "END_OF_ROUND": """<end_of_turn>""",
    },
    "<start_of_turn>model",
)  # https://www.promptingguide.ai/models/gemma#how-to-prompt-gemma-7b


MODEL_ID_TO_TEMPLATES_DICT = {
    "unsloth/llama-3-8b-Instruct-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "Meta-Llama-3.1-8B-Instruct": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "Meta-Llama-3-8B-Instruct": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/llama-3-8b-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit": (
        MISTRAL_INSTRUCT_PROMPT_TEMPLATE_DICT,
        MISTRAL_INSTRUCT_RESPONSE_TEMPLATE,
    ),
    "unsloth/llama-2-7b-chat-bnb-4bit": (LLAMA2_PROMPT_TEMPLATE_DICT, LLAMA2_RESPONSE_TEMPLATE),
    "unsloth/llama-2-7b-bnb-4bit": (LLAMA2_PROMPT_TEMPLATE_DICT, LLAMA2_RESPONSE_TEMPLATE),
    "unsloth/gemma-2b-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-7b-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-2b-it-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-7b-it-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
}

CTX_WEIGHT_FORMAT_TO_FUNC_AND_QUERY_TEMPLATE = {
    "float": {
        "format_func": lambda ctx_w: ctx_w,
        "query_template": {
            False: QUERY_TEMPLATE_FLOAT,
            True: QUERY_TEMPLATE_FLOAT_CTX_W_END,
        },
    },
    "instruction": {
        "format_func": lambda ctx_w: {
            0: "Ignore the context in answering the query.",
            1: "Only consider the context in answering the query.",
        }[ctx_w],
        "query_template": {
            False: QUERY_TEMPLATE_STR,
            True: QUERY_TEMPLATE_STR_CTX_W_END,
        },
    },
}  # Given a format type, return (a) a function which will  map a given context weight (as a float) to its string representation AND (b) the query template for that format type.


def construct_query_with_demonstrations(
    prompt_template_dict: Dict[str, str],
    eos_token: str,
    demonstrations_df: pd.DataFrame,  # can be empty
    val_context: str,
    val_query: str,
    val_answer: str,
    context_weight: int = 1.0,
    context_weight_format: str = "float",
    context_weight_at_end: bool = False,
    do_eval: bool = False,
) -> str:
    format_ctx_weight_func = CTX_WEIGHT_FORMAT_TO_FUNC_AND_QUERY_TEMPLATE[context_weight_format]["format_func"]
    query_template = CTX_WEIGHT_FORMAT_TO_FUNC_AND_QUERY_TEMPLATE[context_weight_format]["query_template"][
        context_weight_at_end
    ]

    system = prompt_template_dict["SYSTEM"].format("Answer the following query considering the provided context.")

    # Construct the demontrations into the string (if they exist)
    rounds = []
    for i, row in demonstrations_df.iterrows():
        query = query_template.format(
            context=row["context"], weight=format_ctx_weight_func(row["weight_context"]), query=row["query"]
        )
        round = prompt_template_dict["ROUND"].format(query, row["answer"])
        round += prompt_template_dict["END_OF_ROUND"]
        rounds.append(round)

    query = query_template.format(context=val_context, weight=format_ctx_weight_func(context_weight), query=val_query)

    out = system
    out += "".join(rounds)
    out += prompt_template_dict["ROUND"].format(
        query,
        "" if do_eval else val_answer + prompt_template_dict["END_OF_ROUND"] + eos_token
        # Must add EOS_TOKEN during training, otherwise your generation will go on forever!
    )

    return out


def sample_few_shot_examples(train_df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Assume that train_df contains 0/1 context weight examples adjacent to each other."""
    shot_indices = train_df[::2].sample(k//2, random_state=seed).index
    shot_indices = [(i, i + 1) for i in shot_indices]
    shot_indices = np.array(shot_indices).flatten()
    shot_sample = train_df.loc[shot_indices]
    return shot_sample


# ALPACA_PROMPT, ALPACA_RESPONSE_TEMPLATE = (
#     """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# {}

# ### Input:
# {}

# ### Response:
# {}""",
#     "Response:",
# )

# GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE = (
#     """<start_of_turn>user
# {}

# {}<end_of_turn>
# <start_of_turn>model
# {}""",
#     "<start_of_turn>model",
# )  # https://www.promptingguide.ai/models/gemma#how-to-prompt-gemma-7b

# GPT2_PROMPT, GPT2_RESPONSE_TEMPLATE = (
#     """{}
# Q: {}
# A: {}""",
#     "A:",
# )

# PHI_PROMPT, PHI_RESPONSE_TEMPLATE = (
#     """Instruct: {}
# {}
# Output: {}""",
#     "Output:",
# )

# MISTRAL_INSTRUCT_PROMPT, MISTRAL_INSTRUCT_RESPONSE_TEMPLATE = (
#     "<s>[INST] {}\n{} [/INST] {}",
#     "[/INST] ",
# )  # https://www.promptingguide.ai/models/mistral-7b#chat-template-for-mistral-7b-instruct

# LLAMA2_PROMPT, LLAMA2_RESPONSE_TEMPLATE = (
#     """<s>[INST] <<SYS>>
# {}
# <</SYS>>

# {} [/INST]{}
# """,
#     "[/INST]",
# )  # https://developer.ibm.com/tutorials/awb-prompt-engineering-llama-2/

# LLAMA3_PROMPT, LLAMA3_RESPONSE_TEMPLATE = (
#     """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# {}<|eot_id|><|start_header_id|>user<|end_header_id|>

# {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}
# """,
#     "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
# )  # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

# PROMPTS_DICT = {
#     "unsloth/mistral-7b-v0.2-bnb-4bit": (ALPACA_PROMPT, ALPACA_RESPONSE_TEMPLATE),
#     "unsloth/mistral-7b-instruct-v0.2-bnb-4bit": (MISTRAL_INSTRUCT_PROMPT, MISTRAL_INSTRUCT_RESPONSE_TEMPLATE),
#     "unsloth/llama-2-7b-bnb-4bit": (LLAMA2_PROMPT, LLAMA2_RESPONSE_TEMPLATE),
#     "unsloth/llama-2-7b-chat-bnb-4bit": (LLAMA2_PROMPT, LLAMA2_RESPONSE_TEMPLATE),
#     "unsloth/llama-3-8b-bnb-4bit": (LLAMA3_PROMPT, LLAMA3_RESPONSE_TEMPLATE),
#     "unsloth/llama-3-8b-Instruct-bnb-4bit": (LLAMA3_PROMPT, LLAMA3_RESPONSE_TEMPLATE),
#     "unsloth/gemma-2b-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
#     "unsloth/gemma-7b-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
#     "unsloth/gemma-2b-it-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
#     "unsloth/gemma-7b-it-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
#     "openai-community/gpt2": (GPT2_PROMPT, GPT2_RESPONSE_TEMPLATE),
#     "microsoft/phi-1_5": (PHI_PROMPT, PHI_RESPONSE_TEMPLATE),
# }


from typing import NamedTuple


def construct_test_results_dir(
    base_results_dir: str, eval_name: str, k_demonstrations: int, context_weight_format: str
):
    eval_id = eval_name
    eval_id += f"-k{k_demonstrations}"
    eval_id += f"-cwf_{context_weight_format}"
    return os.path.join(base_results_dir, eval_id)


class EvalConfig(NamedTuple):
    """Config for evaluating a model's ability to follow context vs prior according to a weight flag."""

    dataset_name: str
    k_demonstrations: int
    context_weight_format: str
