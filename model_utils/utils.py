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

from model_utils.mi_utils import compute_sus_and_persuasion_scores


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
    try_load_as_peft: bool = False,
    padding_side: str = "right",
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

    if peft_config is not None or try_load_as_peft:
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
            tokenizer = prepare_tokenizer(model, padding_side=padding_side)
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
        tokenizer = prepare_tokenizer(model, padding_side=padding_side)
    print(f"Loaded model on device {model.device} with dtype {model.dtype}.")

    torch.cuda.empty_cache()
    gc.collect()

    return model, tokenizer


def prepare_tokenizer(model, set_pad_token=True, padding_side="right"):
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    tokenizer.padding_side = padding_side  # for kbit training apparently you need to pad on the right
    if set_pad_token and tokenizer.pad_token is None:
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
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def merge_save_peft(peft_model, tokenizer, path):
    """Merge the peft model and save to path."""

    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    tokenizer.padding_side = "left"

    return merged_model, tokenizer


##############
# EVALUATION #
##############
def response_startswith_label(response: str, label: str) -> bool:
    return response.startswith(label)


def compute_mr(df, is_response_correct_func=response_startswith_label) -> Tuple[float, float, float]:
    """
    Given a df with columns `predictions`,  `prior_answer`, and `ctx_answer`, return a tuple containing (MR, % of other answers).
    MR = (# prior) / (# prior + # context)
    CR = (# ctx) / (# prior + # context)
    % other answers = (# other) / (# prior + # context + # other)
    """
    if len(df) == 0:
        return None, None
    num_prior_answers = df.apply(
        lambda row: is_response_correct_func(row["predictions"], row["prior_answer"]), axis=1
    ).sum()
    num_ctx_answers = df.apply(
        lambda row: is_response_correct_func(row["predictions"], row["ctx_answer"]), axis=1
    ).sum()
    num_other_answers = len(df) - (num_ctx_answers + num_prior_answers)
    if num_prior_answers + num_ctx_answers == 0:
        print("No correct prior or context answers. Returning None")
        return None, num_other_answers / len(df)
    return num_prior_answers / (num_prior_answers + num_ctx_answers), num_other_answers / len(df)


def compute_pair_acc(df):
    return df.groupby(["context", "query"]).agg("min")["is_correct"].mean()


def compute_metrics(df, is_response_correct_func=response_startswith_label):
    ctx_pref_df = df[df["weight_context"] == 1.0]
    prior_pref_df = df[df["weight_context"] == 0.0]

    context_acc = ctx_pref_df["is_correct"].mean()
    prior_acc = prior_pref_df["is_correct"].mean()

    context_mr, context_other = compute_mr(ctx_pref_df, is_response_correct_func=is_response_correct_func)
    prior_mr, prior_other = compute_mr(prior_pref_df, is_response_correct_func=is_response_correct_func)

    overall_mr, overall_other = compute_mr(df)

    pair_acc = compute_pair_acc(df)

    metrics = {
        "acc": df["is_correct"].mean(),  # Overall accuracy
        "pair_acc": pair_acc,
        "context_acc": context_acc,  # accuracy across the examples that SHOULD follow the context
        "prior_acc": prior_acc,  # accuracy across the examples that SHOULD follow the prior
        "context_mr": context_mr,  # MR across the examples that SHOULD follow the context (we want this to be low)
        "prior_mr": prior_mr,  # MR across the examples that SHOULD follow the context (we want this to be high)
        "overall_mr": overall_mr,  # MR across all examples (we want this to be 50%)
        "context_pct_other": context_other,  # percent of examples featured a non-context or prior answer across examples that SHOULD follow the context (lower better)
        "prior_pct_other": prior_other,  # percent of examples that featured a non-context or prior answer across examples that SHOULD follow the prior (lower better)
        "overall_pct_other": overall_other,  # percent of examples that featured a non-context or prior answer across all examples (lower better)
    }

    if "query_only_is_correct" in df.columns:
        metrics["query_only_acc"] = df["query_only_is_correct"].mean()

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
    is_response_correct_func=response_startswith_label,
    hook=None,
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
        dataset.to_pandas()[["query", "prior_answer", "weight_context"]].drop_duplicates(), preserve_index=False
    )
    queries_only_dataset = queries_only_dataset.rename_column(
        "prior_answer", "labels"
    )  # need to make the labels column

    encoded_dataset = queries_only_dataset.map(
        lambda examples: tokenizer(examples["query"], padding=True, return_tensors="pt"),
        batched=True,
        batch_size=batch_sz,
    ).select_columns(["input_ids", "attention_mask", "labels", "weight_context"])
    encoded_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "weight_context"], device="cuda"
    )  # required for loading correctly into dataloader
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_sz)
    predictions, labels, is_correct_all = [], [], []
    num_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if hook is not None:
                values = torch.tensor(batch["weight_context"] == 1.0)
                hook.set_binary(values)
            batch.pop("weight_context")
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
                is_response_correct_func(response, label) for response, label in zip(decoded_responses, batch["labels"])
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
    max_new_tokens: int = 10,
    batch_sz: int = 8,  # "auto",
    is_response_correct_func=response_startswith_label,
    hook=None,
    feature_collection_hook=None,
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
        lambda examples: tokenizer(examples["text"], padding=True, return_tensors="pt"),
        batched=True,
        batch_size=batch_sz,
    ).select_columns(["input_ids", "attention_mask", "labels", "weight_context"])
    print(encoded_dataset[0])
    encoded_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "weight_context"], device="cuda"
    )  # required for loading correctly into dataloader
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_sz)
    predictions, labels, is_correct_all = [], [], []
    num_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if hook is not None:
                values = torch.tensor(batch["weight_context"] == 1.0)
                hook.set_binary(values)
            if feature_collection_hook is not None:
                feature_collection_hook.attach(model)
            batch.pop("weight_context")
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
                is_response_correct_func(response, label) for response, label in zip(decoded_responses, batch["labels"])
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


def evaluate_model_pscores(
    model,
    tokenizer,
    dataset: Dataset,
    format_func,
    batch_sz: int = 8,  # "auto",
):
    """
    Given a dataset with columns ["text", "labels"], generate answers and evaluate model accuracy against those labels.
    1. Generate predictions from text
    2. Extract answer, compare to labels, and return accuracy
    """
    # import pdb; pdb.set_trace()
    from collections import namedtuple

    context_info = namedtuple("context_info", ["context", "context_weight"])
    # Free gpu memory
    gc.collect()
    torch.cuda.empty_cache()

    if batch_sz == "auto":
        batch_sz = int(2 * int(sum(get_gpu_memory()) / 1000))
        print(f"Setting batch size to {batch_sz} for eval.")
    tokenizer.padding_side = "left"
    contexts = [
        context_info(context=dataset["context"][i], context_weight=dataset["weight_context"][i])
        for i in range(len(dataset))
    ]

    def unpack_sus_and_pscore(example, i):
        sus_score, p_scores = compute_sus_and_persuasion_scores(
            query=example["query"],
            entity=None,
            contexts=contexts,
            format_func=format_func,
            model=model,
            tokenizer=tokenizer,
            answer_map=None,
            bs=batch_sz,
            answer_entity=None,
        )

        return {
            "sus_score": sus_score,
            "p_score": p_scores[
                i
            ],  # the contexts passed in to compute_sus_and_persuasion_scores are in the same order as the rows in the dataset.
        }

    dataset = dataset.map(unpack_sus_and_pscore, with_indices=True)

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
    ANSWER_FORMAT_PROMPT_POSITION: str,
    ADD_ANSWER_FORMAT_PROMPT: bool,
    verbose: bool = False,
):
    DATASET_KWARGS_IDENTIFIABLE = dict(
        seed=SEED,
        train_size=TRAIN_SIZE,
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
    model_id += "-4bit" if MODEL_KWARGS_IDENTIFIABLE["LOAD_IN_4BIT"] else ""
    model_id += "-8bit" if MODEL_KWARGS_IDENTIFIABLE["LOAD_IN_8BIT"] else ""
    if not NO_TRAIN:
        model_id += f"-peft{'_'.join(LORA_MODULES)}" if MODEL_KWARGS_IDENTIFIABLE["PEFT"] else ""
        model_id += f"-bs{MODEL_KWARGS_IDENTIFIABLE['BATCH_SZ']}"
        model_id += (
            f"-ga{MODEL_KWARGS_IDENTIFIABLE['GRAD_ACCUM']}" if MODEL_KWARGS_IDENTIFIABLE["GRAD_ACCUM"] != 1 else ""
        )
        model_id += "-cwe" if CONTEXT_WEIGHT_AT_END else ""
        model_id += f"-cwf_{CONTEXT_WEIGHT_FORMAT}"
        if ADD_ANSWER_FORMAT_PROMPT:
            model_id += f"-afpp_{ANSWER_FORMAT_PROMPT_POSITION}"
    else:
        model_id += "-NT"

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
    demonstrations_context_weight_format: str,
    query_context_weight_format: str,
    context_weight_at_end: bool = False,
    demonstrations_df: pd.DataFrame = pd.DataFrame(),
    do_eval: bool = False,
    answer_format: str = "word",
    add_answer_format_prompt: bool = True,
    answer_format_prompt_position: str = "start",
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
            demonstrations_context_weight_format=demonstrations_context_weight_format,
            query_context_weight_format=query_context_weight_format,
            context_weight_at_end=context_weight_at_end,
            do_eval=do_eval,
            answer_format=answer_format,
            add_answer_format_prompt=add_answer_format_prompt,
            answer_format_prompt_position=answer_format_prompt_position,
        )
        for (context, query, answer, context_weight) in zip(
            examples["context"], examples["query"], examples["answer"], examples["weight_context"]
        )
    ]


QUERY_TEMPLATE_NO_INSTRUCTION = """Context: {context}
Query: {query}"""

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

BASE_TEMPLATE_DICT, BASE_RESPONSE_TEMPLATE = (
    {
        "SYSTEM": "System Instruction: {}\n",
        "ROUND": "User: {}\n\nAssistant: {}",
        "END_OF_ROUND": "\n\n",
    },
    "\n\nAssistant:",
)

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
        "SYSTEM": """<start_of_turn>user\n{} """,
        "ROUND": """{}<end_of_turn>\n<start_of_turn>model\n{}""",
        "END_OF_ROUND": """<end_of_turn>""",
    },
    "<start_of_turn>model",
)  # https://www.promptingguide.ai/models/gemma#how-to-prompt-gemma-7b

MODEL_ID_TO_TEMPLATES_DICT = {
    "unsloth/llama-3-8b-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/llama-3-8b-Instruct-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/llama-3-8b-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "Meta-Llama-3.1-8B-Instruct": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "Meta-Llama-3.1-8B": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "Meta-Llama-3-8B-Instruct": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "Meta-Llama-3-8B": (BASE_TEMPLATE_DICT, BASE_RESPONSE_TEMPLATE),
    "unsloth/llama-3-8b-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit": (
        MISTRAL_INSTRUCT_PROMPT_TEMPLATE_DICT,
        MISTRAL_INSTRUCT_RESPONSE_TEMPLATE,
    ),
    "Mistral-7B-Instruct-v0.3": (
        MISTRAL_INSTRUCT_PROMPT_TEMPLATE_DICT,
        MISTRAL_INSTRUCT_RESPONSE_TEMPLATE,
    ),
    "Mistral-7B-v0.3": (
        MISTRAL_INSTRUCT_PROMPT_TEMPLATE_DICT,
        MISTRAL_INSTRUCT_RESPONSE_TEMPLATE,
    ),
    "unsloth/mistral-7b-v0.3-bnb-4bit": (
        MISTRAL_INSTRUCT_PROMPT_TEMPLATE_DICT,
        MISTRAL_INSTRUCT_RESPONSE_TEMPLATE,
    ),
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit": (
        MISTRAL_INSTRUCT_PROMPT_TEMPLATE_DICT,
        MISTRAL_INSTRUCT_RESPONSE_TEMPLATE,
    ),
    "unsloth/llama-2-7b-chat-bnb-4bit": (LLAMA2_PROMPT_TEMPLATE_DICT, LLAMA2_RESPONSE_TEMPLATE),
    "unsloth/llama-2-7b-bnb-4bit": (LLAMA2_PROMPT_TEMPLATE_DICT, LLAMA2_RESPONSE_TEMPLATE),
    "google/gemma-2-9b": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-7b-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-2b-it-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-7b-it-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-2b-it-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "gemma-2-9b": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "gemma-2-9b-it": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-2-9b-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-2-9b-it-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "openai-community/gpt2": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
}

ANSWER_FORMAT_PROMPT = {
    "word": "Output format: Answer with a single word.",
    "number": "Output format: Answer with a single number.",
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
    "none": {
        "format_func": lambda ctx_w: ctx_w,
        "query_template": {
            False: QUERY_TEMPLATE_NO_INSTRUCTION,
            True: QUERY_TEMPLATE_NO_INSTRUCTION,
        },
    },
}  # Given a format type, return (a) a function which will  map a given context weight (as a float) to its string representation AND (b) the query template for that format type.


def create_pscore_format_func(
    prompt_template_dict: Dict[str, str],
    eos_token: str,
    demonstrations_df: pd.DataFrame,  # can be empty
    demonstrations_context_weight_format: str = "float",
    query_context_weight_format: str = "float",
    context_weight_at_end: bool = False,
    answer_format: str = "word",
    add_answer_format_prompt: bool = True,
):
    return lambda query, entity, context: construct_query_with_demonstrations(
        val_query=query,
        val_context=context.context,
        context_weight=context.context_weight,
        val_answer=None,
        prompt_template_dict=prompt_template_dict,
        eos_token=eos_token,
        demonstrations_df=demonstrations_df,
        demonstrations_context_weight_format=demonstrations_context_weight_format,
        query_context_weight_format=query_context_weight_format,
        context_weight_at_end=context_weight_at_end,
        do_eval=True,
        answer_format=answer_format,
        add_answer_format_prompt=add_answer_format_prompt,
    )


def construct_query_with_demonstrations(
    prompt_template_dict: Dict[str, str],
    eos_token: str,
    demonstrations_df: pd.DataFrame,  # can be empty
    val_context: str,
    val_query: str,
    val_answer: str,
    context_weight: int = 1.0,
    demonstrations_context_weight_format: str = "float",
    query_context_weight_format: str = "float",
    context_weight_at_end: bool = False,
    do_eval: bool = False,
    answer_format: str = "word",
    add_answer_format_prompt: bool = False,
    answer_format_prompt_position: str = "start",
) -> str:
    if demonstrations_context_weight_format is None:
        demonstrations_context_weight_format = query_context_weight_format
    return (
        construct_system_prompt(prompt_template_dict)
        + construct_demonstrations(
            prompt_template_dict=prompt_template_dict,
            eos_token=eos_token,
            demonstrations_df=demonstrations_df,  # can be empty
            context_weight_format=demonstrations_context_weight_format,
            context_weight_at_end=context_weight_at_end,
            answer_format=answer_format,
            add_answer_format_prompt=add_answer_format_prompt,
            answer_format_prompt_position=answer_format_prompt_position,
        )
        + construct_query(
            prompt_template_dict=prompt_template_dict,
            eos_token=eos_token,
            val_context=val_context,
            val_query=val_query,
            val_answer=val_answer,
            context_weight=context_weight,
            context_weight_format=query_context_weight_format,
            context_weight_at_end=context_weight_at_end,
            do_eval=do_eval,
            answer_format=answer_format,
            add_answer_format_prompt=add_answer_format_prompt,
            answer_format_prompt_position=answer_format_prompt_position,
        )
    )


def construct_system_prompt(prompt_template_dict):
    return prompt_template_dict["SYSTEM"].format(
        "Answer the following query considering the provided context. Answer with only one word."
    )


def construct_demonstrations(
    prompt_template_dict: Dict[str, str],
    eos_token: str,
    demonstrations_df: pd.DataFrame,  # can be empty
    context_weight_format: Optional[str] = "float",
    context_weight_at_end: bool = False,
    answer_format: str = "word",
    add_answer_format_prompt: bool = True,
    answer_format_prompt_position: str = "start",
):
    if context_weight_format is None:
        if len(demonstrations_df) > 0:
            raise ValueError(
                "context weight format for demonstrations is None but demonstrations_df is not empty. Either remove the demonstrations or specify how to format them."
            )
        else:
            return ""

    format_ctx_weight_func = CTX_WEIGHT_FORMAT_TO_FUNC_AND_QUERY_TEMPLATE[context_weight_format]["format_func"]
    query_template = CTX_WEIGHT_FORMAT_TO_FUNC_AND_QUERY_TEMPLATE[context_weight_format]["query_template"][
        context_weight_at_end
    ]

    # Construct the demontrations into the string (if they exist)
    rounds = []
    for i, row in demonstrations_df.iterrows():
        query = query_template.format(
            context=row["context"], weight=format_ctx_weight_func(row["weight_context"]), query=row["query"]
        )
        if add_answer_format_prompt:
            if answer_format_prompt_position == "start":
                query = f"{ANSWER_FORMAT_PROMPT[answer_format]}\n{query}"
            else:
                query = f"{query}\n{ANSWER_FORMAT_PROMPT[answer_format]}"
        round = prompt_template_dict["ROUND"].format(query, row["answer"])
        round += prompt_template_dict["END_OF_ROUND"]
        rounds.append(round)

    return "".join(rounds)


def construct_query(
    prompt_template_dict: Dict[str, str],
    eos_token: str,
    val_context: str,
    val_query: str,
    val_answer: str,
    answer_format: str = "word",
    context_weight: int = 1.0,
    context_weight_format: str = "float",
    context_weight_at_end: bool = False,
    do_eval: bool = False,
    add_answer_format_prompt: bool = True,
    answer_format_prompt_position: str = "start",
):
    format_ctx_weight_func = CTX_WEIGHT_FORMAT_TO_FUNC_AND_QUERY_TEMPLATE[context_weight_format]["format_func"]
    query_template = CTX_WEIGHT_FORMAT_TO_FUNC_AND_QUERY_TEMPLATE[context_weight_format]["query_template"][
        context_weight_at_end
    ]
    query = query_template.format(context=val_context, weight=format_ctx_weight_func(context_weight), query=val_query)
    if add_answer_format_prompt:
        if answer_format_prompt_position == "start":
            query = f"{ANSWER_FORMAT_PROMPT[answer_format]}\n{query}"
        else:
            query = f"{query}\n{ANSWER_FORMAT_PROMPT[answer_format]}"
    return prompt_template_dict["ROUND"].format(
        query,
        "" if do_eval else val_answer + prompt_template_dict["END_OF_ROUND"] + eos_token
        # Must add EOS_TOKEN during training, otherwise your generation will go on forever!
    )


def sample_few_shot_examples(train_df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """
    Assume that train_df contains 0/1 context weight examples adjacent to each other.
    k - total number of few shot examples (k / 2 pairs)
    """
    shot_indices = train_df[::2].sample(k // 2, random_state=seed).index
    shot_indices = [(i, i + 1) for i in shot_indices]
    shot_indices = np.array(shot_indices).flatten()
    shot_sample = train_df.loc[shot_indices]
    return shot_sample


ALPACA_PROMPT, ALPACA_RESPONSE_TEMPLATE = (
    """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}""",
    "Response:",
)

GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE = (
    """<start_of_turn>user
{}

{}<end_of_turn>
<start_of_turn>model
{}""",
    "<start_of_turn>model",
)  # https://www.promptingguide.ai/models/gemma#how-to-prompt-gemma-7b

GPT2_PROMPT, GPT2_RESPONSE_TEMPLATE = (
    """{}
Q: {}
A: {}""",
    "A:",
)

PHI_PROMPT, PHI_RESPONSE_TEMPLATE = (
    """Instruct: {}
{}
Output: {}""",
    "Output:",
)

MISTRAL_INSTRUCT_PROMPT, MISTRAL_INSTRUCT_RESPONSE_TEMPLATE = (
    "<s>[INST] {}\n{} [/INST] {}",
    "[/INST] ",
)  # https://www.promptingguide.ai/models/mistral-7b#chat-template-for-mistral-7b-instruct

LLAMA2_PROMPT, LLAMA2_RESPONSE_TEMPLATE = (
    """<s>[INST] <<SYS>>
{}
<</SYS>>

{} [/INST]{}
""",
    "[/INST]",
)  # https://developer.ibm.com/tutorials/awb-prompt-engineering-llama-2/

LLAMA3_PROMPT, LLAMA3_RESPONSE_TEMPLATE = (
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}
""",
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
)  # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

PROMPTS_DICT = {
    "unsloth/mistral-7b-v0.2-bnb-4bit": (ALPACA_PROMPT, ALPACA_RESPONSE_TEMPLATE),
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit": (MISTRAL_INSTRUCT_PROMPT, MISTRAL_INSTRUCT_RESPONSE_TEMPLATE),
    "unsloth/llama-2-7b-bnb-4bit": (LLAMA2_PROMPT, LLAMA2_RESPONSE_TEMPLATE),
    "unsloth/llama-2-7b-chat-bnb-4bit": (LLAMA2_PROMPT, LLAMA2_RESPONSE_TEMPLATE),
    "unsloth/llama-3-8b-bnb-4bit": (LLAMA3_PROMPT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/llama-3-8b-Instruct-bnb-4bit": (LLAMA3_PROMPT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/gemma-2b-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-7b-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-2b-it-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-7b-it-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
    "openai-community/gpt2": (GPT2_PROMPT, GPT2_RESPONSE_TEMPLATE),
    "microsoft/phi-1_5": (PHI_PROMPT, PHI_RESPONSE_TEMPLATE),
}


from typing import NamedTuple


def construct_test_results_dir(
    base_results_dir: str,
    eval_name: str,
    subsplit: str,
    k_demonstrations: int,
    in_domain_demonstrations: bool,
    context_weight_format: str,
    answer_format_prompt_position: str,
    add_answer_format_prompt: bool,
    do_steering: bool,
    steering_prior_value: float,
    steering_context_value: float,
    steering_layer: str,
):
    eval_id = eval_name
    eval_id += f"-sp_{subsplit}"
    eval_id += f"-k{k_demonstrations}" + ("_ID" if in_domain_demonstrations else "_OOD")
    eval_id += f"-cwf_{context_weight_format}"
    if add_answer_format_prompt:
        eval_id += f"-afpp_{answer_format_prompt_position}"
    if do_steering:
        eval_id += f"-steer_l{steering_layer}_p{steering_prior_value}_c{steering_context_value}"
    return os.path.join(base_results_dir, eval_id)


class EvalConfig(NamedTuple):
    """Config for evaluating a model's ability to follow context vs prior according to a weight flag."""

    dataset_name: str
    subsplit: str
    k_demonstrations: int
    context_weight_format: str
    do_steering: bool = False
