import os
import hashlib
from typing import Optional, List, Union, Dict, Tuple
from datasets import Dataset


def format_prompts(
    examples: Union[Dataset, dict], eos_token: str, prompt_template: str, do_eval: bool = False
) -> List[str]:
    """
    Construct a prompt for each example in examples using the prompt_template.

    Args:
        examples - a dataset containing columns ["context", "query", "weight_context", "answer"],
        eos_token - the eos token required to signify the end of the prompt.
        prompt_template - the prompt template for which to fill out with examples
        do_eval - whether to construct the prompt for evaluation mode (True) or training mode (False). For eval mode, the answer is not included in the prompt.

    Returns:
        a list of prompts that combines the instruction, formatted input, and expected answer for each example.
    """
    instructions = len(examples["context"]) * ["Answer the following query considering the provided context."]
    inputs = [
        f"Context: {context} \nContext weight: {weight:.2f}\nQuery: {query}"
        for context, weight, query in zip(examples["context"], examples["weight_context"], examples["query"])
    ]

    # Must add EOS_TOKEN during training, otherwise your generation will go on forever!
    # NOTE: this assumes that eos_token is the end of the answer and there's nothing else in the prompt template after the answer.
    outputs = [answer + eos_token if not do_eval else "" for answer in examples["answer"]]

    texts = [
        prompt_template.format(instruction, inp, output)
        for instruction, inp, output in zip(instructions, inputs, outputs)
    ]

    return texts


def construct_paths_and_dataset_kwargs(
    DATASET_NAME: str,
    SEED: int,
    TRAIN_SIZE: int,
    MODEL_ID: str,
    PEFT: bool,
    LORA_MODULES: List[str],
    LOAD_IN_4BIT: bool,
    LOAD_IN_8BIT: bool,
    BATCH_SZ: int,
    NO_TRAIN: bool,
    OVERWRITE: bool,
    verbose=False,
):
    DATASET_KWARGS_IDENTIFIABLE = dict(
        seed=SEED,
        train_size=TRAIN_SIZE,
        # overwrite=OVERWRITE,
    )
    MODEL_KWARGS_IDENTIFIABLE = dict(
        PEFT=PEFT,
        LORA_MODULES=LORA_MODULES,
        LOAD_IN_4BIT=LOAD_IN_4BIT,
        LOAD_IN_8BIT=LOAD_IN_8BIT,
        BATCH_SZ=BATCH_SZ,
        NO_TRAIN=NO_TRAIN,
    )

    # Paths
    # Construct dataset and data ids
    data_id = DATASET_NAME
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
    #
    # DATASET_KWARGS_IDENTIFIABLE = {
    #     **DATASET_KWARGS_IDENTIFIABLE,
    #     **dict(
    #         train_path=train_path,
    #         val_path=val_path,
    #         test_path=test_path,
    #     ),
    # }

    # Construct model id
    model_id = MODEL_ID
    model_id += f"-peft{'_'.join(LORA_MODULES)}" if MODEL_KWARGS_IDENTIFIABLE["PEFT"] else ""
    model_id += "-4bit" if MODEL_KWARGS_IDENTIFIABLE["LOAD_IN_4BIT"] else ""
    model_id += "-8bit" if MODEL_KWARGS_IDENTIFIABLE["LOAD_IN_8BIT"] else ""
    model_id += f"-bs{MODEL_KWARGS_IDENTIFIABLE['BATCH_SZ']}"
    model_id += "-NT" if NO_TRAIN else ""

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
)

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

PROMPTS_DICT = {
    "unsloth/mistral-7b-v0.2-bnb-4bit": (ALPACA_PROMPT, ALPACA_RESPONSE_TEMPLATE),
    "unsloth/gemma-2b-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-7b-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
    "openai-community/gpt2": (GPT2_PROMPT, GPT2_RESPONSE_TEMPLATE),
    "microsoft/phi-1_5": (PHI_PROMPT, PHI_RESPONSE_TEMPLATE),
}
