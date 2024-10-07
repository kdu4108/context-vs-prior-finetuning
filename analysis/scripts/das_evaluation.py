import sys
sys.path.append("/dlabscratch1/jminder/repositories/context-vs-prior-finetuning")

import torch
import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm, trange
import re
import random

from main import load_model_and_tokenizer
from pyvene import (
    IntervenableModel,
    LowRankRotatedSpaceIntervention,
    IntervenableConfig,
)
import preprocessing.dataset as our_datasets
from analysis.circuit_utils.model import *
from analysis.circuit_utils.validation import *
from analysis.circuit_utils.decoding import *
from analysis.circuit_utils.utils import *
from analysis.circuit_utils.steering import CtxPriorHook
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def accuracy(is_correct):
    return float(sum(is_correct)) / len(is_correct)

def paired_accuracy(is_correct):
    is_correct = np.array(is_correct)
    even_correct = is_correct[::2]
    odd_correct = is_correct[1::2]
    return float(sum(even_correct & odd_correct)) / len(even_correct)

def iia_with_hook(model, dataset_name, hook, tokens, attention_mask, values, answers, tokenizer, batch_size=32, max_index=None, verbose=True):
    dataset: our_datasets.ContextQueryDataset = getattr(our_datasets, dataset_name)()
    if max_index is not None:
        tokens = tokens[:max_index]
        attention_mask = attention_mask[:max_index]
        values = values[:max_index]
        answers = answers[:max_index]
    generations = []
    for i in trange(0, len(tokens), batch_size):
        hook.set_context_prior(values[i:i+batch_size])
        generations.extend(model.generate(tokens[i:i+batch_size], attention_mask=attention_mask[i:i+batch_size], max_new_tokens=10, do_sample=False, temperature=None, top_k=None, top_p=None, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id).tolist())

    generations = [g[len(tokens[i]):] for i, g in enumerate(generations)]
    generations = tokenizer.batch_decode(generations, skip_special_tokens=False)
    is_correct = []
    for i, o in enumerate(generations):
        if verbose and i < 10:
            print("Answer:", f"'{answers[i]}'", "Generation:", f"'{o}'", "Correct:", str(answers[i]).strip() in o, dataset.is_response_correct(o.strip(), str(answers[i]).strip()))
        is_correct.append(dataset.is_response_correct(o.strip(), str(answers[i]).strip()))
    
    return {
        "accuracy": accuracy(is_correct),
        "paired_accuracy": paired_accuracy(is_correct),
        "generations": generations,
        "answers": answers.tolist(),
    }


def get_one_word_instruction(args):
    if "Arithmetic" in args.eval_dataset:
        return "Answer with one number only."
    else:
        return "Answer with one word only."

# def remove_instruction(text, name_of_instruction="Instruction", replace_with="Instruction: Answer with one word only.\n"):
def remove_instruction(text, replace_with, name_of_instruction="Instruction"):
    pattern = f'{name_of_instruction}:.*?(?=Query:)'
    return re.sub(pattern, replace_with, text, flags=re.DOTALL)

def compute_results(model, hook, args, paths, tokenizer, max_index=None, batch_size=16):
    try:
        hook.remove()
    except:
        pass

    name_of_instruction = "Instruction" if args.context_weight_format == "instruction" else "Context weight"
    results = {}
    source_prompt, target_prompt, source_tokens, target_tokens, source_label_index, target_label_index, source_attn_mask, target_attn_mask = collect_data(args, paths, tokenizer, device)
    target_prompt.to_csv("target_prompt_before_clean.csv")

    cleaned_target_texts = [remove_instruction(text, name_of_instruction=name_of_instruction, replace_with=f"Instruction: {get_one_word_instruction(args)}") for text in target_prompt.text.tolist()]
    cleaned_target_tokens = tokenizer(cleaned_target_texts, padding=True, truncation=True, return_tensors="pt")
    cleaned_target_attn_mask = cleaned_target_tokens.attention_mask
    cleaned_target_tokens = cleaned_target_tokens.input_ids
    cleaned_target_texts_noinstr = [remove_instruction(text, name_of_instruction=name_of_instruction, replace_with="") for text in target_prompt.text.tolist()]
    cleaned_target_tokens_noinstr = tokenizer(cleaned_target_texts_noinstr, padding=True, truncation=True, return_tensors="pt")
    torch.save(cleaned_target_tokens_noinstr, "cleaned_target_tokens_noinstr.pt")
    cleaned_target_attn_mask_noinstr = cleaned_target_tokens_noinstr.attention_mask
    cleaned_target_tokens_noinstr = cleaned_target_tokens_noinstr.input_ids

    if args.context_weight_format == "instruction":
        target_texts_withoneword = [text.replace(f"{name_of_instruction}:", f"{name_of_instruction}: {get_one_word_instruction(args)}") for text in target_prompt.text.tolist()]
    else:
        target_texts_withoneword = [text.replace(f"{name_of_instruction}:", f"Instruction: {get_one_word_instruction(args)}\n{name_of_instruction}:") for text in target_prompt.text.tolist()]
    target_tokens_withoneword = tokenizer(target_texts_withoneword, padding=True, truncation=True, return_tensors="pt")
    torch.save(target_tokens_withoneword, "target_tokens_withoneword.pt")
    target_attn_mask_withoneword = target_tokens_withoneword.attention_mask
    target_tokens_withoneword = target_tokens_withoneword.input_ids

    values = torch.tensor(target_prompt["weight_context"] == 1.0)

    test_cases = [
        ("no_instruction", cleaned_target_tokens_noinstr.to(device), cleaned_target_attn_mask_noinstr.to(device)),
        ("baseline", target_tokens, target_attn_mask),
        # ("with_instruction", target_tokens, target_attn_mask),
        # ("against_instruction", source_tokens, source_attn_mask),
        # ("one_word_instruction", target_tokens_withoneword.to(device), target_attn_mask_withoneword.to(device)),
    ]

    if args.shots == 0 and not args.finetuned:
        test_cases =  [
            ("one_word", cleaned_target_tokens.to(device), cleaned_target_attn_mask.to(device)),
            ("baseline_one_word_instruction", target_tokens_withoneword.to(device), target_attn_mask_withoneword.to(device)),
        ] + test_cases
    for name, tokens, attn_mask in test_cases:
        if "baseline" not in name:
            hook.attach(model)
            
        print("Batch size:", batch_size)
        torch.save(tokens, f"{name}_tokens.pt")
        acc = iia_with_hook(model, args.eval_dataset, hook, tokens, attn_mask, values, target_prompt["answer"], tokenizer, batch_size=batch_size, max_index=max_index)
        results[name] = acc
        if "baseline" not in name:
            hook.remove()

    print("Accuracy:", {k: v["accuracy"] for k, v in results.items()})
    print("Pair accuracy:", {k: v["paired_accuracy"] for k, v in results.items()})
    return results


def run_experiment(type, args, paths, hook, result_dir, max_index=100, overwrite=False):
    experiment_name = f"{type}_{'ft_' if args.finetuned else 'zs_' if args.shots == 0 else 'fs_'}{args.context_weight_format}"
    print(f"Running {type} with {experiment_name}")
    if os.path.exists(f"{result_dir}/{experiment_name}.csv") and not overwrite:
        print(f"Skipping {type} because it already exists")
        return
    model, tokenizer = load_model_and_tokenizer_from_args(paths, args)
    res = compute_results(model, hook, args, paths, tokenizer, batch_size=32 if args.shots == 0 else (12 if "gemma" in args.model_id else 32), max_index=max_index)
    pd.DataFrame(res).to_csv(f"{result_dir}/{experiment_name}.csv")


MODELS = {
    "Meta-Llama-3.1-8B-Instruct": {
        "base": "Meta-Llama-3.1-8B",
        "instruct": "Meta-Llama-3.1-8B-Instruct",
        "model-store": "/dlabscratch1/public/llm_weights/llama3.1_hf/",
    },
    "gemma-2-9b-it": {
        "base": "gemma-2-9b",
        "instruct": "gemma-2-9b-it",
        "model-store": "/dlabscratch1/public/llm_weights/gemma_hf/",
    },
    "Mistral-7B-Instruct-v0.3": {
        "base": "Mistral-7B-v0.3",
        "instruct": "Mistral-7B-Instruct-v0.3",
        "model-store": "/dlabscratch1/public/llm_weights/mistral_hf/",
    },#/dlabscratch1/public/llm_weights/mistral_hf/Mistral-7B-v0.3
    
}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--proj", type=str, required=True)
    parser.add_argument("--devrun", action="store_true")
    parser.add_argument("--prior-value", type=float, required=True, default=-6.0)
    parser.add_argument("--dataset", type=str, required=True, default="BaseFakepedia")
    parser.add_argument("--subsplit", type=str, required=True, default="nodup_relpid")
    parser.add_argument("--context-value", type=float, required=True, default=-6.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eval-ft", action="store_true")

    cli_args = parser.parse_args()

    PATHS, args = get_decoding_args(
        cwf="instruction",
        model_id=cli_args.model_id,
        model_store=MODELS[cli_args.model_id]["model-store"],
        finetuned=True,
        load_in_4bit=True,
    )

    model, tokenizer = load_model_and_tokenizer_from_args(PATHS, args)

    layer = int(cli_args.proj.split("-")[-1][1:].split(".")[0])
    print(f"Using layer {layer}")

    hidden_size = model.config.hidden_size

    proj = LowRankRotatedSpaceIntervention(embed_dim=hidden_size, low_rank_dimension=1)
    proj.load_state_dict(state_dict=torch.load(cli_args.proj))
    proj.to(device)

    hook = CtxPriorHook(proj, layer, context_value=cli_args.context_value, prior_value=cli_args.prior_value)

    result_dir = f"analysis/results_das_dev/{args.model_id}/{hook.prior_value}_{hook.context_value}/{cli_args.dataset}-sp_{cli_args.subsplit}"
    os.makedirs(result_dir, exist_ok=True)

    base_args = [
        "--shots", "0",
        "--no-filtering",
        "--eval-dataset", cli_args.dataset,
        "--eval-subsplit", cli_args.subsplit,
        "--finetune-seed", "3",
        "--load-4bit"
    ]

    experiments = [
        ("base", base_args + ["--no-filtering", "--shots", "10", "--model-id", MODELS[cli_args.model_id]["base"], "--context-weight-format", "instruction", "--model-store", MODELS[cli_args.model_id]["model-store"]]),
        # ("instruct", base_args + ["--no-filtering", "--finetuned", "--shots", "0", "--model-id", MODELS[cli_args.model_id]["instruct"], "--context-weight-format", "instruction", "--model-store", MODELS[cli_args.model_id]["model-store"]]),
        # ("instruct", base_args + ["--no-filtering", "--shots", "0", "--model-id", MODELS[cli_args.model_id]["instruct"], "--context-weight-format", "instruction", "--model-store", MODELS[cli_args.model_id]["model-store"]]),
        
        # ("base", base_args + ["--no-filtering", "--shots", "0", "--model-id", MODELS[cli_args.model_id]["base"], "--context-weight-format", "instruction", "--model-store", MODELS[cli_args.model_id]["model-store"]]),
        #("base", base_args + ["--no-filtering", "--shots", "10", "--model-id", MODELS[cli_args.model_id]["base"], "--context-weight-format", "float", "--model-store", MODELS[cli_args.model_id]["model-store"]]),
        
        
        # Instruct model experiments
        
        
        # ("instruct", base_args + ["--no-filtering", "--shots", "0", "--model-id", MODELS[cli_args.model_id]["instruct"], "--context-weight-format", "float", "--model-store", MODELS[cli_args.model_id]["model-store"]]),
        # ("instruct", base_args + ["--no-filtering", "--shots", "0", "--model-id", MODELS[cli_args.model_id]["instruct"], "--context-weight-format", "float", "--model-store", MODELS[cli_args.model_id]["model-store"]]),
        
        
        # ("base", base_args + ["--no-filtering", "--shots", "10", "--model-id", MODELS[cli_args.model_id]["base"], "--context-weight-format", "float", "--model-store", MODELS[cli_args.model_id]["model-store"]]),



        # (base_args + ["--no-filtering", "--model-id", MODELS[cli_args.model_id]["base"], "--context-weight-format", "instruction", "--model-store", MODELS[cli_args.model_id]["model-store"]]),
        # ("base", base_args + ["--no-filtering", "--shots", "10", "--model-id", MODELS[cli_args.model_id]["base"], "--context-weight-format", "float", "--model-store", MODELS[cli_args.model_id]["model-store"]]),
        # ("instruct", base_args + ["--no-filtering", "--shots", "0", "--model-id", MODELS[cli_args.model_id]["instruct"], "--context-weight-format", "float", "--model-store", MODELS[cli_args.model_id]["model-store"]]),
        
        
        
        
        # # ("instruct", base_args + ["--no-filtering", "--model-id", MODELS[cli_args.model_id]["finetuned"], "--context-weight-format", "float", "--finetuned", "--model-store", MODELS[cli_args.model_id]["model-store"]]),
        # ("instruct", base_args + ["--no-filtering", "--shots", "10", "--model-id", MODELS[cli_args.model_id]["instruct"], "--context-weight-format", "instruction", "--model-store", MODELS[cli_args.model_id]["model-store"]]),
        # ("instruct", base_args + ["--no-filtering", "--shots", "10", "--model-id", MODELS[cli_args.model_id]["instruct"], "--context-weight-format", "float", "--model-store", MODELS[cli_args.model_id]["model-store"]]),


        # # Base model experiments
        # # (base_args + ["--no-filtering", "--model-id", MODELS[cli_args.model_id]["base"], "--context-weight-format", "float", "--model-store", MODELS[cli_args.model_id]["model-store"]]),
    ]

    if cli_args.eval_ft:
        experiments.append(("base", base_args + ["--no-filtering", "--shots", "0", "--finetuned", "--model-id", MODELS[cli_args.model_id]["base"], "--context-weight-format", "instruction", "--model-store", MODELS[cli_args.model_id]["model-store"]]))
        experiments.append(("base", base_args + ["--no-filtering", "--shots", "0", "--finetuned", "--model-id", MODELS[cli_args.model_id]["base"], "--context-weight-format", "float", "--model-store", MODELS[cli_args.model_id]["model-store"]]))
        

    parser = get_default_parser()
    for type, exp_args in experiments:
        args = parser.parse_args(exp_args)
        paths = paths_from_args(args)
        print(args)
        run_experiment(type, args, paths, hook, result_dir, max_index=100 if cli_args.devrun else None, overwrite=cli_args.overwrite)

if __name__ == "__main__":
    main()
