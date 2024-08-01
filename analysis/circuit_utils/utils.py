import argparse
import os
import torch
from loguru import logger
import pandas as pd

from nnpatch import activation_patch, attribution_patch, Site, Sites, activation_zero_patch, attribution_zero_patch
from nnpatch.api.llama import Llama3
from nnpatch.site import HeadSite, MultiSite

from model_utils.utils import load_model_and_tokenizer
from model_utils.utils import construct_query_with_demonstrations, MODEL_ID_TO_TEMPLATES_DICT

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-store", default="/dlabscratch1/public/llm_weights/llama3.1_hf/")
    parser.add_argument("--model-id", default="Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--context-weight-format", "-CWF", default="float", choices=["float", "instruction"])
    parser.add_argument("--finetune-configuration", "-FTC", default="peftq_proj_k_proj_v_proj_o_proj")
    parser.add_argument("--finetune-training-args", "-FTCA", default="bs8-ga2", type=str)
    parser.add_argument("--finetune-seed", "-FTS", default=3, type=int)
    parser.add_argument("--finetune-training-samples", "-FTTS", default=2048, type=int)
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--shots", default=10, type=int)
    parser.add_argument("--dataset", "-DS", type=str, help="Name of the dataset class", default="BaseFakepedia")
    parser.add_argument(
        "-SP",
        "--subsplit",
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
    return parser
    
def paths_from_args(args):
    BASE_MODEL = os.path.join(args.model_store, args.model_id)
    if args.finetuned:
        args.shots = 0
        MODEL_NAME = f"{args.model_id}-{args.finetune_configuration}-{args.finetune_training_args}-cwf_{args.context_weight_format}"
    else:
        MODEL_NAME = f"{args.model_id}-{args.finetune_training_args}-NT-cwf_float"
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    DATAROOT = os.path.join(PROJECT_DIR, "data", args.dataset)
    TRAIN_DATA = os.path.join(DATAROOT, "splits", args.subsplit, "train.csv")
    
    DATASET_CONFIG_NAME = f"{args.dataset}_{args.subsplit}-ts{args.finetune_training_samples}"

    
    MERGED_MODEL = os.path.join(DATAROOT, DATASET_CONFIG_NAME, str(args.finetune_seed), "models", MODEL_NAME, "merged")
    PEFT_MODEL = os.path.join(DATAROOT, DATASET_CONFIG_NAME, str(args.finetune_seed), "models", MODEL_NAME, "model")
    VAL_DATA_ALL = os.path.join(DATAROOT, "splits", args.subsplit, "val.csv")
    TRAIN_DATA_ALL = os.path.join(DATAROOT, "splits", args.subsplit, "train.csv")
    
    RESULTS_DIR = os.path.join(DATAROOT, DATASET_CONFIG_NAME, str(args.finetune_seed), "models", MODEL_NAME,  "results", f"{args.dataset}-k{args.shots}-cwf_{args.context_weight_format}")
    FEW_SHOT_SAMPLE = os.path.join(RESULTS_DIR, "few_shot_sample.csv")
    TEST_DATA = os.path.join(RESULTS_DIR, "test.csv")
    
    return {
        "BASE_MODEL": BASE_MODEL,
        "MODEL_NAME": MODEL_NAME,
        "DATAROOT": DATAROOT,
        "TRAIN_DATA": TRAIN_DATA,
        "DATASET_CONFIG_NAME": DATASET_CONFIG_NAME,
        "PEFT_MODEL": PEFT_MODEL,
        "MERGED_MODEL": MERGED_MODEL,
        "VAL_DATA_ALL": VAL_DATA_ALL,
        "TRAIN_DATA_ALL": TRAIN_DATA_ALL,
        "RESULTS_DIR": RESULTS_DIR,
        "FEW_SHOT_SAMPLE": FEW_SHOT_SAMPLE,
        "TEST_DATA": TEST_DATA
    }
    

def load_model_and_tokenizer_from_args(paths, args):
    if args.finetuned:
        model, tokenizer = load_model_and_tokenizer(paths["MERGED_MODEL"], False, False, False, padding_side="left")
    else:
        model, tokenizer = load_model_and_tokenizer(paths["BASE_MODEL"], False, False, False, padding_side="left")
    return model, tokenizer


def filter_for_true_pairs(data):
    # recompute the is_correct
    data["is_correct"] = data.apply(lambda x: x["predictions"].startswith(x["answer"]), axis=1)
    # filter out the false samples (both odd and even indices need to be true)
    trues = data[data.is_correct]
    even_true = trues[trues.index % 2 == 0 ].index
    odd_true = trues[trues.index % 2 == 1 ].index

    true_indices = set(even_true) & set([i-1 for i in odd_true])
    true_indices = sorted(list(true_indices) + [i+1 for i in true_indices])
    data = data.iloc[true_indices]
    return data

def collect_data(args, PATHS, tokenizer, device):
    test_data = pd.read_csv(PATHS["TEST_DATA"])
    
    # filter test data for correct predictions
    n_before = len(test_data)
    if not args.no_filtering:
        test_data = filter_for_true_pairs(test_data)
        # reset index
        test_data.reset_index(drop=True, inplace=True)
        logger.info(f"Filtered {n_before - len(test_data)} samples")
    
    if not args.finetuned and args.new_few_shots is not None:
        logger.info("Loading Few Shot Sample")
        prompt_template_dict, response_template = MODEL_ID_TO_TEMPLATES_DICT[args.model_id]

        shot_sample = pd.read_csv(PATHS["FEW_SHOT_SAMPLE"])
        test_data.text = test_data.apply(
            lambda row: construct_query_with_demonstrations(
                prompt_template_dict=prompt_template_dict,
                response_template=response_template,
                val_context=row.context,
                val_query=row.query,
                context_weight=row.weight_context,
                context_weight_format=args.context_weight_format,
                demonstrations_df=shot_sample[:args.new_few_shots],
                do_eval=True,
                val_answer=row.answer
            )
        )
        

    if args.n_samples == -1:
        args.n_samples = len(test_data)  
        
    if args.context_info_flow:
        clean_prompt = test_data[test_data.weight_context == 1.0].reset_index(drop=True)
    elif args.prior_info_flow:
        clean_prompt = test_data[test_data.weight_context == 0.0]
        clean_prompt = clean_prompt.groupby("answer").first().reset_index()
    else:
        clean_prompt = test_data
        
    clean_prompt = clean_prompt.iloc[args.dataset_index: args.dataset_index+args.n_samples]
    
    if args.context_to_prior:
        corrupted_prompt = clean_prompt.copy()[clean_prompt.weight_context == 0.0]
        clean_prompt = clean_prompt[clean_prompt.weight_context == 1.0]
    elif args.prior_to_context:
        corrupted_prompt = clean_prompt.copy()[clean_prompt.weight_context == 1.0]
        clean_prompt = clean_prompt[clean_prompt.weight_context == 0.0]
    elif args.context_info_flow or args.prior_info_flow:
        clean_to_corrupted_index =  [(i + 1) % len(clean_prompt) for i in range(len(clean_prompt))]
        corrupted_prompt = clean_prompt.iloc[clean_to_corrupted_index]
    else:
        clean_to_corrupted_index =  [(i + 1) if (i % 2) == 0 else i-1 for i in range(len(clean_prompt))]
        corrupted_prompt = clean_prompt.iloc[clean_to_corrupted_index]
    
    if args.n_samples == -1:
        args.n_samples = len(clean_prompt)
    elif args.n_samples > len(clean_prompt):
        logger.warning(f"Only {len(clean_prompt)} samples available, reducing n_samples to this value")
        args.n_samples = len(clean_prompt)
        
    correct_index = torch.tensor([tokenizer.encode("\n" + a)[1] for a in clean_prompt.answer]).to(device)
    incorrect_index = torch.tensor([tokenizer.encode("\n" + a)[1] for a in corrupted_prompt.answer]).to(device)

    clean_data = clean_prompt
    corrupted_data = corrupted_prompt
    clean_prompt = clean_prompt.text.tolist()
    corrupted_prompt = corrupted_prompt.text.tolist()
    
    clean_tokens = tokenizer(clean_prompt, return_tensors="pt", padding=True)
    attention_mask_clean = clean_tokens["attention_mask"].to(device)
    corrupted_tokens = tokenizer(corrupted_prompt, return_tensors="pt", padding=True)
    attention_mask_corrupted = corrupted_tokens["attention_mask"].to(device)
    clean_tokens = clean_tokens["input_ids"].to(device)
    corrupted_tokens = corrupted_tokens["input_ids"].to(device)
    
    return clean_data, corrupted_data, clean_tokens, corrupted_tokens, correct_index, incorrect_index, attention_mask_clean, attention_mask_corrupted


def combine_batches(list_of_outs):
    combined = {}
    # combine batches
    total = 0
    for out_batch, batch_len in list_of_outs:
        for head_name, value in out_batch.items():
            if torch.any(value.isnan()):
                logger.warning(f"NaN in {head_name}")
                continue
            if head_name not in combined:
                combined[head_name] = value * batch_len
            else:
                combined[head_name] += value * batch_len
            total += batch_len
                
    for key in combined:
        combined[key] /= total
    return combined


def batch_act_patch(args, nnmodel, sites, clean_tokens, corrupted_tokens, correct_index, incorrect_index, attention_mask_clean, attention_mask_corrupted, force_model_confidence=False):
    if args.batch_size == -1:
        args.batch_size = len(clean_tokens)
    logger.info(f"ACT PATCH with batch size {args.batch_size}")
    results = []
    for i in range(0, len(clean_tokens), args.batch_size):
        HeadSite.reset()
        logger.info(f"Batch {i}")
        if args.zero:
            activation_zero_patch(nnmodel, Llama3, sites, corrupted_tokens[i:i+args.batch_size], incorrect_index[i:i+args.batch_size], target_attention_mask=attention_mask_corrupted[i:i+args.batch_size])
        else:
            out = activation_patch(nnmodel, Llama3, sites, clean_tokens[i:i+args.batch_size], corrupted_tokens[i:i+args.batch_size], correct_index[i:i+args.batch_size], incorrect_index[i:i+args.batch_size], source_attention_mask=attention_mask_clean[i:i+args.batch_size], target_attention_mask=attention_mask_corrupted[i:i+args.batch_size], force_model_confidence=args.force_model_confidence)
        results.append((out, len(clean_tokens[i:i+args.batch_size])))
        
    all_results = combine_batches(results)
    return all_results