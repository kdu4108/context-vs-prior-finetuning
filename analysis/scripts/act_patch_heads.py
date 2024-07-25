import sys
sys.path.append("/dlabscratch1/jminder/repositories/context-vs-prior-finetuning")
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from nnsight import NNsight
from nnsight.models.LanguageModel import LanguageModel
import torch
import pandas as pd
from loguru import logger
import os
from transformer_lens import HookedTransformer
import numpy as np
from tqdm import tqdm, trange
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import torch.nn as nn
import circuitsvis as cv
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import lightning.pytorch as pl
import argparse

from nnpatch import activation_patch, attribution_patch, Site, Sites, activation_zero_patch, attribution_zero_patch
from nnpatch.site import HeadSite
from nnpatch.api.llama import Llama3

from analysis.circuit_utils.visualisation import *
from analysis.circuit_utils.model import *
from analysis.circuit_utils.validation import *
from analysis.circuit_utils.few_shot import *
from analysis.circuit_utils.utils import get_default_parser, paths_from_args, load_model_and_tokenizer_from_args, filter_for_true_pairs

from main import load_model_and_tokenizer
from model_utils.utils import construct_query_with_demonstrations, MODEL_ID_TO_TEMPLATES_DICT

device = "cuda:0"


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("--dataset-index", default=0, type=int)
    parser.add_argument("--n-samples", default=-1, type=int)
    parser.add_argument("--output_dir", default="patching_results")
    parser.add_argument("--name", default="")
    parser.add_argument("--context-info-flow", action="store_true")
    parser.add_argument("--prior-info-flow", action="store_true")
    parser.add_argument("--context-to-prior", action="store_true")
    parser.add_argument("--prior-to-context", action="store_true")
    parser.add_argument("--topk-search", default=200, type=int)
    parser.add_argument("--new-few-shots", default=None, type=int)
    parser.add_argument("--zero", action="store_true")
    parser.add_argument("--no-filtering", action="store_true")
    args = parser.parse_args()
    logger.info(args)
    
    assert not (args.context_info_flow and (args.context_to_prior or args.prior_to_context)), "Cannot have both context info flow and context to prior"
    assert not (args.context_to_prior and args.prior_to_context), "Cannot have both context to prior and prior to context"
    assert args.dataset_index % 2 == 0, "Index must be even"
    assert args.n_samples % 2 == 0 or args.n_samples == -1, "Number of samples must be even"
    PATHS = paths_from_args(args)
    
    logger.info(PATHS)
    model, tokenizer = load_model_and_tokenizer_from_args(PATHS, args)

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
        
    nnmodel = NNsight(model)

    if args.context_info_flow:
        clean_prompt = test_data[test_data.weight_context == 1.0]
    elif args.prior_info_flow:
        clean_prompt = test_data[test_data.weight_context == 0.0]
        if not args.no_filtering:
            clean_prompt = clean_prompt.groupby("answer").first().reset_index()
    else:
        clean_prompt = test_data.iloc[args.dataset_index: args.dataset_index+ args.n_samples]
    
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
    
    print(clean_prompt.answer)
    print(corrupted_prompt.answer)
    correct_index = torch.tensor([tokenizer.encode("\n" + a)[1] for a in clean_prompt.answer]).to(device)
    incorrect_index = torch.tensor([tokenizer.encode("\n" + a)[1] for a in corrupted_prompt.answer]).to(device)

    clean_prompt = clean_prompt.text.tolist()
    corrupted_prompt = corrupted_prompt.text.tolist()

    N_LAYERS = model.config.num_hidden_layers

    tokenizer.padding_side = "left"
    
    clean_tokens = tokenizer(clean_prompt, return_tensors="pt", padding=True)
    attention_mask_clean = clean_tokens["attention_mask"].to(device)
    corrupted_tokens = tokenizer(corrupted_prompt, return_tensors="pt", padding=True)
    attention_mask_corrupted = corrupted_tokens["attention_mask"].to(device)
    clean_tokens = clean_tokens["input_ids"].to(device)
    corrupted_tokens = corrupted_tokens["input_ids"].to(device)
    
    site_names = ["o"]

    sites = Sites(site_names=site_names, seq_pos=None, seq_pos_type="last")
    
    print(correct_index.shape, incorrect_index.shape)
    top_attn_heads = []
    attr_all = []
    for i in range(len(correct_index)):
        HeadSite.reset()
        if args.zero:
            out = attribution_zero_patch(nnmodel, Llama3, sites, corrupted_tokens[i:i+1], incorrect_index[i:i+1], target_attention_mask=attention_mask_corrupted[i:i+1])
        else:
            out = attribution_patch(nnmodel, Llama3, sites, clean_tokens[i:i+1], corrupted_tokens[i:i+1], correct_index[i:i+1], incorrect_index[i:i+1], source_attention_mask=attention_mask_clean[i:i+1], target_attention_mask=attention_mask_corrupted[i:i+1])
        
        attr_o_ldvs = out["o"]  
        attr_all.append(attr_o_ldvs)   
        
    attr_all = torch.stack(attr_all)
    attr_all = attr_all.mean(dim=0)
    top_attn_heads = attr_all.flatten().abs().topk(args.topk_search).indices.tolist()

    # find topk heads   
    logger.info(top_attn_heads)
    logger.info("Identified {} unique heads".format(len(top_attn_heads)))
    attn_heads = [divmod(el, 32) for el in top_attn_heads]
    top_attn_heads = attr_all.flatten().abs().topk(args.topk_search).indices.tolist()
    
    
    seq_len = clean_tokens.shape[1]
    sites_list = [
        {
            "site_name": "o",
            "layer": l,
            "head": h,
            "seq_pos":  torch.tensor([-1])
        } for l, h in attn_heads
    ]
    
    verification_sites = Sites.from_list(sites_list)
    
    torch.cuda.empty_cache()
    nnmodel.eval()
    logger.info("ACT PATCH")
    if args.zero:
        act_out = activation_zero_patch(nnmodel, Llama3, verification_sites, corrupted_tokens, incorrect_index, target_attention_mask=attention_mask_corrupted)
    else:
        act_out = activation_patch(nnmodel, Llama3, verification_sites, clean_tokens, corrupted_tokens, correct_index, incorrect_index, source_attention_mask=attention_mask_clean, target_attention_mask=attention_mask_corrupted)
    
    
    run_name = f"last_{'zero_' if args.zero else ''}{'ft' if args.finetuned else f'fs{args.shots}'}_i{args.dataset_index}{'_' if len(args.name) else ''}{args.name}_cwf-{args.context_weight_format}_n{args.n_samples}"

    if args.context_info_flow:
        run_name += "_cif"
    if args.context_to_prior:
        run_name += "_ctp"
    if args.prior_to_context:
        run_name += "_ptc"
    if args.prior_info_flow:
        run_name += "_pif"
    if len(args.name):
        run_name += "_" + args.name
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving to {os.path.join(args.output_dir, run_name + '.pt')}")    
    torch.save({
        'activation_patching': act_out,
        'attribution_patching': attr_all,
        'prompt': clean_prompt,
        'corrupted_prompt': corrupted_prompt,
        }, os.path.join(args.output_dir, run_name + ".pt")
    )    