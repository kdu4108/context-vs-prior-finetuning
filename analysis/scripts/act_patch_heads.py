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
from analysis.circuit_utils.utils import get_default_parser, paths_from_args, load_model_and_tokenizer_from_args, filter_for_true_pairs, collect_data, batch_act_patch, combine_batches
from main import load_model_and_tokenizer
from model_utils.utils import construct_query_with_demonstrations, MODEL_ID_TO_TEMPLATES_DICT

device = "cuda:0"

if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("--dataset-index", default=0, type=int)
    parser.add_argument("--n-samples", default=-1, type=int)
    parser.add_argument("--output-dir", default="patching_results")
    parser.add_argument("--name", default="")
    parser.add_argument("--context-info-flow", action="store_true")
    parser.add_argument("--prior-info-flow", action="store_true")
    parser.add_argument("--context-to-prior", action="store_true")
    parser.add_argument("--prior-to-context", action="store_true")
    parser.add_argument("--topk-search", default=256, type=int)
    parser.add_argument("--new-few-shots", default=None, type=int)
    parser.add_argument("--zero", action="store_true")
    parser.add_argument("--no-filtering", action="store_true")
    parser.add_argument("--batch-size", default=-1, type=int)
    parser.add_argument("--force_model_confidence", action="store_true")
    parser.add_argument("--heads", default=["o"], nargs="+")
    parser.add_argument("--overwrite", "-O", action="store_true")


    args = parser.parse_args()
    logger.info(args)
    
    assert not (args.context_info_flow and (args.context_to_prior or args.prior_to_context)), "Cannot have both context info flow and context to prior"
    assert not (args.context_to_prior and args.prior_to_context), "Cannot have both context to prior and prior to context"
    assert args.dataset_index % 2 == 0, "Index must be even"
    assert args.n_samples % 2 == 0 or args.n_samples == -1, "Number of samples must be even"
    PATHS = paths_from_args(args)
    
    logger.info(PATHS)

    model, tokenizer = load_model_and_tokenizer_from_args(PATHS, args)    
    clean_prompt, corrupted_prompt, clean_tokens, corrupted_tokens, correct_index, incorrect_index, attention_mask_clean, attention_mask_corrupted = collect_data(args, PATHS, tokenizer, device)

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
    
    run_name += f"_heads{args.heads}"
    
    if os.path.exists(os.path.join(args.output_dir, run_name + ".pt")) and not args.overwrite:
        logger.info(f"File {run_name} already exists, skipping")
        sys.exit(0)

    nnmodel = NNsight(model)
    
    site_names = args.heads

    sites = Sites(site_names=site_names, seq_pos=None, seq_pos_type="last")
    
    top_attn_heads = {head_name: [] for head_name in args.heads}
    attr_all = {head_name: [] for head_name in args.heads}
    
    if args.batch_size == -1:
        args.batch_size = len(clean_tokens)
        logger.info(f"Setting batch size to data length :{args.batch_size}")
        
    attr_batch_size = max(1, args.batch_size // 6) # requires backward pass -> lower batch size
    logger.info(f"ATTR PATCH with batch size {attr_batch_size}")
    attr_out = []
    for i in range(0, len(correct_index), attr_batch_size):
        HeadSite.reset()
        if args.zero:
            out = attribution_zero_patch(nnmodel, Llama3, sites, corrupted_tokens[i:i+attr_batch_size], incorrect_index[i:i+attr_batch_size], target_attention_mask=attention_mask_corrupted[i:i+attr_batch_size])
        else:
            out = attribution_patch(nnmodel, Llama3, sites, clean_tokens[i:i+attr_batch_size], corrupted_tokens[i:i+attr_batch_size], correct_index[i:i+attr_batch_size], incorrect_index[i:i+attr_batch_size], source_attention_mask=attention_mask_clean[i:i+attr_batch_size], target_attention_mask=attention_mask_corrupted[i:i+attr_batch_size], force_model_confidence=args.force_model_confidence)

        attr_out.append((out, len(corrupted_tokens[i:i+attr_batch_size])))
        
    torch.save(attr_out, "attr_out.pt")
    
    attr_all = combine_batches(attr_out)
    for head_name in args.heads:
        top_attn_heads[head_name] = attr_all[head_name].flatten().abs().topk(args.topk_search).indices.tolist()
        
    # create site for topk heads  
    sites_list = []
    for head_name in args.heads:
        attn_heads = [divmod(el, 32) for el in top_attn_heads[head_name]]
    
    
        seq_len = clean_tokens.shape[1]
        sites_list.extend([
            {
                "site_name": head_name,
                "layer": l,
                "head": h,
                "seq_pos":  torch.tensor([-1])
            } for l, h in attn_heads
        ])
        
    logger.info("Identified {} unique heads per head site".format(len(top_attn_heads[head_name])))
    logger.info(top_attn_heads)
    verification_sites = Sites.from_list_of_dicts(sites_list)

    torch.cuda.empty_cache()
    nnmodel.eval()

    act_out_combined = batch_act_patch(args, nnmodel, sites, clean_tokens, corrupted_tokens, correct_index, incorrect_index, attention_mask_clean, attention_mask_corrupted, force_model_confidence=args.force_model_confidence)
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving to {os.path.join(args.output_dir, run_name + '.pt')}")    
    torch.save({
        'activation_patching': act_out_combined,
        'attribution_patching': attr_all,
        'prompt': clean_prompt,
        'corrupted_prompt': corrupted_prompt,
        }, os.path.join(args.output_dir, run_name + ".pt")
    )    