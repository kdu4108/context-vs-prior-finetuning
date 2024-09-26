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
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import lightning.pytorch as pl
import argparse

from nnpatch import activation_patch, attribution_patch, Site, Sites, activation_zero_patch, attribution_zero_patch
from nnpatch.site import HeadSite, MultiSite
from nnpatch.api.llama import Llama3

from analysis.circuit_utils.visualisation import *
from analysis.circuit_utils.model import *
from analysis.circuit_utils.validation import *
from analysis.circuit_utils.few_shot import *
from analysis.circuit_utils.utils import get_default_parser, paths_from_args, load_model_and_tokenizer_from_args, collect_data, batch_act_patch
from model_utils.utils import construct_query_with_demonstrations, MODEL_ID_TO_TEMPLATES_DICT

device = "cuda:0"


def get_file(args):
    file_list = os.listdir(args.output_dir)
    run_name = f"last_{'zero_' if args.zero else ''}{'ft' if args.finetuned else f'fs{args.shots}'}_i{args.dataset_index}{'_' if len(args.name) else ''}{args.name}_cwf-{args.context_weight_format}_n"
    suffix = ""
    if args.context_info_flow:
        suffix += "_cif"
    if args.context_to_prior:
        suffix += "_ctp"
    if args.prior_to_context:
        suffix += "_ptc"
    if args.prior_info_flow:
        suffix += "_pif"
    if len(args.name):
        suffix += "_" + args.name
    
    suffix += f"_heads{args.source_heads}.pt"

    candidates = [f for f in file_list if f.startswith(run_name) and f.endswith(suffix)]
    if len(candidates) > 1:
        logger.warning(f"Multiple files found for {run_name} {suffix}")
        logger.warning(candidates)
        candidates = sorted(candidates)
        logger.warning(f"Using {candidates[0]}")
    elif len(candidates) == 0:
        logger.warning(f"No files found for {run_name} {suffix}")
        sys.exit(1)
    return candidates[0], run_name, suffix

        
def multihead_patch(args, PATHS, file, model, tokenizer):
    tokenizer.padding_side = "left"
    
    clean_prompt, corrupted_prompt, clean_tokens, corrupted_tokens, correct_index, incorrect_index, attention_mask_clean, attention_mask_corrupted = collect_data(args, PATHS, tokenizer, device)
        
    nnmodel = NNsight(model)

    N_LAYERS = model.config.num_hidden_layers

    

    site_names = args.heads

    heads = {}
    logger.info(f"Loading ACT PATCH results from {file}")
    act_patch_results = torch.load(os.path.join(args.output_dir, file))
    act_patch = act_patch_results["activation_patching"]
    
    all_results = {}
    sites_list = []
    for head in args.heads:
        heads[head] = act_patch[head][:, -1, :]
        if args.layer_range[0] != 0 or args.layer_range[1] != -1:
            logger.info(f"Only using layers {args.layer_range[0]} to {args.layer_range[1]}")
            heads[head][0:args.layer_range[0]] = 0
            heads[head][args.layer_range[1]:] = 0
            if args.topk == -1:
                heads[head][args.layer_range[0]:args.layer_range[1]] = 1
                
        if args.layers is not None:
            tmp = torch.zeros_like(heads[head])
            if args.topk == -1:
                tmp[args.layers] = 1
            else:
                tmp[args.layers] = heads[head][args.layers]
            heads[head] = tmp
            
        if args.topk != -1:
            logger.info(f"Using topk {args.topk} for head {head}")
            topk = args.topk
        else:
            logger.info(f"Using all specified layers for head {head}")
            # set to num elements
            topk = torch.numel(heads[head])
            
        heads[head] = heads[head].flatten().topk(topk)
        # filter out zero values
        topk_is_non_zero = ~torch.isclose(heads[head].values, torch.tensor(0.0), atol=1e-6)
        if topk_is_non_zero.sum() < args.topk:
            logger.warning(f"Only {topk_is_non_zero.sum()} non-zero values found for head {head}, reducing topk to this value")

        heads[head] = heads[head].indices[topk_is_non_zero].tolist()
        heads[head] = [divmod(el, 32) for el in heads[head]]
        
        
        sites = [
            Site.get_site(
                Llama3, head, l, h, torch.tensor([-1]), "default"
            ) for l, h in heads[head]
        ]
        print(sites)
        multi_site = MultiSite(sites)
        sites_list.append(multi_site)
    
    
    sites = Sites.from_list_of_sites(sites_list)
    
    all_results = batch_act_patch(args, nnmodel, sites, clean_tokens, corrupted_tokens, correct_index, incorrect_index, attention_mask_clean, attention_mask_corrupted, force_model_confidence=args.force_model_confidence)

    act_patch_sum = {
        head: act_patch[head][:, -1, :][torch.tensor(heads[head])[:, 0], torch.tensor(heads[head])[:, 1]].sum() for head in args.heads
    }
    return all_results, act_patch_sum   

def multihead_patch_greedy(args, PATHS, file, model, tokenizer):
    tokenizer.padding_side = "left"
    
    clean_prompt, corrupted_prompt, clean_tokens, corrupted_tokens, correct_index, incorrect_index, attention_mask_clean, attention_mask_corrupted = collect_data(args, PATHS, tokenizer, device)
    
    nnmodel = NNsight(model)

    N_LAYERS = model.config.num_hidden_layers

    

    site_names = args.heads

    logger.info(f"Loading ACT PATCH results from {file}")
    act_patch_results = torch.load(os.path.join(args.output_dir, file))
    act_patch = act_patch_results["activation_patching"]
    
    for head in args.heads:
        current_score = act_patch[head][:, -1, :].flatten()
        candidates = torch.arange(torch.numel(current_score)).tolist()
        all_results = {}
        sites_list = []
        heads = []
        scores = []
        while len(heads) < args.topk:
            # top head
            tophead = current_score.topk(1).indices.item()
            tophead_flat = candidates[tophead]
            tophead = tuple(divmod(tophead_flat, 32))
            logger.info(f"Adding head {tophead}")
            heads.append(tophead)
            logger.info(f"Current heads {heads}")
            logger.info(f"Current score {current_score[tophead_flat]}")
            scores.append(current_score[tophead_flat].item())

            sites_list += [
                Site.get_site(
                    Llama3, head, tophead[0], tophead[1], torch.tensor([-1]), "default"
                )
            ]
            
            search_sites = []
            candidates = []
            for c in range(N_LAYERS*32):
                l, h = divmod(c, 32)
                if (l, h) in heads:
                    continue
                candidates.append(c)
                search_sites.append(
                    MultiSite(sites_list + [
                        Site.get_site(
                            Llama3, head, l, h, torch.tensor([-1]), "default"
                        )
                    ])
                )
            
            sites = Sites.from_list_of_sites(search_sites)
            # run act patch
            all_results = batch_act_patch(args, nnmodel, sites, clean_tokens, corrupted_tokens, correct_index, incorrect_index, attention_mask_clean, attention_mask_corrupted, force_model_confidence=args.force_model_confidence)
            print(all_results)
            current_score = all_results[f"MultiSite(('{head}',))"].flatten()
            
            
    return heads, scores

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
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--new-few-shots", default=None, type=int)
    parser.add_argument("--zero", action="store_true")
    parser.add_argument("--no-filtering", action="store_true")
    parser.add_argument("--batch-size", default=-1, type=int)
    parser.add_argument("--force-model-confidence", action="store_true")
    parser.add_argument("--heads", default=["o"], nargs="+")
    parser.add_argument("--source-heads", default=["o", "q"], nargs="+")
    parser.add_argument("--layer-range", "-LR", default=[0, -1], nargs=2, type=int)
    parser.add_argument("--layers", default=None, nargs="+", type=int)
    parser.add_argument("--greedy", action="store_true")


    
    args = parser.parse_args()
    logger.info(args)
    
    file, run_name, suffix = get_file(args)
    
    assert args.layers is None or args.layer_range == [0, -1], "Cannot specify both layers and layer range"
    assert not (args.context_info_flow and (args.context_to_prior or args.prior_to_context)), "Cannot have both context info flow and context to prior"
    assert not (args.context_to_prior and args.prior_to_context), "Cannot have both context to prior and prior to context"
    assert args.dataset_index % 2 == 0, "Index must be even"
    assert args.n_samples % 2 == 0 or args.n_samples == -1, "Number of samples must be even"
    PATHS = paths_from_args(args)
    
    logger.info(PATHS)
    model, tokenizer = load_model_and_tokenizer_from_args(PATHS, args)

    if args.greedy:
        heads, scores  = multihead_patch_greedy(args, PATHS, file, model, tokenizer)
    else:
        all_results, act_patch_sum = multihead_patch(args, PATHS, file,  model, tokenizer)
    
    torch.save({"heads": heads, "scores": scores}, os.path.join(args.output_dir, f"multihead_{'greedy_' if args.greedy else ''}{run_name}{suffix}"))