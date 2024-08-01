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
    parser.add_argument("--output_dir", default="patching_results")
    parser.add_argument("--topk-search", default=200, type=int)
    parser.add_argument("--zero", action="store_true")
    parser.add_argument("--batch-size", default=20, type=int)

    args = parser.parse_args()
    logger.info(args)
    

    PATHS = paths_from_args(args)
    
    logger.info(PATHS)
    model, tokenizer = load_model_and_tokenizer_from_args(PATHS, args)
    
    prompts = [
        "When John and Mary went to the shops, John gave the bag to",
        "When John and Mary went to the shops, Mary gave the bag to",
        "When Tom and James went to the park, James gave the ball to",
        "When Tom and James went to the park, Tom gave the ball to",
        "When Dan and Sid went to the shops, Sid gave an apple to",
        "When Dan and Sid went to the shops, Dan gave an apple to",
        "After Martin and Amy went to the park, Amy gave a drink to",
        "After Martin and Amy went to the park, Martin gave a drink to",
    ]

    answers = [
        (" Mary", " John"),
        (" John", " Mary"),
        (" Tom", " James"),
        (" James", " Tom"),
        (" Dan", " Sid"),
        (" Sid", " Dan"),
        (" Martin", " Amy"),
        (" Amy", " Martin"),
    ]

    tokenizer.padding_side = "left"
    clean_tokens = tokenizer(prompts, return_tensors="pt")
    attention_mask_clean = clean_tokens["attention_mask"]
    clean_tokens = clean_tokens["input_ids"]
    corrupted_tokens = clean_tokens[
        [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
    ]
    attention_mask_corrupted = attention_mask_clean[
        [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
    ]

    answer_token_indices = torch.tensor(
        [
            [tokenizer(answers[i][j])["input_ids"][0] for j in range(2)]
            for i in range(len(answers))
        ]
    )
    
    
    nnmodel = NNsight(model)

    N_LAYERS = model.config.num_hidden_layers

    
    site_names = ["o"]

    sites = Sites(site_names=site_names, seq_pos=None, seq_pos_type="last")
    
    correct_index = answer_token_indices[:, 0]
    incorrect_index = answer_token_indices[:, 1]
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
    if args.batch_size == -1:
        args.batch_size = len(clean_tokens)
        logger.info(f"Setting batch size to data length :{args.batch_size}")
    logger.info(f"ACT PATCH with batch size {args.batch_size}")
    act_out = []
    for i in range(0, len(clean_tokens), args.batch_size):
        HeadSite.reset()
        logger.info(f"Batch {i}")
        if args.zero:
            act_out_batch = activation_zero_patch(nnmodel, Llama3, verification_sites, corrupted_tokens[i:i+args.batch_size], incorrect_index[i:i+args.batch_size], target_attention_mask=attention_mask_corrupted[i:i+args.batch_size])   
        else:
            act_out_batch = activation_patch(nnmodel, Llama3, verification_sites, clean_tokens[i:i+args.batch_size], corrupted_tokens[i:i+args.batch_size], correct_index[i:i+args.batch_size], incorrect_index[i:i+args.batch_size], source_attention_mask=attention_mask_clean[i:i+args.batch_size], target_attention_mask=attention_mask_corrupted[i:i+args.batch_size])
        act_out.append((act_out_batch, len(corrupted_tokens[i:i+args.batch_size])))
    
    act_out_combined = {}
    # combine batches
    for act_out_batch, batch_len in act_out:
        for key, value in act_out_batch.items():
            if key not in act_out:
                act_out_combined[key] = value * batch_len
            else:
                act_out_combined[key] += value * batch_len
    for key in act_out_combined:
        act_out_combined[key] /= len(clean_tokens)
    
    run_name = f"ioi"

    
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving to {os.path.join(args.output_dir, run_name + '.pt')}")    
    torch.save({
        'activation_patching': act_out_combined,
        'attribution_patching': attr_all,
        'prompt': prompts,
        'corrupted_prompt': None,
        }, os.path.join(args.output_dir, run_name + ".pt")
    )    