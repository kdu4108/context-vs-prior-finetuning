import sys
sys.path.append(".")
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from functools import partial
import einops
from nnsight import NNsight
from nnsight.models.LanguageModel import LanguageModel
import torch
import pandas as pd
import os
from transformer_lens import HookedTransformer
import numpy as np
from tqdm.notebook import tqdm, trange
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import lightning.pytorch as pl
from nnsight import NNsight
from datasets import Dataset
import re
import random
from analysis.circuit_utils.visualisation import *
from analysis.circuit_utils.model import *
from analysis.circuit_utils.validation import *
from analysis.circuit_utils.decoding import get_decoding_args
from analysis.circuit_utils.utils import *

from main import load_model_and_tokenizer
from torch.nn import CrossEntropyLoss
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str, required=True)
parser.add_argument("--model-store", type=str, required=True)
parser.add_argument("--cwf", type=str, required=True)
parser.add_argument("--dataset", type=str, default="BaseFakepedia")
args = parser.parse_args()

PATHS, args = get_decoding_args(
    cwf=args.cwf,
    model_id=args.model_id,
    model_store=args.model_store,
    finetuned=True,
    dataset=args.dataset
)
print("Merging Models")
print(PATHS["PEFT_MODEL"])
model, tok = load_peft(PATHS["PEFT_MODEL"], PATHS["BASE_MODEL"])
print("Merging Models")
print(PATHS["MERGED_MODEL"])
merge_save_peft(model, tok, PATHS["MERGED_MODEL"])

