from pyvene import (
    IntervenableModel,
    LowRankRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
import pandas as pd
import re
import os
from torch.nn import CrossEntropyLoss
import torch
from tqdm import trange
from datasets import Dataset
from main import load_model_and_tokenizer
from model_utils.utils import is_response_correct, MODEL_ID_TO_TEMPLATES_DICT, construct_query_with_demonstrations 
import random

from analysis.circuit_utils.utils import encode_answer

def load_map_training_data(tokenizer, args, PATHS):
    train_data = pd.read_csv(PATHS["TRAIN_DATA"])
    val_data = pd.read_csv(PATHS["TRAIN_DATA"].split("train")[0] + "val.csv")
    train_data.head()

    prompt_template_dict, response_template = MODEL_ID_TO_TEMPLATES_DICT[args.model_id]
    train_data["text"] = train_data.apply(lambda row: construct_query_with_demonstrations(
                    prompt_template_dict=prompt_template_dict,
                    val_context=row.context,
                    eos_token=tokenizer.eos_token,
                    val_query=row.query,
                    context_weight=row.weight_context,
                    context_weight_format=args.context_weight_format,
                    demonstrations_df=pd.DataFrame(),
                    do_eval=True,
                    val_answer=row.answer
                ), axis=1)
    train_data["text_inverted"] = train_data.apply(lambda row: construct_query_with_demonstrations(
                prompt_template_dict=prompt_template_dict,
                val_context=row.context,
                eos_token=tokenizer.eos_token,
                val_query=row.query,
                context_weight=abs(row.weight_context - 1),
                context_weight_format=args.context_weight_format,
                demonstrations_df=pd.DataFrame(),
                do_eval=True,
                val_answer=row.prior_answer if row.answer == row.ctx_answer else row.ctx_answer
            ), axis=1)
    val_data["text"] = val_data.apply(lambda row: construct_query_with_demonstrations(
                    prompt_template_dict=prompt_template_dict,
                    val_context=row.context,
                    eos_token=tokenizer.eos_token,
                    val_query=row.query,
                    context_weight=row.weight_context,
                    context_weight_format=args.context_weight_format,
                    demonstrations_df=pd.DataFrame(),
                    do_eval=True,  
                    val_answer=row.answer
                ), axis=1)
    val_data["text_inverted"] = val_data.apply(lambda row: construct_query_with_demonstrations(
                    prompt_template_dict=prompt_template_dict,
                    val_context=row.context,
                    eos_token=tokenizer.eos_token,
                    val_query=row.query,
                    context_weight=abs(row.weight_context - 1),
                    context_weight_format=args.context_weight_format,
                    demonstrations_df=pd.DataFrame(),
                    do_eval=True,
                    val_answer=row.prior_answer if row.answer == row.ctx_answer else row.ctx_answer
                ), axis=1)
    return train_data, val_data


def remove_instruction(text, name_of_instruction="Instruction", replace_with="Instruction: Answer with one word only.\n"):
    pattern = f'{name_of_instruction}:.*?(?=Query:)'
    return re.sub(pattern, replace_with, text, flags=re.DOTALL)


def create_dataset(source_tokens, target_tokens, source_label_index, target_label_index, source_attn_mask, target_attn_mask, *args):
    # Create a dataset with the same structure as the original data
    dataset = Dataset.from_dict({
        'input_ids': target_tokens.tolist(),
        'attention_mask': target_attn_mask.tolist(),
        'source_input_ids': source_tokens.tolist(),
        'source_attention_mask': source_attn_mask.tolist(),
        'labels': torch.stack([source_label_index, target_label_index], dim=1).tolist(),
        'sources->base': [[target_tokens.shape[1] - 1]] * len(target_tokens),
        # 'source_0->base.0.pos': [[target_tokens.shape[1] - 1]] * len(target_tokens),
        # 'source_0->base.1.pos': [[target_tokens.shape[1] - 1]] * len(target_tokens),
        'subspaces': [0] * len(target_tokens),
    })
    return dataset

def prepare_train_data(args, PATHS, tokenizer, device, same_query=False, remove_weight=False, name_of_instruction="Instruction", seed=42):
    train_data, val_data = load_map_training_data(tokenizer, args, PATHS)
    random.seed(seed)
    if same_query:
        target_to_source_index =  [(i + 1) if (i % 2) == 0 else i-1 for i in range(len(train_data))]
    else:
        target_to_source_index =  []
        for i in trange(len(train_data)):
            j_start = random.randint(0, len(train_data) - 1)
            j = j_start + 1
            while j != j_start:
                j = (j + 1) % len(val_data)
                if i == j:
                    continue
                
                if train_data.iloc[i].query != train_data.iloc[j].query and train_data.iloc[i].ctx_answer != train_data.iloc[j].ctx_answer \
                    and train_data.iloc[i].weight_context != train_data.iloc[j].weight_context \
                    and train_data.iloc[i].prior_answer != train_data.iloc[j].prior_answer:
                    target_to_source_index.append(j)
                    break
            if len(target_to_source_index) != i + 1:
                raise ValueError(f"No source found for {i}")

    target_to_source_index = torch.tensor(target_to_source_index)
    source_data = train_data.iloc[target_to_source_index]
    
    target_texts = train_data["text"].tolist()
    if remove_weight:
        target_texts = [remove_instruction(text, name_of_instruction=name_of_instruction) for text in target_texts]
    source_texts = source_data["text"].tolist()
    
    source_answer = [train_data.iloc[i]["prior_answer"] if train_data.iloc[i]["answer"] == train_data.iloc[i]["ctx_answer"] else train_data.iloc[i]["ctx_answer"] for i in range(len(train_data))]
    
    source_index, target_index  = encode_answer(source_answer, train_data["answer"].tolist(), tokenizer, device, args)
    same_answer_indices = (source_index == target_index).detach().cpu()
    target_texts = [target_texts[i] for i in range(len(target_texts)) if not same_answer_indices[i]]
    source_texts = [source_texts[i] for i in range(len(source_texts)) if not same_answer_indices[i]]
    source_index = source_index[~same_answer_indices]
    target_index = target_index[~same_answer_indices]
    
    source_tokens = tokenizer(source_texts, return_tensors="pt", padding=True)
    attention_mask_source = source_tokens["attention_mask"].to(device)
    target_tokens = tokenizer(target_texts, return_tensors="pt", padding=True)
    attention_mask_target = target_tokens["attention_mask"].to(device)
    source_tokens = source_tokens["input_ids"].to(device)
    target_tokens = target_tokens["input_ids"].to(device)
    
    max_len = max(source_tokens.shape[1], target_tokens.shape[1])
    source_tokens = torch.nn.functional.pad(source_tokens, (max_len - source_tokens.shape[1], 0), value=tokenizer.pad_token_id)
    target_tokens = torch.nn.functional.pad(target_tokens, (max_len - target_tokens.shape[1], 0), value=tokenizer.pad_token_id)
    attention_mask_source = torch.nn.functional.pad(attention_mask_source, (max_len - attention_mask_source.shape[1], 0), value=0)
    attention_mask_target = torch.nn.functional.pad(attention_mask_target, (max_len - attention_mask_target.shape[1], 0), value=0)
    
    texts_inverted = train_data["text_inverted"].tolist()
    texts_inverted = [texts_inverted[i] for i in range(len(texts_inverted)) if not same_answer_indices[i]]
    target_inverted_tokens = tokenizer(texts_inverted, return_tensors="pt", padding=True)
    attention_mask_target_inverted = target_inverted_tokens["attention_mask"].to(device)
    target_inverted_tokens = target_inverted_tokens["input_ids"].to(device)
    return source_tokens, target_tokens, source_index, target_index, attention_mask_source, attention_mask_target, target_inverted_tokens, attention_mask_target_inverted

def filter_confident_samples(args, model, tt, tit, ti, si, amt, amti, batch_size=48):
    os.makedirs("cache", exist_ok=True)
    print(f"Checking for cache/confident_indices_{args.model_id}.pt")
    if os.path.exists(f"cache/confident_indices_{args.model_id}.pt"):
        return torch.load(f"cache/confident_indices_{args.model_id}.pt")

    print(f"Checking for analysis/cache/confident_indices_{args.model_id}.pt")   
    if os.path.exists(f"analysis/cache/confident_indices_{args.model_id}.pt"):
        return torch.load(f"analysis/cache/confident_indices_{args.model_id}.pt")
        
    confident_indices = []
    device = model.device
    for i in trange(0, tt.shape[0], batch_size):
        batch_end = min(i + batch_size, tt.shape[0])
        
        # Get batch data
        batch_tt = tt[i:batch_end].to(device)
        batch_tit = tit[i:batch_end].to(device)
        batch_ti = ti[i:batch_end].to(device)
        batch_si = si[i:batch_end].to(device)
        batch_amt = amt[i:batch_end].to(device)
        batch_amti = amti[i:batch_end].to(device)
        
        # Get model outputs
        with torch.no_grad():
            outputs_tt = model(batch_tt, attention_mask=batch_amt).logits
            outputs_tit = model(batch_tit, attention_mask=batch_amti).logits
        
        # Get the indices of the highest logits
        max_logit_indices_tt = outputs_tt[:, -1, :].argmax(dim=-1)
        max_logit_indices_tit = outputs_tit[:, -1, :].argmax(dim=-1)
        # Check if the highest logit indices match the target indices
        match_tt = (max_logit_indices_tt == batch_ti)
        match_tit = (max_logit_indices_tit == batch_si)
        
        not_same_answer_in_gt = (batch_ti != batch_si)
        # Combine conditions and add to confident_indices
        confident_batch = torch.where(match_tt & match_tit & not_same_answer_in_gt)[0] + i
        confident_indices.extend(confident_batch.tolist())

    confident_indices = torch.tensor(confident_indices)
    print(f"Number of confident samples: {len(confident_indices)}/{tt.shape[0]}")
    print(f"Saving to cache/confident_indices_{args.model_id}.pt")
    torch.save(confident_indices, f"cache/confident_indices_{args.model_id}.pt")
    return confident_indices


def inputs_collator(inputs, device):
    for k, v in inputs.items():
        if "->" in k:
            inputs[k] = torch.tensor(v)
        elif "subspace" in k:
            inputs[k] = v
        elif v is not None:
            inputs[k] = torch.tensor(v).to(device)
    return inputs

def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = [b[k] for b in batch]
    return out

def compute_loss(logits, labels):
   
    last_token_logits = logits[:, -1, :]
    loss_fct = CrossEntropyLoss()

    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = loss_fct(last_token_logits, labels[:, 0])
    
    # print("Batch Peek:", tokenizer.decode(last_token_logits.argmax(dim=-1)[0]))
    # print("Batch Label:", tokenizer.decode(labels[0]))
    return loss


def compute_metrics(eval_preds, eval_labels, tokenizer=None, return_target_accuracy=True, verbose=False,):
    total_count = 0
    correct_count = 0
    alter_correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        len_before = len(eval_label)
        eval_pred = eval_pred[eval_label[:,0] != eval_label[:,1]]
        eval_label = eval_label[eval_label[:,0] != eval_label[:,1]]
        len_after = len(eval_label)
        if len_before != len_after:
            print(f"Filtered {len_before - len_after} samples due to equal first token")
        pred_test_labels = torch.argmax(eval_pred[:, -1, :], dim=-1)
        correct_labels = eval_label[:, 0] == pred_test_labels
        alter_correct_count += (eval_label[:, 1] == pred_test_labels).sum().tolist()
        if verbose:
            for b_idx in range(len(eval_pred)):
                print("Pred:", tokenizer.decode(pred_test_labels[b_idx].item()), " Labels (source):", tokenizer.decode(eval_label[b_idx][0].item()), " Labels (target):", tokenizer.decode(eval_label[b_idx][1].item()))
                print("Correct:", correct_labels[b_idx].item())
                print("Alter Correct:", (eval_label[b_idx][1] == pred_test_labels[b_idx]).item())
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
    accuracy = round(correct_count / total_count, 2)
    alter_accuracy = round(alter_correct_count / total_count, 2)
    if return_target_accuracy:
        if alter_accuracy + accuracy > 1:
            print(alter_correct_count, correct_count, total_count, accuracy, alter_accuracy)
            raise ValueError("Accuracy is greater than 1")
        return {"accuracy": accuracy, "target_accuracy": alter_accuracy}
    return {"accuracy": accuracy}


def print_random_sample(dataset, tokenizer, prefix=""):
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]
    
    source_tokens = sample['source_input_ids']
    target_tokens = sample['input_ids']
    
    print(f"{prefix} Sample {idx}:")
    print("Source tokens:", tokenizer.decode(source_tokens))
    print("Target tokens:", tokenizer.decode(target_tokens))
    print("Target answer:", tokenizer.decode([sample['labels'][1]]))
    print("Source answer:", tokenizer.decode([sample['labels'][0]]))
    print()