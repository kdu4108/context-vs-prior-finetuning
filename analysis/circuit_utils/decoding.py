import torch
from tqdm import trange


def get_rank(x, indices):
    vals = x[range(len(x)), indices]
    return (x > vals[:, None]).long().sum(1)

def get_prob(x, indices):
    return x.softmax(-1)[range(len(x)), indices]

def get_ranks(logits, indices):
    ranks = torch.stack([get_rank(logits[i].cpu().detach(), indices.cpu().detach()) for i in range(len(logits))]).float()
    stdv = ranks.std(-1).unsqueeze(1)
    mean = ranks.mean(-1).unsqueeze(1)
    median = ranks.median(-1).values.unsqueeze(1)
    return mean, stdv, median

def get_probs(logits, indices):
    probs = torch.stack([get_prob(logits[i].cpu().detach(), indices.cpu().detach()) for i in range(len(logits))]).float()
    stdv = probs.std(-1).unsqueeze(1)
    mean = probs.mean(-1).unsqueeze(1)
    median = probs.median(-1).values.unsqueeze(1)
    return mean, stdv, median

    
def get_logits(logits, indices):
    print(logits.shape, indices.shape)
    batch_range = torch.arange(logits.shape[1]).to(logits.device)
    return logits[:, batch_range, indices.to(logits.device)]
    
def patch_scope(nnmodel, tokenizer, residuals):
    # id_prompt_target = "cat -> cat\n1135 -> 1135\nhello -> hello\n?"
    id_prompt_target = "I'm thinking about the word ?"
    # id_prompt_target = "My internals represent the word ?"
    id_prompt_tokens = tokenizer(id_prompt_target, return_tensors="pt", padding=True)["input_ids"].to(nnmodel.device)
    all_logits = []
    for i in trange(len(nnmodel.model.layers)):
        with nnmodel.trace(id_prompt_tokens.repeat(residuals.shape[1], 1), validate=False, scan=False):
            nnmodel.model.layers[i].output[0][:,-1,:] = residuals[i, :, :]
            logits = nnmodel.lm_head.output[:, -1, :].save()
        all_logits.append(logits.value.detach().cpu())
        
    all_logits = torch.stack(all_logits)
    return all_logits