import torch
from tqdm import trange

from nnpatch import Site, MultiSite, batched_average_cache
from nnpatch.api.llama import Llama3


from analysis.circuit_utils.utils import get_default_parser, paths_from_args, collect_data
from analysis.circuit_utils.visualisation import create_patch_scope_lplot

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
    batch_range = torch.arange(logits.shape[1]).to(logits.device)
    return logits[:, batch_range, indices.to(logits.device)]
    
def patch_scope(nnmodel, tokenizer, residuals, verbose=False):
    id_prompt_target = "cat -> cat\n1135 -> 1135\nhello -> hello\n?"
    # id_prompt_target = "I'm thinking about the word ?"
    # id_prompt_target = "My internals represent the word ?"
    id_prompt_tokens = tokenizer(id_prompt_target, return_tensors="pt", padding=True)["input_ids"].to(nnmodel.device)
    # id_prompt_tokens = torch.tensor([[tokenizer.bos_token_id]]).to(nnmodel.device)
    all_logits = []
    lrange = trange(len(nnmodel.model.layers)) if verbose else range(len(nnmodel.model.layers))
    for i in lrange:
        with nnmodel.trace(id_prompt_tokens.repeat(residuals.shape[1], 1), validate=False, scan=False):
            nnmodel.model.layers[i].output[0][:,-1,:] = residuals[i, :, :]
            logits = nnmodel.lm_head.output[:, -1, :].save()
        all_logits.append(logits.value.detach().cpu())
        
    all_logits = torch.stack(all_logits)
    return all_logits

def get_patched_residuals(nnmodel, site, source_tokens, target_tokens, source_attention_mask, target_attention_mask, scan=False, validate=False, average_site=None):
    """
    Performs patched inference on a neural network model and returns the residuals.

    This function runs a clean inference on source tokens to cache activations,
    then runs inference on target tokens while patching with the cached activations.
    It optionally applies an average site patch as well.

    Args:
        nnmodel: The neural network model to perform inference on.
        site: The site object specifying where to cache and patch activations.
        source_tokens: Input tokens for the source (caching) run.
        target_tokens: Input tokens for the target (patching) run.
        source_attention_mask: Attention mask for the source tokens.
        target_attention_mask: Attention mask for the target tokens.
        scan (bool): Whether to use scan mode in nnsight tracing.
        validate (bool): Whether to use validate mode in nnsight tracing.
        average_site: Optional average site for additional patching.

    Returns:
        torch.Tensor: Stacked residuals from all layers of the model.
    """
    
    site.reset()
    residuals = [[] for _ in range(len(nnmodel.model.layers))]
    
    # Clean run
    with nnmodel.trace(source_tokens, attention_mask=source_attention_mask, scan=scan, validate=validate) as invoker:
        site.cache(nnmodel)
                
    with nnmodel.trace(target_tokens, attention_mask=target_attention_mask, scan=scan, validate=validate) as invoker:
        site.patch(nnmodel)
        if average_site is not None:
            average_site.patch(nnmodel)
        for i in range(len(nnmodel.model.layers)):
            residuals[i].append(nnmodel.model.layers[i].output[0][:,-1,:].save())
            
    for i in range(len(nnmodel.model.layers)):
        residuals[i][-1] = residuals[i][-1].value.detach().cpu()
            
    residuals = torch.stack([torch.cat([r.detach() for r in res]) for res in residuals])
    torch.cuda.empty_cache()
    return residuals

def get_double_patched_residuals(nnmodel, site_1, site_2, source_1_tokens, source_2_tokens, target_tokens, source_1_attention_mask, source_2_attention_mask, target_attention_mask, scan=False, validate=False, average_site=None):
    """
    Performs double patched inference on a neural network model and returns the residuals.

    This function runs two clean inferences on source_1 and source_2 tokens to cache activations,
    then runs inference on target tokens while patching with both cached activations.
    It optionally applies an average site patch as well.

    Args:
        nnmodel: The neural network model to perform inference on.
        site_1: The first site object specifying where to cache and patch activations.
        site_2: The second site object specifying where to cache and patch activations.
        source_1_tokens: Input tokens for the first source (caching) run.
        source_2_tokens: Input tokens for the second source (caching) run.
        target_tokens: Input tokens for the target (patching) run.
        source_1_attention_mask: Attention mask for the first source tokens.
        source_2_attention_mask: Attention mask for the second source tokens.
        target_attention_mask: Attention mask for the target tokens.
        scan (bool): Whether to use scan mode in nnsight tracing.
        validate (bool): Whether to use validate mode in nnsight tracing.
        average_site: Optional average site for additional patching.

    Returns:
        torch.Tensor: Stacked residuals from all layers of the model.
    """

    residuals = [[] for _ in range(len(nnmodel.model.layers))]
    
    # Clean runs
    with nnmodel.trace(source_1_tokens, attention_mask=source_1_attention_mask, scan=scan, validate=validate) as invoker:
        site_1.cache(nnmodel)
    with nnmodel.trace(source_2_tokens, attention_mask=source_2_attention_mask, scan=scan, validate=validate) as invoker:
        site_2.cache(nnmodel)
    
    with nnmodel.trace(target_tokens, attention_mask=target_attention_mask, scan=scan, validate=validate) as invoker:
        site_1.patch(nnmodel)
        site_2.patch(nnmodel)
        if average_site is not None:
            average_site.patch(nnmodel)
        for i in range(len(nnmodel.model.layers)):
            residuals[i].append(nnmodel.model.layers[i].output[0][:,-1,:].save())
            
    for i in range(len(nnmodel.model.layers)):
        residuals[i][-1] = residuals[i][-1].value.detach().cpu()
            
    residuals = torch.stack([torch.cat([r.detach() for r in res]) for res in residuals])
    torch.cuda.empty_cache()
    return residuals

def batch_patched_residuals(nnmodel, site, source_tokens, target_tokens, source_attention_mask, target_attention_mask, batch_size=32, scan=False, validate=False):
    residuals = []
    for i in range(0, source_tokens.shape[0], batch_size):
        residuals.append(get_patched_residuals(nnmodel, site, source_tokens[i:i+batch_size], target_tokens[i:i+batch_size], source_attention_mask[i:i+batch_size], target_attention_mask[i:i+batch_size], scan=scan, validate=validate))
    return torch.cat(residuals)

def get_single_residuals(nnmodel, tokens, attention_mask, layer, scan=False, validate=False):
    residuals = []
    # Clean run
    with nnmodel.trace(tokens, attention_mask=attention_mask, scan=scan, validate=validate) as invoker:
        residuals.append(nnmodel.model.layers[layer].output[0][:,-1,:].save())
        nnmodel.model.layers[layer].output[0].stop()
    residuals[-1] = residuals[-1].value.detach().cpu()
            
    residuals = torch.cat(residuals, dim=0)
    torch.cuda.empty_cache()
    return residuals

def batch_get_single_residuals(nnmodel, tokens, attention_mask, layer, batch_size=32, scan=False, validate=False):
    residuals = []
    for i in trange(0, tokens.shape[0], batch_size):
        residuals.append(get_single_residuals(nnmodel, tokens[i:i+batch_size], attention_mask[i:i+batch_size], layer, scan=scan, validate=validate))
    return torch.cat(residuals)


def get_decoding_args(cwf="instruction", model_id="Meta-Llama-3.1-8B", model_store="", finetuned=True, no_filtering=False, batch_size=32, n_samples=1000, load_in_4bit=False, shots=0, dataset="BaseFakepedia"):
    base_args = [
        "--context-weight-format", cwf, 
        "--n-samples", str(n_samples),
        "--output-dir", model_store, 
        "--source-heads", "o", "q",
        "--topk", "-1",
        "--batch-size", str(batch_size),
        "--model-id", model_id,
        "--model-store", model_store,
        "--shots", str(shots),
        "--finetune-training-args", None,
        "--dataset", dataset
    ]
    if load_in_4bit:
        base_args.append("--load-4bit")
    if finetuned:
        base_args.append("--finetuned")
    if no_filtering:
        base_args.append("--no-filtering")
    parser = get_default_parser()
    args = parser.parse_args(base_args)
    PATHS = paths_from_args(args)
    return PATHS, args

def config_to_site(config, api=Llama3, model=None):
    ms = []
    for head in config:
        if "heads" not in config[head]:
            config[head]["heads"] = list(range(api.N_QO_HEADS(model)))
        ms.extend([Site.get_site(api, head, l, head=h, seq_pos=torch.tensor([-1]), cache_name="site1") for h in config[head]["heads"] for l in config[head]["layers"]])
    return MultiSite(ms)

def prepare_sites(site_1_config, site_2_config, average_site_config=None, api=Llama3, model=None):
    site_1 = config_to_site(site_1_config, api=api, model=model)
    site_2 = config_to_site(site_2_config, api=api, model=model)
    
    if average_site_config is not None:
        average_site = config_to_site(average_site_config, api=api, model=model)
    else:
        average_site = None
    return site_1, site_2, average_site
    
def merge_results(props, ranks):
    probs = (torch.cat([el[0] for el in props], dim=1), torch.cat([el[1] for el in props], dim=1), torch.cat([el[2] for el in props], dim=1))
    ranks = (torch.cat([el[0] for el in ranks], dim=1), torch.cat([el[1] for el in ranks], dim=1), torch.cat([el[2] for el in ranks], dim=1))
    return probs, ranks


def get_data(args, PATHS, tokenizer, device="cuda"):
    target_df, source_df, target_tokens, source_tokens, target_answer_index, source_answer_index, attention_mask_target, attention_mask_source = collect_data(args, PATHS, tokenizer, device)
    context_df = target_df[target_df.weight_context == 1.0]
    prior_df = source_df.iloc[context_df.index]
    context_tokens = target_tokens[context_df.index]
    prior_tokens = source_tokens[context_df.index]
    context_answer_index = target_answer_index[context_df.index]
    prior_answer_index = source_answer_index[context_df.index]
    attention_mask_context = attention_mask_target[context_df.index]
    attention_mask_prior = attention_mask_source[context_df.index]
    context_1_tokens = context_tokens
    context_1_answer = context_answer_index

    context_df = context_df.reset_index(drop=True)
    # 
    two_index = []
    for i in range(len(context_tokens)):
        j = i+1
        while True: 
            j = j % (len(context_tokens)-1)
            if context_df.iloc[i].query != context_df.iloc[j].query:
                two_index.append(j)
                break
            j += 1
    two_index = torch.tensor(two_index)
    context_2_tokens = context_tokens[two_index]
    context_2_answer = context_answer_index[two_index]
    context_1_attention_mask = attention_mask_context
    context_2_attention_mask = attention_mask_context[two_index]

    prior_1_tokens = prior_tokens
    prior_1_answer = prior_answer_index
    prior_1_attention_mask = attention_mask_prior

    two_index = []
    for i in range(len(prior_tokens)):
        j = i+1
        while True: 
            j = j % (len(prior_tokens)-1)
            if prior_df.iloc[i].query != prior_df.iloc[j].query and prior_df.iloc[i].prior_answer != prior_df.iloc[j].prior_answer:
                two_index.append(j)
                break
            j += 1
    two_index = torch.tensor(two_index)
    prior_2_tokens = prior_tokens[two_index]
    prior_2_answer = prior_answer_index[two_index]
    prior_2_attention_mask = attention_mask_prior[two_index]
    context_3_tokens = context_tokens[two_index]
    context_3_answer = context_answer_index[two_index]
    context_3_attention_mask = attention_mask_context[two_index]

    return target_tokens, attention_mask_target, context_1_tokens, context_2_tokens, context_3_tokens, prior_1_tokens, prior_2_tokens, context_1_attention_mask, context_2_attention_mask, context_3_attention_mask, prior_1_attention_mask, prior_2_attention_mask, context_1_answer, context_2_answer, context_3_answer, prior_1_answer, prior_2_answer
# def get_plot_context_double_patch(site_1_config, site_2_config, average_site_config=None, batch_size=24, title="", show_rank=True, show_prob=True):  
#     site_1, site_2, average_site = prepare_sites(site_1_config, site_2_config, average_site_config)
    
#     average_site = batched_average_cache(nnmodel, all_tokens, all_attn_mask, average_site, batch_size=batch_size)
    
#     residuals = []
#     for i in range(0, context_1_tokens.shape[0], batch_size):
#         site_1.reset()
#         site_2.reset()
#         residuals.append(get_double_patched_residuals(nnmodel, site_1, site_2, context_1_tokens[i:i+batch_size], context_2_tokens[i:i+batch_size], prior_tokens[i:i+batch_size], context_1_attention_mask[i:i+batch_size], context_2_attention_mask[i:i+batch_size], prior_attention_mask[i:i+batch_size], scan=False, validate=False, average_site=average_site))

        
#     residuals = torch.cat(residuals, dim=1)

#     aggregation = "median"
#     logits = patch_scope(nnmodel, tokenizer, residuals)
#     a_rank = get_ranks(logits, context_2_answer)
#     b_rank = get_ranks(logits, context_1_answer)
#     c_rank = get_ranks(logits, prior_answer)
#     a_prob = get_probs(logits, context_2_answer)
#     b_prob = get_probs(logits, context_1_answer)
#     c_prob = get_probs(logits, prior_answer)

#     probs, ranks = merge_results([a_prob, b_prob, c_prob],[a_rank, b_rank, c_rank])
    
#     return create_patch_scope_lplot(probs=probs, ranks=ranks, aggregation=aggregation, b_layers={h: site_1_config[h]["layers"] for h in site_1_config}, a_layers={h: site_2_config[h]["layers"] for h in site_2_config}, avg_layers={h: average_site_config[h]["layers"] for h in average_site_config}, a_title="Alt CTX", b_title="CTX", c_title="PRIOR", title=f"Patching PatchScope: {title}", add_rank=show_rank, add_prob=show_prob)

# def get_plot_prior_double_patch(site_1_config, site_2_config, average_site_config=None, batch_size=24, title="", show_rank=True, show_prob=True):
#     site_1, site_2, average_site = prepare_sites(site_1_config, site_2_config, average_site_config)
#     average_site = batched_average_cache(nnmodel, all_tokens, all_attn_mask, average_site, batch_size=batch_size)

#     residuals = []
#     for i in range(0, context_1_tokens.shape[0], batch_size):
#         site_1.reset()
#         site_2.reset()
#         residuals.append(get_double_patched_residuals(nnmodel, site_1, site_2, prior_1_tokens[i:i+batch_size], prior_2_tokens[i:i+batch_size], context_1_tokens[i:i+batch_size], prior_1_attention_mask[i:i+batch_size], prior_2_attention_mask[i:i+batch_size], context_1_attention_mask[i:i+batch_size], scan=False, validate=False, average_site=average_site))

#     residuals = torch.cat(residuals, dim=1)


#     aggregation = "median"
#     logits = patch_scope(nnmodel, tokenizer, residuals)
#     a_rank = get_ranks(logits, prior_2_answer)
#     b_rank = get_ranks(logits, prior_1_answer)
#     c_rank = get_ranks(logits, context_1_answer)
#     a_prob = get_probs(logits, prior_2_answer)
#     b_prob = get_probs(logits, prior_1_answer)
#     c_prob = get_probs(logits, context_1_answer)

#     probs, ranks = merge_results([a_prob, b_prob, c_prob],[a_rank, b_rank, c_rank])
#     return create_patch_scope_lplot(probs=probs, ranks=ranks, aggregation=aggregation, b_layers={h: site_1_config[h]["layers"] for h in site_1_config}, a_layers={h: site_2_config[h]["layers"] for h in site_2_config}, avg_layers={h: average_site_config[h]["layers"] for h in average_site_config}, a_title="Alt PRIOR", b_title="PRIOR", c_title="CTX", title=f"Patching PatchScope: {title}", add_rank=show_rank, add_prob=show_prob)

def get_plot_prior_patch(nnmodel, tokenizer, all_tokens, all_attn_mask, prior_1_tokens, prior_2_tokens, prior_1_attention_mask, prior_2_attention_mask, prior_1_answer, prior_2_answer, site_1_config, average_site_config=None, batch_size=24, N_LAYERS=42, title="", output_dir="plots", api=Llama3, max_index=None):    
    if max_index is not None:
        prior_1_tokens = prior_1_tokens[:max_index]
        prior_2_tokens = prior_2_tokens[:max_index]
        prior_1_attention_mask = prior_1_attention_mask[:max_index]
        prior_2_attention_mask = prior_2_attention_mask[:max_index]
        prior_1_answer = prior_1_answer[:max_index]
        prior_2_answer = prior_2_answer[:max_index]
    site_1, _, average_site = prepare_sites(site_1_config, {}, average_site_config, api=api, model=nnmodel)
    
    if average_site_config is not None: 
        average_site = batched_average_cache(nnmodel, all_tokens, all_attn_mask, average_site, batch_size=batch_size)
    else:
        average_site = None

    residuals = []
    for i in range(0, prior_1_tokens.shape[0], batch_size):
        site_1.reset()
        residuals.append(get_patched_residuals(nnmodel, site_1, prior_2_tokens[i:i+batch_size], prior_1_tokens[i:i+batch_size], prior_2_attention_mask[i:i+batch_size], prior_1_attention_mask[i:i+batch_size], scan=False, validate=False, average_site=average_site))

    residuals = torch.cat(residuals, dim=1)

    aggregation = "median"
    logits = patch_scope(nnmodel, tokenizer, residuals)
    a_rank = get_ranks(logits, prior_2_answer)
    b_rank = get_ranks(logits, prior_1_answer)
    a_prob = get_probs(logits, prior_2_answer)
    b_prob = get_probs(logits, prior_1_answer)

    probs, ranks = merge_results([a_prob, b_prob],[a_rank, b_rank])
    
    figr = create_patch_scope_lplot(probs=probs, aggregation=aggregation, ranks=ranks, a_layers={h: site_1_config[h]["layers"] for h in site_1_config}, b_layers={}, avg_layers={h: average_site_config[h]["layers"] for h in average_site_config} if average_site_config is not None else {}, a_title="ALT PRIOR", b_title="", c_title="PRIOR", N_LAYERS=N_LAYERS, title=None, add_rank=True, add_prob=False)
    figr.write_image(f"{output_dir}/{title}_rank.png", scale=2)
    figp = create_patch_scope_lplot(probs=probs, aggregation=aggregation, ranks=ranks, a_layers={h: site_1_config[h]["layers"] for h in site_1_config}, b_layers={}, avg_layers={h: average_site_config[h]["layers"] for h in average_site_config} if average_site_config is not None else {}, a_title="ALT PRIOR", b_title="", c_title="PRIOR", N_LAYERS=N_LAYERS, title=None, add_rank=False, add_prob=True)
    figp.write_image(f"{output_dir}/{title}_prob.png", scale=2)
    return figr, figp

def get_plot_context_patch(nnmodel, tokenizer, all_tokens, all_attn_mask, context_1_tokens, context_2_tokens, context_1_attention_mask, context_2_attention_mask, context_1_answer, context_2_answer, site_1_config, average_site_config=None, batch_size=24, N_LAYERS=32, title="", output_dir="plots", api=Llama3):
    site_1, _, average_site = prepare_sites(site_1_config, {}, average_site_config, api=api, model=nnmodel)
    
    if average_site_config is not None: 
        average_site = batched_average_cache(nnmodel, all_tokens, all_attn_mask, average_site, batch_size=batch_size)
    else:
        average_site = None

    residuals = []
    for i in range(0, context_1_tokens.shape[0], batch_size):
        site_1.reset()
        residuals.append(get_patched_residuals(nnmodel, site_1, context_2_tokens[i:i+batch_size], context_1_tokens[i:i+batch_size], context_2_attention_mask[i:i+batch_size], context_1_attention_mask[i:i+batch_size], scan=False, validate=False, average_site=average_site))

    residuals = torch.cat(residuals, dim=1)

    aggregation = "median"
    logits = patch_scope(nnmodel, tokenizer, residuals)
    a_rank = get_ranks(logits, context_2_answer)
    b_rank = get_ranks(logits, context_1_answer)
    a_prob = get_probs(logits, context_2_answer)
    b_prob = get_probs(logits, context_1_answer)

    probs, ranks = merge_results([a_prob, b_prob],[a_rank, b_rank])
    
    figr = create_patch_scope_lplot(probs=probs, aggregation=aggregation, ranks=ranks, a_layers={h: site_1_config[h]["layers"] for h in site_1_config}, b_layers={}, avg_layers={h: average_site_config[h]["layers"] for h in average_site_config} if average_site_config is not None else {}, a_title="ALT CTX", b_title="", c_title="CTX", N_LAYERS=N_LAYERS, title=None, add_rank=True, add_prob=False)
    figr.write_image(f"{output_dir}/{title}_rank.png", scale=2)
    figp = create_patch_scope_lplot(probs=probs, aggregation=aggregation, ranks=ranks, a_layers={h: site_1_config[h]["layers"] for h in site_1_config}, b_layers={}, avg_layers={h: average_site_config[h]["layers"] for h in average_site_config} if average_site_config is not None else {}, a_title="ALT CTX", b_title="", c_title="CTX", N_LAYERS=N_LAYERS, title=None, add_rank=False, add_prob=True)
    figp.write_image(f"{output_dir}/{title}_prob.png", scale=2)
    return figr, figp

def get_plot_weightcp_patch(nnmodel, tokenizer, all_tokens, all_attn_mask, context_1_tokens, prior_1_tokens, context_1_attention_mask, prior_1_attention_mask, context_1_answer, prior_1_answer, site_1_config, average_site_config=None, N_LAYERS=32, batch_size=24, title="", output_dir="plots", api=Llama3):    
    site_1, _, average_site = prepare_sites(site_1_config, {}, average_site_config, api=api, model=nnmodel)
    if average_site_config is not None: 
        average_site = batched_average_cache(nnmodel, all_tokens, all_attn_mask, average_site, batch_size=batch_size)
    else:
        average_site = None

    residuals = []
    for i in range(0, context_1_tokens.shape[0], batch_size):
        site_1.reset()
        residuals.append(get_patched_residuals(nnmodel, site_1, context_1_tokens[i:i+batch_size], prior_1_tokens[i:i+batch_size], context_1_attention_mask[i:i+batch_size], prior_1_attention_mask[i:i+batch_size], scan=False, validate=False, average_site=average_site))


    residuals = torch.cat(residuals, dim=1)

    aggregation = "median"
    logits = patch_scope(nnmodel, tokenizer, residuals)
    a_rank = get_ranks(logits, context_1_answer)
    b_rank = get_ranks(logits, prior_1_answer)
    a_prob = get_probs(logits, context_1_answer)
    b_prob = get_probs(logits, prior_1_answer)

    probs, ranks = merge_results([a_prob, b_prob],[a_rank, b_rank])
    
    figr = create_patch_scope_lplot(probs=probs, aggregation=aggregation, ranks=ranks, a_layers={h: site_1_config[h]["layers"] for h in site_1_config}, b_layers={}, avg_layers={h: average_site_config[h]["layers"] for h in average_site_config} if average_site_config is not None else {}, a_title="CTX", b_title="", c_title="PRIOR", N_LAYERS=N_LAYERS, title=None, add_rank=True, add_prob=False)
    figr.write_image(f"{output_dir}/{title}_rank.png", scale=2)
    figp = create_patch_scope_lplot(probs=probs, aggregation=aggregation, ranks=ranks, a_layers={h: site_1_config[h]["layers"] for h in site_1_config}, b_layers={}, avg_layers={h: average_site_config[h]["layers"] for h in average_site_config} if average_site_config is not None else {}, a_title="CTX", b_title="", c_title="PRIOR", N_LAYERS=N_LAYERS, title=None, add_rank=False, add_prob=True)
    figp.write_image(f"{output_dir}/{title}_prob.png", scale=2)
    return figr, figp


def get_plot_weightpc_patch(nnmodel, tokenizer, all_tokens, all_attn_mask, prior_1_tokens, context_1_tokens, prior_1_attention_mask, context_1_attention_mask,  prior_1_answer, context_1_answer, site_1_config, average_site_config=None, batch_size=24, N_LAYERS=32, title="", output_dir="plots", api=Llama3):
    site_1, _, average_site = prepare_sites(site_1_config, {}, average_site_config, api=api, model=nnmodel)
    if average_site_config is not None: 
        average_site = batched_average_cache(nnmodel, all_tokens, all_attn_mask, average_site, batch_size=batch_size)
    else:
        average_site = None

    residuals = []
    for i in range(0, context_1_tokens.shape[0], batch_size):
        site_1.reset()
        residuals.append(get_patched_residuals(nnmodel, site_1, prior_1_tokens[i:i+batch_size], context_1_tokens[i:i+batch_size], prior_1_attention_mask[i:i+batch_size], context_1_attention_mask[i:i+batch_size], scan=False, validate=False, average_site=average_site))

    residuals = torch.cat(residuals, dim=1)

    aggregation = "median"
    logits = patch_scope(nnmodel, tokenizer, residuals)
    a_rank = get_ranks(logits, prior_1_answer)
    b_rank = get_ranks(logits, context_1_answer)
    a_prob = get_probs(logits, prior_1_answer)
    b_prob = get_probs(logits, context_1_answer)

    probs, ranks = merge_results([a_prob, b_prob],[a_rank, b_rank])
    
    figr = create_patch_scope_lplot(probs=probs, aggregation=aggregation, ranks=ranks, a_layers={h: site_1_config[h]["layers"] for h in site_1_config}, b_layers={}, avg_layers={h: average_site_config[h]["layers"] for h in average_site_config} if average_site_config is not None else {}, a_title="PRIOR", b_title="", c_title="CTX", N_LAYERS=N_LAYERS, title=None, add_rank=True, add_prob=False)
    figr.write_image(f"{output_dir}/{title}_rank.png", scale=2)
    figp = create_patch_scope_lplot(probs=probs, aggregation=aggregation, ranks=ranks, a_layers={h: site_1_config[h]["layers"] for h in site_1_config}, b_layers={}, avg_layers={h: average_site_config[h]["layers"] for h in average_site_config} if average_site_config is not None else {}, a_title="PRIOR", b_title="", c_title="CTX", N_LAYERS=N_LAYERS, title=None, add_rank=False, add_prob=True)
    figp.write_image(f"{output_dir}/{title}_prob.png", scale=2)
    return figr, figp


def generate_title(site_config, prefix):
    """
    Automatically generates a title based on the site configuration.
    """
    if len(site_config) == 0:
        return prefix + "No patching"
    if "o" in site_config and "layers" in site_config["o"]:
        layers = site_config["o"]["layers"]
        continuous_ranges = []
        current_range = [layers[0]]
        
        for layer in layers[1:]:
            if layer == current_range[-1] + 1:
                current_range.append(layer)
            else:
                if len(current_range) > 1:
                    continuous_ranges.append(f"{current_range[0]}-{current_range[-1]}")
                else:
                    continuous_ranges.append(str(current_range[0]))
                current_range = [layer]
        
        if len(current_range) > 1:
            continuous_ranges.append(f"{current_range[0]}-{current_range[-1]}")
        else:
            continuous_ranges.append(str(current_range[0]))
        
        return prefix + f"O L{'+'.join(continuous_ranges)}"
    raise ValueError("Invalid site configuration")


# def get_plot_prior_self_patch(site_1_config, site_2_config, average_site_config=None, batch_size=24, title=""):    
#     site_1, site_2, average_site = prepare_sites(site_1_config, site_2_config, average_site_config)
#     average_site = batched_average_cache(nnmodel, all_tokens, all_attn_mask, average_site, batch_size=batch_size)

#     residuals = []
#     for i in range(0, context_1_tokens.shape[0], batch_size):
#         site_1.reset()
#         site_2.reset()
#         residuals.append(get_double_patched_residuals(nnmodel, site_1, site_2, prior_1_tokens[i:i+batch_size], prior_2_tokens[i:i+batch_size], prior_1_tokens[i:i+batch_size], prior_1_attention_mask[i:i+batch_size], prior_2_attention_mask[i:i+batch_size], prior_1_attention_mask[i:i+batch_size], scan=False, validate=False, average_site=average_site))

#     residuals = torch.cat(residuals, dim=1)


#     aggregation = "median"
#     logits = patch_scope(nnmodel, tokenizer, residuals)
#     a_rank = get_ranks(logits, prior_2_answer, aggregation)
#     b_rank = get_ranks(logits, prior_1_answer, aggregation)
#     c_rank = get_ranks(logits, prior_1_answer, aggregation)
#     a_prob = get_probs(logits, prior_2_answer, aggregation)
#     b_prob = get_probs(logits, prior_1_answer, aggregation)
#     c_prob = get_probs(logits, prior_1_answer, aggregation)

#     probs, ranks = merge_results([a_prob, b_prob, c_prob],[a_rank, b_rank, c_rank])
    
#     fig = create_patch_scope_plot(probs=probs, ranks=ranks, b_layers={h: site_1_config[h]["layers"] for h in site_1_config}, a_layers={h: site_2_config[h]["layers"] for h in site_2_config}, avg_layers={h: average_site_config[h]["layers"] for h in average_site_config}, a_title="Alt PRIOR", b_title="PRIOR", c_title="PRIOR", title=None)
#     fig.write_image(f"plots/{title}.png", scale=2)
#     return fig
