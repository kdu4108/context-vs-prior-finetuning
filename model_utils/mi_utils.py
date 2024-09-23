import gc
from typing import Callable, List, Dict, Optional, Set, Tuple, Union
from collections import Counter
import math
import numpy as np
import re
import pandas as pd
import torch
import scipy.stats as sst
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Approximate x ∈ Σ∗ with a set of contexts from a dataset
# 2. Approximate p(y|x, q[e]) with monte carlo samples of y given x and q[e].
# 3. Approximate p(y|q[e]) with monte carlo samples of x.
# 4. Approximate p(x) with samples from a corpus (empirical distribution), but meaning/interpretation is complicated.
# Run a model on each of these sentences and get a score


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. This is modified from huggingface's
    `transformers.modeling_utils.create_position_ids_from_input_ids`.

    :param torch.Tensor x:
    :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) - 1) * mask
    return (
        incremental_indices.long()
    )  # + padding_idx (for some reason this is here in the OG code, but I can't make sense of why)


def estimate_prob_x_given_e(entity: str, contexts: Set[str], contexts_counter: Optional[Counter] = None):
    """
    Returns a (len(contexts),) nparray containing the probability of each context.

    Args:
        contexts - a set of unique contexts
        contexts_counter - a counter mapping the counts of each context in the list of contexts
    """
    if contexts_counter is not None:
        return np.array([contexts_counter[c] / contexts_counter.total() for c in contexts])

    # Otherwise, assume uniform distribution over contexts
    return np.ones(len(contexts)) / len(contexts)


def get_prob_next_word(model: AutoModelForCausalLM, tokens: Dict[str, torch.LongTensor]):
    """
    Args:
        model
        tokens - dict of
            {
                "input_ids": torch tensor of token IDs with shape (bs, context_width),
                "attention_mask: torch tensor of attention mask with shape (bs, context_width)
            }

    Returns:
        (bs, vocab_sz) tensor containing the probability distribution over the vocab of the next token for each sequence in the batch `tokens`.
    """
    try:
        position_ids = create_position_ids_from_input_ids(tokens["input_ids"], model.config.pad_token_id)
        logits = model(**tokens, position_ids=position_ids)["logits"]  # shape: (bs, mcw, vocab_sz)
    except TypeError as e:  # noqa: F841
        # print(
        #     f"Failed to make forward pass with position_ids; do you have a sufficient transformers library version? (e.g. >=4.30.0 ish?)\nFull error: {e}"
        # )
        logits = model(**tokens)["logits"]  # shape: (bs, mcw, vocab_sz)
    return logits[:, -1, :]  # shape: (bs, vocab_sz)


def check_answer_map(model, answer_map):
    special_model_tokens = set([v for (k, v) in vars(model.config).items() if k.endswith("token_id")])
    counter = Counter()
    for k, idxs in answer_map.items():
        for idx in idxs:
            idx = idx.item()
            if idx in special_model_tokens:
                counter[idx] += 1
    if counter:
        raise ValueError(
            f"WARNING: some of the tokens in your answer map correspond to special tokens of the model you may not have intended. This could occur if one of your tokens to the model is unknown and therefore was given an ID of an UNK, PAD, EOS, etc. token. Here are the counts of each special token in your answer map: {counter}."
        )
        print(
            f"WARNING: some of the tokens in your answer map correspond to special tokens of the model you may not have intended. This could occur if one of your tokens to the model is unknown and therefore was given an ID of an UNK, PAD, EOS, etc. token. Here are the counts of each special token in your answer map: {counter}."
        )


def score_model_for_next_word_prob(
    prompts: List[str],
    model,
    tokenizer,
    start: int = 0,
    end: Optional[int] = None,
    answer_map: Dict[int, List[str]] = None,
) -> torch.FloatTensor:
    """
    Args:
        prompts - list of prompts on which to score the model and get probability distribution for next word
        model
        tokenizer
        start, end - optional indices at which to slice the prompts dataset for scoring. By default, slice the whole dataset.
        answer_map - dict from the answer support (as an int) to the tokens which qualify into the respective answer.

    Returns:
        (end-start, answer_vocab_sz)-shaped torch float tensor representing the logit distribution (over the answer vocab) for the next token for all prompts in range start:end.
    """
    tokens = tokenizer(prompts[start:end], padding=True, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )  # shape: (len(contexts), max_context_width)
    last_word_logits = get_prob_next_word(model, tokens)  # shape: (len(contexts), vocab_sz)

    if answer_map is not None:
        check_answer_map(model, answer_map)
        last_word_logits_agg = torch.zeros(last_word_logits.shape[0], len(answer_map), device=model.device)
        for answer, option_ids in answer_map.items():
            logit_vals = torch.index_select(
                input=last_word_logits, dim=1, index=option_ids
            )  # shape; (bs, len(option_ids))
            last_word_logits_agg[:, answer] = torch.sum(logit_vals, dim=1)
        last_word_logits = last_word_logits_agg

    return last_word_logits


def generate(
    prompts: List[str],
    model,
    tokenizer,
    max_output_length,
    start: int = 0,
    end: Optional[int] = None,
) -> torch.FloatTensor:
    """
    Args:
        prompts - list of prompts on which to score the model and get probability distribution for next word
        model
        tokenizer
        start, end - optional indices at which to slice the prompts dataset for scoring. By default, slice the whole dataset.
        max_output_length - how many tokens to generate

    Returns:
        (end-start, max_output_length)-shaped torch long tensor representing the generated outputfor all prompts in range start:end.
    """
    tokens = tokenizer(prompts[start:end], padding=True, return_tensors="pt").to(
        model.device
    )  # shape: (len(contexts), max_context_width)
    output_tokens = model.generate(**tokens, max_length=len(tokens["input_ids"][0]) + max_output_length)[
        :, -max_output_length:
    ]

    return output_tokens


def determine_bs(f, model, tokenizer, prompts, **kwargs):
    raise NotImplementedError("TODO for when I want to optimize for efficiently maxing out GPU usage in batch scoring")


def sharded_score_model(
    f: Callable,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    bs: Optional[int] = None,
    **kwargs,
) -> torch.FloatTensor:
    if bs is None:
        bs = determine_bs(f, model, tokenizer, prompts, **kwargs)

    num_batches = math.ceil(len(prompts) / bs)
    output = []
    for b in range(num_batches):
        start, end = b * bs, min((b + 1) * bs, len(prompts))
        output.append(
            f(model=model, tokenizer=tokenizer, prompts=prompts, start=start, end=end, **kwargs).detach().cpu()
        )
        torch.cuda.empty_cache()
        gc.collect()

    return torch.cat(output, dim=0).float()


def estimate_prob_next_word_given_x_and_entity(
    query,
    entity: str,
    contexts: Union[Set[str], List[dict]],
    format_func: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    bs=32,
    answer_map=None,
    answer_entity: Optional[str] = None,
):
    """
    Args:
        entity: str - the entity of interest
        contexts: List[str] - list of contexts appended to the query regarding entity

    Returns:
      samples - a list of torch longtensors of shape (num_samples, max_length) with length len(contexts)
      possible_outputs - a dict mapping from all observed outputs to its unique index.
    """
    complete_queries = [format_func(query, entity, context) for context in contexts]
    if tokenizer.padding_side != "left":
        raise ValueError(
            f"Expected tokenizer {tokenizer} to have padding side of `left` for batch generation, instead has padding side of `{tokenizer.padding_side}`. Please make sure you initialize the tokenizer to use left padding."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id != model.config.eos_token_id:
        print("Setting model.config.pad_token_id to model.config.eos_token_id")
        model.config.pad_token_id = model.config.eos_token_id

    last_word_logits: torch.FloatTensor = sharded_score_model(
        f=score_model_for_next_word_prob,
        model=model,
        tokenizer=tokenizer,
        prompts=complete_queries,
        bs=bs,
        answer_map=answer_map,
    )  # shape: (len(contexts), vocab_sz)

    last_word_probs = torch.nn.functional.softmax(last_word_logits, dim=1)  # shape: (len(contexts, vocab_sz))

    return last_word_probs.detach().cpu().numpy()


def estimate_prob_y_given_context_and_entity(
    query: str,
    entity: str,
    contexts: Set[str],
    format_func: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_samples=None,
    max_output_length=1,
    answer_map=None,
    bs=32,
    answer_entity: Optional[str] = None,
):
    """
    Args:
        output_samples - a list of sampled outputs from the model given the context and entity.
                         Outputs need not be unique.

    Returns:
        a (len(set(output_samples)),) nparray containing the probability of each output.
    """
    if max_output_length > 1 and num_samples is None:
        raise ValueError(
            "Estimating p(y | x, q[e]) for outputs y with length >1 requires sampling. Please specify a value for num_samples."
        )

    if num_samples is not None:
        return sample_y_given_x_and_entity(
            query=query,
            entity=entity,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            num_samples=num_samples,
            max_output_length=max_output_length,
            answer_entity=answer_entity,
        )  # TODO: implement this to work for answer_entity

    return estimate_prob_next_word_given_x_and_entity(
        query=query,
        entity=entity,
        contexts=contexts,
        format_func=format_func,
        model=model,
        tokenizer=tokenizer,
        answer_map=answer_map,
        bs=bs,
        answer_entity=answer_entity,
    )


def sample_y_given_x_and_entity(
    query,
    entity: str,
    contexts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_samples=1,
    max_output_length=10,
    bs=32,
    answer_entity: Optional[str] = None,
) -> torch.LongTensor:
    """
    Args:
        entity: str - the entity of interest
        contexts: List[str] - list of contexts appended to the query regarding entity
        max_output_length: int - max number of tokens to output. Default to 10 because most entities are 1-5 words long

    Returns:
        a (len(contexts), max_output_length)-shaped tensor of the model's tokens for each complete query prefixed with a context in contexts.
    """
    complete_queries = [format_query(query, entity, context, answer=answer_entity) for context in contexts]

    if tokenizer.padding_side != "left":
        raise ValueError(
            f"Expected tokenizer {tokenizer} to have padding side of `left` for batch generation, instead has padding side of `{tokenizer.padding_side}`. Please make sure you initialize the tokenizer to use left padding."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id != model.config.eos_token_id:
        print("Setting model.config.pad_token_id to model.config.eos_token_id")
        model.config.pad_token_id = model.config.eos_token_id

    output_tokens = sharded_score_model(
        f=generate,
        model=model,
        tokenizer=tokenizer,
        prompts=complete_queries,
        bs=bs,
        max_output_length=max_output_length,
    )

    return output_tokens  # shape: (len(contexts), max_output_length)


def construct_regex_for_answer_from_context_template(template):
    """
    Given a context template like "{entity} is the highest point of", constructs a regex which can return the answer from a sentence completing that template (until the end of a period and new line).
    So, we want a regex that can return "Asia" for the sentence "Jamaica is the highest point of Asia.\n"
    """
    # Patterns for entity and answer placeholders
    entity_pattern = r"(?:.+)"  # non matching group
    answer_pattern = r"(.*?)"  # matching group

    # Escape special characters in the template, then replace placeholders
    template_escaped = re.escape(template)
    template_with_patterns = template_escaped.replace("\\{entity\\}", entity_pattern).replace(
        "\\{answer\\}", answer_pattern
    )

    # The final regex pattern captures the answer
    regex_pattern = template_with_patterns + r"(?=\.\n|$)"
    return regex_pattern


def extract_answer(context_template: str, sentence: str):
    """
    Given a context_template (e.g., "{entity} is the highest point of")
    and a sentence (e.g. "Jamaica is the highest point of Asia.\n"),
    returns the answer in the context (e.g., "Asia").
    """
    regex = construct_regex_for_answer_from_context_template(context_template)
    match = re.search(regex, sentence)
    if match:
        return match.group(1)

    print("No match found")
    return None


def p_score_ent_diff(prob_y_given_e, prob_y_given_context_and_entity):
    """
    Defining p_score as:
      H(Y|E=e) - H(Y|X=x, E=e)
    Return shape: (|X|,)
    """
    H_y_given_e: np.float16 = -np.sum(prob_y_given_e * np.nan_to_num(np.log(prob_y_given_e)))  # shape: ()
    H_y_given_x_e: np.float16 = -np.sum(
        prob_y_given_context_and_entity * np.nan_to_num(np.log(prob_y_given_context_and_entity)), axis=1
    )  # shape: (|X|,)
    persuasion_scores: np.ndarray = H_y_given_e - H_y_given_x_e  # shape: (|X|,)
    return persuasion_scores


def p_score_kl(prob_y_given_e, prob_y_given_context_and_entity):
    """
    Defining p_score as:
      \sum_{y \in Y} p(y | X=x, E=e) * log(p(y | X=x, E=e) / p(y | E=e) # noqa
    Return shape: (|X|,)
    """
    log_prob_ratio = np.nan_to_num(np.log(prob_y_given_context_and_entity / prob_y_given_e))  # shape: (|X|, |Y|)
    persuasion_scores: np.ndarray = np.sum(prob_y_given_context_and_entity * log_prob_ratio, axis=1)  # shape: (|X|,)
    return persuasion_scores


def p_scores_per_context(p_scores, contexts_set, contexts, dtype=np.float64):
    """
    Given a p-score for each context in contexts_set, returns the p-scores for each context in contexts.
    p_scores and contexts_set should be the same length.
    """
    context_to_pscore = {context: score for context, score in zip(contexts_set, p_scores.astype(dtype))}
    persuasion_scores: List[float] = [context_to_pscore[context] for context in contexts]  # shape: (len(contexts),)
    return persuasion_scores


def compute_sus_and_persuasion_scores(
    query: str,
    entity: str,
    contexts: List[Dict[str, str]],
    format_func: str,
    model,
    tokenizer,
    answer_map: Dict[int, List[str]] = None,
    bs: int = 32,
    answer_entity: Optional[str] = None,
) -> Tuple[float, float]:
    """
    (1) Computes the conditional mutual information I(X; Y | q[e]) of answer Y and context X when conditioned on query regarding entity e.

    I(X; Y | q[e]) = \sum_{x \in X} \sum_{y \in Y} (p(x, y | q[e]) * log(p(y | x, q[e]) / p(y | q[e]))) # noqa: W605

    So we need to monte carlo estimate:
        (1) p(y | x, q[e])                                             , shape: (|X|, |Y|)
        (2) p(x | q[e])                                                , shape: (|X|,)
        (3) p(x, y | q[e]) = p(y | x, q[e]) * p(x | q[e])              , shape: (|X|, |Y|)
        (4) p(y | q[e]) = \sum_{x \in X} (p(y | x, q[e]) * p(x | q[e])), shape: (|Y|,) # noqa: W605


    (2) Furthermore, computes the half pointwise conditional MI I(Y; X=x | q[e]).

    I(Y; X=x | q[e]) = H(Y | q[e]) - H(Y | X=x, q[e])
                     = - \sum_{y \in Y} (p(y | q[e]) * log(p(y | q[e]))) + \sum_{y \in Y} (p(y | x, q[e]) * log(p(y | x, q[e])))

    Args:
        entity: str - the entity of interest
        contexts: List[str] - list of contexts prepended to the query
        model - the model to use for scoring
        tokenizer - the tokenizer to use for tokenizing the contexts and query
        answer_map - dict from the answer support (as an int) to the tokens which qualify into the respective answer.
        bs - batch size to use for scoring the model
        answer_entity - the entity representing the "answer" to a question. Formatted into closed queries.

    Returns:
        sus_score - the susceptibility score for the given entity
        persuasion_scores - the persuasion scores each context for the given entity
    """
    contexts_counter = Counter(contexts)
    contexts_set = sorted(list(set(contexts)))

    prob_x_given_e = estimate_prob_x_given_e(entity, contexts_set, contexts_counter=contexts_counter)  # shape: (|X|,)
    model.eval()
    with torch.no_grad():
        prob_y_given_context_and_entity = estimate_prob_y_given_context_and_entity(
            query,
            entity,
            contexts_set,
            format_func,
            model,
            tokenizer,
            answer_map=answer_map,
            bs=bs,
            answer_entity=answer_entity,
        )  # shape: (|X|, |Y|)

    prob_x_y_given_e = np.einsum("ij, i -> ij", prob_y_given_context_and_entity, prob_x_given_e)  # shape: (|X|, |Y|)
    prob_y_given_e = np.einsum("ij, i -> j", prob_y_given_context_and_entity, prob_x_given_e)  # shape: (|Y|,)

    sus_score: np.float16 = np.sum(
        prob_x_y_given_e * np.nan_to_num(np.log(prob_y_given_context_and_entity / prob_y_given_e))
    )  # shape: ()

    persuasion_scores_kl = p_score_kl(prob_y_given_e, prob_y_given_context_and_entity)  # shape: (|X|,)
    persuasion_scores_kl: List[float] = p_scores_per_context(
        persuasion_scores_kl, contexts_set=contexts_set, contexts=contexts
    )

    return sus_score, persuasion_scores_kl
