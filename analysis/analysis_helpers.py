from ast import literal_eval
from typing import List, Dict, Tuple
import os
import hashlib
import json
import re

from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from scipy.stats import ttest_ind
import statsmodels.api as sm
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

# import torch
import wandb

from utils import load_artifact_from_wandb, construct_paths_and_dataset_kwargs, construct_artifact_name


def add_val_df_to_wandb(
    yago_qec: dict,
    DATASET_NAME: str,
    RAW_DATA_PATH: str,
    SEED: int,
    MODEL_ID: str,
    LOAD_IN_8BIT: bool,
    QUERY_ID: str,
    MAX_CONTEXTS: int,
    MAX_ENTITIES: int,
    CAP_PER_TYPE: bool,
    ABLATE_OUT_RELEVANT_CONTEXTS: bool,
    DEDUPLICATE_ENTITIES: bool,
    UNIFORM_CONTEXTS: bool,
    ENTITY_SELECTION_FUNC_NAME: str,
    ENTITY_TYPES: List[str],
    QUERY_TYPES: List[str],
    CONTEXT_TYPES: List[str],
    ANSWER_MAP: Dict[int, List[str]],
) -> str:
    # Construct paths from run parameters and construct DATASET_KWARGS_IDENTIFIABLE
    (
        data_dir,
        input_dir,
        entities_path,
        contexts_path,
        queries_path,
        answers_path,
        val_data_path,
        model_dir,
        results_dir,
        val_results_path,
        mr_results_path,
        data_id,
        model_id,
        DATASET_KWARGS_IDENTIFIABLE,
    ) = construct_paths_and_dataset_kwargs(
        DATASET_NAME=DATASET_NAME,
        RAW_DATA_PATH=RAW_DATA_PATH,
        SEED=SEED,
        MODEL_ID=MODEL_ID,
        LOAD_IN_8BIT=LOAD_IN_8BIT,
        QUERY_ID=QUERY_ID,
        MAX_CONTEXTS=MAX_CONTEXTS,
        MAX_ENTITIES=MAX_ENTITIES,
        CAP_PER_TYPE=CAP_PER_TYPE,
        UNIFORM_CONTEXTS=UNIFORM_CONTEXTS,
        DEDUPLICATE_ENTITIES=DEDUPLICATE_ENTITIES,
        ENTITY_SELECTION_FUNC_NAME=ENTITY_SELECTION_FUNC_NAME,
        ABLATE_OUT_RELEVANT_CONTEXTS=ABLATE_OUT_RELEVANT_CONTEXTS,
        OVERWRITE=False,
        ENTITY_TYPES=ENTITY_TYPES,
        QUERY_TYPES=QUERY_TYPES,
        CONTEXT_TYPES=CONTEXT_TYPES,
        ANSWER_MAP=ANSWER_MAP,
    )
    os.makedirs(results_dir, exist_ok=True)
    print(f"Logging results to w&b run {wandb.run}.")
    artifact_name = construct_artifact_name(data_id, SEED, model_id)
    artifact = wandb.Artifact(name=artifact_name, type="val_df_contexts_per_qe")
    artifact.add_dir(local_path=results_dir)
    wandb.run.log_artifact(artifact)
    return artifact_name


def load_val_df_from_wandb(
    yago_qec: dict,
    DATASET_NAME: str,
    RAW_DATA_PATH: str,
    SEED: int,
    MODEL_ID: str,
    LOAD_IN_8BIT: bool,
    QUERY_ID: str,
    MAX_CONTEXTS: int,
    MAX_ENTITIES: int,
    CAP_PER_TYPE: bool,
    ABLATE_OUT_RELEVANT_CONTEXTS: bool,
    DEDUPLICATE_ENTITIES: bool,
    UNIFORM_CONTEXTS: bool,
    ENTITY_SELECTION_FUNC_NAME: str,
    ENTITY_TYPES: List[str],
    QUERY_TYPES: List[str],
    CONTEXT_TYPES: List[str],
    ANSWER_MAP: Dict[int, List[str]],
    convert_cols=["contexts", "entity", "persuasion_scores", "persuasion_scores_kl", "relevant_context_inds"],
    verbose=False,
    overwrite_df=False,
) -> str:
    # Construct paths from run parameters and construct DATASET_KWARGS_IDENTIFIABLE
    (
        data_dir,
        input_dir,
        entities_path,
        contexts_path,
        queries_path,
        answers_path,
        val_data_path,
        model_dir,
        results_dir,
        val_results_path,
        mr_results_path,
        data_id,
        model_id,
        DATASET_KWARGS_IDENTIFIABLE,
    ) = construct_paths_and_dataset_kwargs(
        DATASET_NAME=DATASET_NAME,
        RAW_DATA_PATH=RAW_DATA_PATH,
        SEED=SEED,
        MODEL_ID=MODEL_ID,
        LOAD_IN_8BIT=LOAD_IN_8BIT,
        QUERY_ID=QUERY_ID,
        MAX_CONTEXTS=MAX_CONTEXTS,
        MAX_ENTITIES=MAX_ENTITIES,
        CAP_PER_TYPE=CAP_PER_TYPE,
        UNIFORM_CONTEXTS=UNIFORM_CONTEXTS,
        DEDUPLICATE_ENTITIES=DEDUPLICATE_ENTITIES,
        ENTITY_SELECTION_FUNC_NAME=ENTITY_SELECTION_FUNC_NAME,
        ABLATE_OUT_RELEVANT_CONTEXTS=ABLATE_OUT_RELEVANT_CONTEXTS,
        OVERWRITE=False,
        ENTITY_TYPES=ENTITY_TYPES,
        QUERY_TYPES=QUERY_TYPES,
        CONTEXT_TYPES=CONTEXT_TYPES,
        ANSWER_MAP=ANSWER_MAP,
    )
    os.makedirs(results_dir, exist_ok=True)
    print(f"Loading results from w&b run {wandb.run}.")

    artifact_name = construct_artifact_name(data_id, SEED, model_id)
    if wandb.run is not None:
        # TODO: this might result in bugs if we need to recompute.
        # Download val_df_per_qe from wandb (if not already cached there)
        artifact, files = load_artifact_from_wandb(artifact_name, save_dir=results_dir, verbose=verbose)

    return files


def construct_mr_df_given_query_id(
    yago_qec: dict,
    DATASET_NAME: str,
    RAW_DATA_PATH: str,
    SEED: int,
    MODEL_ID: str,
    LOAD_IN_8BIT: bool,
    QUERY_ID: str,
    MAX_CONTEXTS: int,
    MAX_ENTITIES: int,
    CAP_PER_TYPE: bool,
    ABLATE_OUT_RELEVANT_CONTEXTS: bool,
    DEDUPLICATE_ENTITIES: bool,
    UNIFORM_CONTEXTS: bool,
    ENTITY_SELECTION_FUNC_NAME: str,
    ENTITY_TYPES: List[str],
    QUERY_TYPES: List[str],
    CONTEXT_TYPES: List[str],
    ANSWER_MAP: Dict[int, List[str]],
    convert_cols=["contexts", "entity", "sampled_answergroups"],
    verbose=False,
    overwrite_df=False,
) -> pd.DataFrame:
    # Construct paths from run parameters and construct DATASET_KWARGS_IDENTIFIABLE
    (
        data_dir,
        input_dir,
        entities_path,
        contexts_path,
        queries_path,
        answers_path,
        val_data_path,
        model_dir,
        results_dir,
        val_results_path,
        mr_results_path,
        data_id,
        model_id,
        DATASET_KWARGS_IDENTIFIABLE,
    ) = construct_paths_and_dataset_kwargs(
        DATASET_NAME=DATASET_NAME,
        RAW_DATA_PATH=RAW_DATA_PATH,
        SEED=SEED,
        MODEL_ID=MODEL_ID,
        LOAD_IN_8BIT=LOAD_IN_8BIT,
        QUERY_ID=QUERY_ID,
        MAX_CONTEXTS=MAX_CONTEXTS,
        MAX_ENTITIES=MAX_ENTITIES,
        CAP_PER_TYPE=CAP_PER_TYPE,
        UNIFORM_CONTEXTS=UNIFORM_CONTEXTS,
        DEDUPLICATE_ENTITIES=DEDUPLICATE_ENTITIES,
        ENTITY_SELECTION_FUNC_NAME=ENTITY_SELECTION_FUNC_NAME,
        ABLATE_OUT_RELEVANT_CONTEXTS=ABLATE_OUT_RELEVANT_CONTEXTS,
        OVERWRITE=False,
        ENTITY_TYPES=ENTITY_TYPES,
        QUERY_TYPES=QUERY_TYPES,
        CONTEXT_TYPES=CONTEXT_TYPES,
        ANSWER_MAP=ANSWER_MAP,
    )
    try:
        return pd.read_csv(
            mr_results_path,
            index_col=0,
            converters={**{col: literal_eval for col in convert_cols}, **{"answer": str}},
        )
    except FileNotFoundError as e:
        print(f"Unable to find file at {mr_results_path}, full error: {e}")
        return None


def construct_df_given_query_id(
    yago_qec: dict,
    DATASET_NAME: str,
    RAW_DATA_PATH: str,
    SEED: int,
    MODEL_ID: str,
    LOAD_IN_8BIT: bool,
    QUERY_ID: str,
    MAX_CONTEXTS: int,
    MAX_ENTITIES: int,
    CAP_PER_TYPE: bool,
    ABLATE_OUT_RELEVANT_CONTEXTS: bool,
    DEDUPLICATE_ENTITIES: bool,
    UNIFORM_CONTEXTS: bool,
    ENTITY_SELECTION_FUNC_NAME: str,
    ENTITY_TYPES: List[str],
    QUERY_TYPES: List[str],
    CONTEXT_TYPES: List[str],
    ANSWER_MAP: Dict[int, List[str]],
    convert_cols={
        "contexts",
        "entity",
        "persuasion_scores",
        "persuasion_scores_kl",
        "relevant_context_inds",
        "sampled_answergroups",
    },
    verbose=False,
    overwrite_df=False,
) -> pd.DataFrame:
    # Construct paths from run parameters and construct DATASET_KWARGS_IDENTIFIABLE
    (
        data_dir,
        input_dir,
        entities_path,
        contexts_path,
        queries_path,
        answers_path,
        val_data_path,
        model_dir,
        results_dir,
        val_results_path,
        mr_results_path,
        data_id,
        model_id,
        DATASET_KWARGS_IDENTIFIABLE,
    ) = construct_paths_and_dataset_kwargs(
        DATASET_NAME=DATASET_NAME,
        RAW_DATA_PATH=RAW_DATA_PATH,
        SEED=SEED,
        MODEL_ID=MODEL_ID,
        LOAD_IN_8BIT=LOAD_IN_8BIT,
        QUERY_ID=QUERY_ID,
        MAX_CONTEXTS=MAX_CONTEXTS,
        MAX_ENTITIES=MAX_ENTITIES,
        CAP_PER_TYPE=CAP_PER_TYPE,
        UNIFORM_CONTEXTS=UNIFORM_CONTEXTS,
        DEDUPLICATE_ENTITIES=DEDUPLICATE_ENTITIES,
        ENTITY_SELECTION_FUNC_NAME=ENTITY_SELECTION_FUNC_NAME,
        ABLATE_OUT_RELEVANT_CONTEXTS=ABLATE_OUT_RELEVANT_CONTEXTS,
        OVERWRITE=False,
        ENTITY_TYPES=ENTITY_TYPES,
        QUERY_TYPES=QUERY_TYPES,
        CONTEXT_TYPES=CONTEXT_TYPES,
        ANSWER_MAP=ANSWER_MAP,
    )
    # Analysis dir
    analysis_dir = os.path.join(model_dir, "analysis")
    save_path = os.path.join(analysis_dir, "val_df_per_qe.csv")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    if verbose:
        print(f"Analysis dir: {analysis_dir}")
        print(f"Results dir: {results_dir}")
        print(f"Model dir: {model_dir}")
        print(f"Data id: {data_id}")

    # artifact_name = f"{data_id}-{SEED}-{model_id}-val_df_per_qe".replace("/", ".")
    # artifact_name = f"val_df_per_qe-{hashlib.sha256(artifact_name.encode()).hexdigest()[:8]}"
    # if wandb.run is not None and not os.path.exists(save_path):
    #     # TODO: this might result in bugs if we need to recompute.
    #     # Download val_df_per_qe from wandb (if not already cached there)
    #     artifact, files = load_artifact_from_wandb(artifact_name, save_dir=analysis_dir, verbose=verbose)

    if not overwrite_df and os.path.exists(save_path):
        if verbose:
            print(f"Loading val_df_per_qe for {QUERY_ID} from cache.")
        # print(f"Loading val_df_per_qe for {QUERY_ID} from cache at {save_path}.")
        val_df_per_qe = pd.read_csv(save_path, converters={"answer": str})
        for col in convert_cols:
            if col in val_df_per_qe:
                val_df_per_qe[col] = val_df_per_qe[col].apply(literal_eval)
    else:
        print(f"Computing val_df_per_qe for {QUERY_ID}.")

        # Set random seeds
        # torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        try:
            val_df_contexts_per_qe = pd.read_csv(
                val_results_path,
                index_col=0,
                converters={"answer": str},  # Must convert answer first for number cases
                # converters={**{col: literal_eval for col in convert_cols}, **{"answer": str}},
            )
        except FileNotFoundError as e:
            print(f"Unable to find file at {val_results_path}, full error: {e}")
            return None

        try:
            mr_df = pd.read_csv(
                mr_results_path,
                index_col=0,
                converters={"answer": str},  # Must convert answer first for number cases
                # converters={**{col: literal_eval for col in ["contexts", "entity", "sampled_answergroups"]}, **{"answer": str}},
            )
            val_df_contexts_per_qe = val_df_contexts_per_qe.merge(
                mr_df,
                on=["q_id", "query_form", "entity", "answer", "contexts"],
            )
            if len(mr_df) != len(val_df_contexts_per_qe):
                print(
                    "Warning: val_df_contexts_per_qe and mr_df have different lengths, meaning they did not fully match in all rows on the merge."
                )
        except FileNotFoundError as e:
            print(f"Unable to find mr_results_path at {mr_results_path}, full error: {e}")
            mr_df = None

        for col in convert_cols:
            if col in val_df_contexts_per_qe:
                val_df_contexts_per_qe[col] = val_df_contexts_per_qe[col].apply(literal_eval)

        if verbose:
            print("val_df_contexts_per_qe info:")
            print(val_df_contexts_per_qe.info())
            print(val_df_contexts_per_qe["entity"].value_counts())
            print(val_df_contexts_per_qe.iloc[0]["contexts"][:10])

        entities_df_tidy = pd.DataFrame(
            [
                (k, ent)
                for k, ents in yago_qec[QUERY_ID].items()
                for ent in ents
                if k in {"entities", "fake_entities", "gpt_fake_entities"}
            ],
            columns=["type", "entity"],
        )
        real_ents = set(entities_df_tidy[entities_df_tidy["type"] == "entities"]["entity"])
        fake_ents = set(entities_df_tidy[entities_df_tidy["type"] == "gpt_fake_entities"]["entity"])
        entities_df_tidy["entity"] = entities_df_tidy["entity"].apply(lambda x: (x,))
        entities_df_tidy = entities_df_tidy.drop_duplicates(
            ["entity"], keep="first"
        )  # hack: "entities" alphabetically precedes "gpt_fake_entities", so this keeps the duplicate entity as with the "real" category

        if verbose:
            print("Entities info:")
            print("# unique real ents:", len(real_ents))
            print("# unique fake ents:", len(fake_ents))
            print("# overlapping ents:", len(real_ents.intersection(fake_ents)))
            print("Overlapping ents:", real_ents.intersection(fake_ents))
            # print(
            #     entities_df["entities"].value_counts(),
            #     entities_df["fake_entities"].value_counts(),
            # )
            print(entities_df_tidy.head())
            print(entities_df_tidy.info())

        qids_to_ec = pd.DataFrame(
            [(k, v["entity_types"]) for k, v in yago_qec.items()],
            columns=["q_id", "entity_classes"],
        )
        cols = [
            "q_id",
            "query_form",
            "entity",
            "answer",
            "contexts",
            "persuasion_scores",
            "persuasion_scores_kl",
            "type",
            "susceptibility_score",
            "entity_score",
        ]
        if "sampled_mr" in val_df_contexts_per_qe:
            cols += ["sampled_mr", "sampled_answergroups", "sampled_outputs"]

        val_df_per_qe = val_df_contexts_per_qe.merge(
            entities_df_tidy,
            left_on="entity",
            right_on="entity",
            how="left",
        )[cols]
        val_df_per_qe = val_df_per_qe.merge(qids_to_ec)  # add entity types

        query_forms = val_df_per_qe["query_form"].unique()
        closed_qfs = yago_qec[QUERY_ID]["query_forms"]["closed"]
        open_qfs = yago_qec[QUERY_ID]["query_forms"]["open"]

        val_df_per_qe.loc[val_df_per_qe["query_form"].isin(closed_qfs), "query_type"] = "closed"
        val_df_per_qe.loc[val_df_per_qe["query_form"].isin(open_qfs), "query_type"] = "open"

        val_df_per_qe["relevant_context_inds"] = val_df_per_qe.apply(
            lambda row: [i for i, context in enumerate(row["contexts"]) if row["entity"][0] in context],
            axis=1,
        )
        if verbose:
            print("val_df_per_qe info:")
            print(val_df_per_qe)

            print(f"query forms: {query_forms}")

        val_df_per_qe["model_id"] = model_id
        val_df_per_qe.to_csv(save_path, index=False)

        # if wandb.run is not None:
        #     artifact = wandb.Artifact(name=artifact_name, type="resultsdf")
        #     artifact.add_file(local_path=save_path)
        #     if verbose:
        #         print(f"Logging artifact: {artifact.name}.")
        #         # print(f"Logging artifact: {artifact.name} containing {save_path}")
        #     wandb.run.log_artifact(artifact)

    return val_df_per_qe


def permutation_test(persuasion_scores: List[float], relevant_context_inds: List[int], k=100, alpha=0.95):
    persuasion_scores = np.array(persuasion_scores)

    relevant_p_scores = persuasion_scores[relevant_context_inds]
    irrelevant_p_scores = np.delete(persuasion_scores, relevant_context_inds)

    irrelevant_samples = np.random.choice(irrelevant_p_scores, size=(k, len(relevant_context_inds)), replace=True)
    irrelevant_sample_means = np.mean(irrelevant_samples, axis=1)
    relevant_p_scores_mean = np.mean(relevant_p_scores)
    relevant_p_score_percentile = (relevant_p_scores_mean > irrelevant_sample_means).mean()

    return relevant_p_score_percentile, relevant_p_score_percentile > alpha


def percent_ents_passing_pscore_permutation_test(val_df_per_qe, query_type="closed"):
    if query_type is not None:
        val_df_per_qe = val_df_per_qe[val_df_per_qe["query_type"] == query_type]

    return (
        np.array(
            [
                permutation_test(
                    val_df_per_qe.iloc[i]["persuasion_scores"],
                    val_df_per_qe.iloc[i]["relevant_context_inds"],
                )[0]
                for i in range(len(val_df_per_qe))
            ]
        ),
        np.array(
            [
                permutation_test(
                    val_df_per_qe.iloc[i]["persuasion_scores"],
                    val_df_per_qe.iloc[i]["relevant_context_inds"],
                )[1]
                for i in range(len(val_df_per_qe))
            ]
        ).mean(),
    )


def ttest(
    df,
    group1="entities",
    group2="fake_entities",
    score_col="susceptibility_score",
    type_col="type",
    permutations=None,
    alternative="less",
):
    try:
        sus_scores_real = df[df[type_col] == group1][score_col]
        sus_scores_fake = df[df[type_col] == group2][score_col]
        ttest_res = ttest_ind(sus_scores_real, sus_scores_fake, alternative=alternative, permutations=permutations)
        t_stat, p_value = ttest_res.statistic, ttest_res.pvalue
        # print(t_stat, p_value)
        cohen_d = t_stat * np.sqrt(
            (len(sus_scores_real) + len(sus_scores_fake)) / (len(sus_scores_real) * len(sus_scores_fake))
        )
        cohen_d2 = (np.mean(sus_scores_real) - np.mean(sus_scores_fake)) / np.sqrt(
            (
                np.var(sus_scores_real, ddof=1) * (len(sus_scores_real) - 1)
                + np.var(sus_scores_fake, ddof=1) * (len(sus_scores_fake) - 1)
            )
            / (len(sus_scores_real) + len(sus_scores_fake) - 2)
        )

        if not np.isclose(cohen_d, cohen_d2):
            print(f"For query_id {df['q_id'].unique()[0]}, cohen_d={cohen_d} and cohen_d2={cohen_d2} don't match.")

        # effect_size,
        return {
            "effect_size": cohen_d,
            "p_value": p_value,
            "n": len(sus_scores_fake) + len(sus_scores_real),
        }
    except ZeroDivisionError:
        return {
            "effect_size": None,
            "p_value": None,
            "n": len(sus_scores_fake) + len(sus_scores_real),
        }


def compute_ttest_scores_dfs_for_qt(
    qid_to_score_df: Dict[str, pd.DataFrame],
    query_type: str,
    group1: str,
    group2: str,
    score_col: str,
    type_col: str = "type",
    permutations=None,
    alternative="less",
):
    ttest_scores = [
        {
            "query": k,
            **ttest(
                v[v["query_type"] == query_type],
                group1=group1,
                group2=group2,
                score_col=score_col,
                type_col=type_col,
                permutations=permutations,
                alternative=alternative,
            ),
        }
        for k, v in qid_to_score_df.items()
    ]
    ttest_res_df = pd.DataFrame(ttest_scores).sort_values(by="p_value").reset_index(drop=True)
    ttest_res_df = ttest_res_df[ttest_res_df["p_value"].notna()]
    ttest_res_df["bh_adj_p_value"] = sm.stats.multipletests(pvals=ttest_res_df["p_value"], alpha=0.05, method="fdr_bh")[
        1
    ]

    return ttest_res_df


def compute_ttest_scores_dfs(
    qid_to_score_df: Dict[str, pd.DataFrame],
    group1: str,
    group2: str,
    score_col: str,
    type_col: str = "type",
    permutations=None,
    alternative="less",
):
    ttest_res_open_df = compute_ttest_scores_dfs_for_qt(
        qid_to_score_df=qid_to_score_df,
        query_type="open",
        group1=group1,
        group2=group2,
        score_col=score_col,
        type_col=type_col,
        permutations=permutations,
        alternative=alternative,
    )
    ttest_res_closed_df = compute_ttest_scores_dfs_for_qt(
        qid_to_score_df=qid_to_score_df,
        query_type="closed",
        group1=group1,
        group2=group2,
        score_col=score_col,
        type_col=type_col,
        permutations=permutations,
        alternative=alternative,
    )

    return ttest_res_open_df, ttest_res_closed_df


def count_open_closed_sig_group_match(ttest_res_open_df, ttest_res_closed_df, upper_p_bound=0.95, lower_p_bound=0.05):
    # Identify how much the open and closed queries share the same significant groups (p<0.05, in between 0.05/0.95, and p>0.95)
    ttest_res_open_df = ttest_res_open_df[ttest_res_open_df["p_value"].notna()]
    ttest_res_closed_df = ttest_res_closed_df[ttest_res_closed_df["p_value"].notna()]
    open_sig_less = ttest_res_open_df[ttest_res_open_df["p_value"] < lower_p_bound]["query"]
    open_sig_not = ttest_res_open_df[
        (lower_p_bound <= ttest_res_open_df["p_value"]) & (ttest_res_open_df["p_value"] < upper_p_bound)
    ]["query"]
    open_sig_more = ttest_res_open_df[ttest_res_open_df["p_value"] >= upper_p_bound]["query"]
    open_query_to_sig_cat = {
        query: sig_cat
        for sig_cat, query in [("less", q) for q in open_sig_less]
        + [("not", q) for q in open_sig_not]
        + [("more", q) for q in open_sig_more]
    }

    closed_sig_less = ttest_res_closed_df[ttest_res_closed_df["p_value"] < lower_p_bound]["query"]
    closed_sig_not = ttest_res_closed_df[
        (lower_p_bound <= ttest_res_closed_df["p_value"]) & (ttest_res_closed_df["p_value"] < upper_p_bound)
    ]["query"]
    closed_sig_more = ttest_res_closed_df[ttest_res_closed_df["p_value"] >= upper_p_bound]["query"]
    closed_query_to_sig_cat = {
        query: sig_cat
        for sig_cat, query in [("less", q) for q in closed_sig_less]
        + [("not", q) for q in closed_sig_not]
        + [("more", q) for q in closed_sig_more]
    }

    open_vs_closed_sig_df = pd.DataFrame(
        [
            (query, open_query_to_sig_cat[query], closed_query_to_sig_cat[query])
            for query in ttest_res_open_df["query"].unique()
        ],
        columns=["query", "open", "closed"],
    )
    return open_vs_closed_sig_df[open_vs_closed_sig_df["open"] == open_vs_closed_sig_df["closed"]]


def count_num_significant_queries(
    df,
    col_name="p_value",
    alpha=0.05,
    alternative="two-sided",
):
    df = df[df[col_name].notna()]
    ranges = {
        (0, alpha): f"significant ({alternative})",
        (alpha, 1): "insignificant",
    }
    # Initialize counters for each range for both Open and Closed p-values
    count = dict()  # {range_val: 0 for range_val in ranges.keys()}
    prop = dict()  # {range_val: 0 for range_val in ranges.keys()}

    # Count rows for Open p-values
    for range_val, cat_name in ranges.items():
        count[cat_name] = df[
            (df[col_name] > range_val[0] if range_val[0] > 0 else df[col_name] >= range_val[0])
            & (df[col_name] <= range_val[1])
        ].shape[0]
        prop[cat_name] = count[cat_name] / len(df)

    return {"count": count, "proportion": prop}


def save_ttest_df_to_json(
    ttest_res_open_df,
    ttest_res_closed_df,
    qid_to_val_df_per_qe,
    analysis_dir: str,
    filename: str = "qid_to_sus_ttest_res_and_entities.json",
):
    # Generate map from qid to t-test results and entities for analyzing why some queries are less significant
    qid_to_ttest_res_and_entities = {
        qid: {
            "closed": {
                "p_value": ttest_res_closed_df[ttest_res_closed_df["query"] == qid]["p_value"].item(),
                "effect_size": ttest_res_closed_df[ttest_res_closed_df["query"] == qid]["effect_size"].item(),
            },
            "open": {
                "p_value": ttest_res_open_df[ttest_res_open_df["query"] == qid]["p_value"].item(),
                "effect_size": ttest_res_open_df[ttest_res_open_df["query"] == qid]["effect_size"].item(),
            },
            "entities": [x[0] for x in df[df["type"] == "entities"]["entity"].tolist()],
        }
        for qid, df in qid_to_val_df_per_qe.items()
    }
    path = os.path.join(analysis_dir, filename)
    print(path)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(qid_to_ttest_res_and_entities, fp, ensure_ascii=False, indent=4)


def combine_open_and_closed_dfs(ttest_res_open_df, ttest_res_closed_df):
    ttest_res_open_df["Query"] = ttest_res_open_df["query"].apply(
        lambda x: ("reverse-" if "reverse" in x else "") + x.split("/")[-1]
    )
    ttest_res_closed_df["Query"] = ttest_res_closed_df["query"].apply(
        lambda x: ("reverse-" if "reverse" in x else "") + x.split("/")[-1]
    )
    ttest_res_open_df = ttest_res_open_df.set_index("Query")
    ttest_res_closed_df = ttest_res_closed_df.set_index("Query")

    ttest_res_open_df["Cohen's $d$"] = ttest_res_open_df["effect_size"]
    ttest_res_closed_df["Cohen's $d$"] = ttest_res_closed_df["effect_size"]
    ttest_res_open_df["$p$"] = ttest_res_open_df["p_value"]
    ttest_res_closed_df["$p$"] = ttest_res_closed_df["p_value"]

    test_results_by_qid = pd.concat(
        [
            ttest_res_open_df.sort_values("Query")[["Cohen's $d$", "$p$"]],
            ttest_res_closed_df.sort_values("Query")[["Cohen's $d$", "$p$"]],
        ],
        keys=["Open", "Closed"],
        axis=1,
    )

    return test_results_by_qid


def write_to_latex_test_results_by_qid(
    test_results_by_qid: pd.DataFrame,
    analysis_dir: str,
    filename: str,
):
    with open(os.path.join(analysis_dir, filename), "w") as outfile:
        latex_table = test_results_by_qid.to_latex(
            index=True,
            longtable=False,
            # caption="For most queries, persuasion scores are significantly higher for relevant contexts than irrelevant contexts (when averaged across entities).",
            # label="tab:p_score_test_results_by_qid",
            column_format="lrrrr",
            float_format="{:0.2f}".format,
            header=True,
            # bold_rows=True,
            multicolumn=True,
            multicolumn_format="c",
        )
        print(latex_table, file=outfile)

    return latex_table


def write_to_latex_test_sus_and_per_results_by_qid(
    test_results_by_qid: pd.DataFrame,
    analysis_dir: str,
    filename: str,
):
    # note: will still need to manually add cmidrules to make it look prettier
    with open(os.path.join(analysis_dir, filename), "w") as outfile:
        latex_table = test_results_by_qid.to_latex(
            index=True,
            longtable=False,
            # caption="For most queries, persuasion scores are significantly higher for relevant contexts than irrelevant contexts (when averaged across entities).",
            # label="tab:p_score_test_results_by_qid",
            column_format="lrrrrrrrr",
            float_format="{:0.2f}".format,
            header=True,
            # bold_rows=True,
            multicolumn=True,
            multicolumn_format="c",
        )
        print(latex_table, file=outfile)

    return latex_table


def explode_val_df_per_qe(val_df_per_qe: pd.DataFrame, columns: List[str]):
    for col in columns:
        if isinstance(val_df_per_qe[col].iloc[0], str):
            val_df_per_qe[col] = val_df_per_qe[col].apply(literal_eval)

    val_df_per_qe["combined"] = val_df_per_qe.apply(
        lambda row: list(zip(*[row[col] for col in columns])),
        axis=1,
    )
    exploded_df = pd.DataFrame(val_df_per_qe.explode("combined"))
    exploded_df[columns] = pd.DataFrame(exploded_df["combined"].tolist(), index=exploded_df.index)
    exploded_df.drop(columns=["combined"], inplace=True)
    val_df_per_qe.drop(columns=["combined"], inplace=True)

    return exploded_df


def infer_context_type(context: str, context_types: Dict[str, str]):
    # Define the regex pattern to match "{entity} is the capital of {answer}."
    context_types_regexes = {
        f"^{v.replace('{entity}', '(.+)').replace('{answer}', '(.+)')}$": k for k, v in context_types.items()
    }
    # pattern = r"^(.+) is the capital of (.+)\.$"
    for regex, ct in context_types_regexes.items():
        if re.match(regex, context):
            return ct

    return None


###############################
# Model-wide analysis helpers #
###############################
def convert_test_results_dict_to_df(test_results_dict_per_model: dict):
    """
    Convert test results dict of the format
    {
        "EleutherAI/pythia-70m-deduped": {
            "open": {
                "count": {
                    "significant (less)": 0,
                    "insignificant": 123
                },
                "proportion": {
                    "significant (less)": 0.0,
                    "insignificant": 1.0
                }
                },
            "closed": {
                "count": {
                    "significant (less)": 27,
                    "insignificant": 96
                },
                "proportion": {
                    "significant (less)": 0.21951219512195122,
                    "insignificant": 0.7804878048780488
                }
            }
        },
        "EleutherAI/pythia-410m-deduped": {
            "open": {
                ...
            }
        }
    }
    to a df with columns ['model_name', 'Model size', 'Query type', 'metric_type', 'significance', 'value']
    """
    df_data = []

    for model_name, details in test_results_dict_per_model.items():
        for category, metrics in details.items():
            for metric_type, values in metrics.items():
                for significance, value in values.items():
                    row = {
                        "model_name": model_name,
                        "Model size": model_name.split("-")[1],
                        "Model size count": get_param_size(model_name),
                        "Query type": category,
                        "metric_type": metric_type,
                        "significance": significance,
                        "value": value,
                    }
                    df_data.append(row)

    # Creating DataFrame
    df = pd.DataFrame(df_data)
    df = df.sort_values(by="Model size count")

    return df


def convert_test_results_dict_to_sig_proportion_df(test_results_dict_per_model: dict):
    df = convert_test_results_dict_to_df(test_results_dict_per_model)
    df = df[(df["metric_type"] == "proportion") & (df["significance"] != "insignificant")]
    return df


def build_effect_sz_df(
    open_results_per_model: Dict[str, pd.DataFrame],
    closed_results_per_model: Dict[str, pd.DataFrame],
):
    """
    Given open and closed results dicts of the format
    {
        "model_name": pd.DataFrame(columns=[query, effect_size, p_value, n, bh_adj_p_value]),
        ...
    }

    Create a df concatenating all dfs for both open and closed query types.
    """
    open_effect_sz_df = (
        pd.concat(
            [df for df in open_results_per_model.values()],
            keys=open_results_per_model.keys(),
            axis=0,
            # ignore_index=True,
        )
        .reset_index(names=["model_name", "temp_index"])
        .drop(columns="temp_index")
    )
    open_effect_sz_df["query_type"] = "open"

    closed_effect_sz_df = (
        pd.concat(
            [df for df in closed_results_per_model.values()],
            keys=closed_results_per_model.keys(),
            axis=0,
            # ignore_index=True,
        )
        .reset_index(names=["model_name", "temp_index"])
        .drop(columns="temp_index")
    )
    closed_effect_sz_df["query_type"] = "closed"

    effect_sz_df = pd.concat([open_effect_sz_df, closed_effect_sz_df])
    effect_sz_df["Model size"] = effect_sz_df["model_name"].apply(lambda x: x.split("-")[1])
    return effect_sz_df


def build_mean_effect_sz_df(
    open_results_per_model: Dict[str, pd.DataFrame],
    closed_results_per_model: Dict[str, pd.DataFrame],
):
    """
    Given open and closed results dicts of the format
    {
        "model_name": pd.DataFrame(columns=[query, effect_size, p_value, n, bh_adj_p_value]),
        ...
    }

    Create a df with the mean effect size for each model and query types across queries.
    """
    effect_sz_df = build_effect_sz_df(open_results_per_model, closed_results_per_model)
    # effect_sz_df
    mean_effect_sz_df = (
        effect_sz_df.groupby(["model_name", "Model size", "query_type"])
        .agg(mean_effect_sz=("effect_size", "mean"))
        .reset_index()
    )
    mean_effect_sz_df["Model size count"] = mean_effect_sz_df["model_name"].apply(get_param_size)
    mean_effect_sz_df = mean_effect_sz_df.sort_values(by="Model size count")
    return mean_effect_sz_df


def get_param_size(model_name: str):
    units_to_num_map = {"m": 1e6, "b": 1e9}
    num_params_str = model_name.split("-")[1]
    num, unit = num_params_str[:-1], num_params_str[-1]
    return float(num) * units_to_num_map[unit]


def compute_and_summarize_ttest_results(
    qid_to_val_df_per_qe,
    group1: str,
    group2: str,
    score_col: str,
    type_col: str,
    permutations: int,
    alternative: str,
):
    ttest_res_open_df, ttest_res_closed_df = compute_ttest_scores_dfs(
        qid_to_val_df_per_qe,
        group1=group1,
        group2=group2,
        score_col=score_col,
        type_col=type_col,
        permutations=permutations,
        alternative=alternative,
    )
    test_results_dict = {
        "open": count_num_significant_queries(
            ttest_res_open_df,
            col_name="bh_adj_p_value",
            alpha=0.05,
            alternative=alternative,
        ),
        "closed": count_num_significant_queries(
            ttest_res_closed_df,
            col_name="bh_adj_p_value",
            alpha=0.05,
            alternative=alternative,
        ),
    }

    return ttest_res_open_df, ttest_res_closed_df, test_results_dict


def get_test_results_and_plot_per_model(
    df_all_models: List[Tuple[str, pd.DataFrame]],
    group1: str,
    group2: str,
    score_col: str,
    type_col: str,
    permutations: int,
    alternative: str,
    title: str,
    save_path: str,
    cm: matplotlib.colors.Colormap = sns.color_palette("coolwarm_r", as_cmap=True),
):
    """
    Args:
        df_all_models - list of (model-name, df)-tuples containing at least a score_col and type_col with values of group1 and group2.

    Returns:
        (1) a dict mapping from {model_name -> df of the effect sizes and p-values for each open query}
        (2) a dict mapping from {model_name -> df of the effect sizes and p-values for each closed query}
        (3) a dict mapping from {model_name -> dict containing the percent of significant results for both open and closed queries}

    Plots:
        a big page-sized thing with effect sizes and p-values for all models for both open and closed queries
    """
    open_results_per_model = dict()
    closed_results_per_model = dict()
    test_results_per_model = dict()

    n_rows_sf, n_cols_sf = int(len(df_all_models) // 2), 2
    fig = plt.figure(constrained_layout=True, figsize=(n_cols_sf * 12, n_rows_sf * 5))
    subfigs = fig.subfigures(nrows=n_rows_sf, ncols=n_cols_sf)
    for i, (model_id, df_m) in enumerate(df_all_models):
        subfig = subfigs[i // n_cols_sf, i % n_cols_sf]
        print(f"MODEL ID: {model_id}")
        qid_to_val_df_per_qe = {qid: df_m[df_m["QUERY_ID"] == qid] for qid in df_m["QUERY_ID"].unique()}
        ttest_res_open_df, ttest_res_closed_df, test_results_dict = compute_and_summarize_ttest_results(
            qid_to_val_df_per_qe,
            group1=group1,
            group2=group2,
            score_col=score_col,
            type_col=type_col,
            permutations=permutations,
            alternative=alternative,
        )

        open_results_per_model[model_id] = ttest_res_open_df
        closed_results_per_model[model_id] = ttest_res_closed_df
        test_results_per_model[model_id] = test_results_dict

        axes = subfig.subplots(nrows=1, ncols=2)
        for i, (qt, df) in enumerate(
            {
                "open": ttest_res_open_df.sort_values(by="effect_size"),
                "closed": ttest_res_closed_df.sort_values(by="effect_size"),
            }.items()
        ):
            ax = axes[i]
            sns.stripplot(
                data=df,
                x="query",
                y="effect_size",
                hue="p_value",
                ax=ax,
                size=8,
                legend=None,
                palette=cm,
                hue_norm=matplotlib.colors.LogNorm(1e-3, 1),
            )
            ax.set(xticklabels=[])
            ax.set_title(qt)
            ax.set_ylim(
                min(
                    ttest_res_open_df["effect_size"].min(),
                    ttest_res_closed_df["effect_size"].min(),
                )
                - 0.2,
                max(
                    ttest_res_open_df["effect_size"].max(),
                    ttest_res_closed_df["effect_size"].max(),
                )
                + 0.2,
            )
            ax.set_xlabel("Query")
            ax.set_ylabel("Effect size")
        sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(1e-3, 1), cmap=cm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax)
        subfig.suptitle(f"{model_id.split('/')[1]}", size="x-large")

        # Number and proportion of queries that are significant
        print(json.dumps(test_results_dict, indent=2))

    fig.suptitle(
        title,
        y=1.05,
        size="x-large",
    )

    fig.savefig(save_path, bbox_inches="tight")

    return open_results_per_model, closed_results_per_model, test_results_per_model


def plot_prop_queries_significant_per_model(
    test_results_per_model_df,
    title,
    save_path,
    x="Model size",
    y="value",
    hue="Query type",
    marker="o",
    markersize=12,
    palette=None,
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6.5))
    sns.lineplot(
        data=test_results_per_model_df,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        marker=marker,
        markersize=markersize,
        palette=palette,
    )
    ax.set_ylabel("Proportion of significant queries")
    fig.suptitle(title, y=0.9)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")


def plot_effect_sz_per_model(
    mean_effect_sz_df,
    title,
    save_path,
    x="Model size",
    y="mean_effect_sz",
    hue="query_type",
    marker="o",
    markersize=12,
    palette=None,
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6.5))
    sns.lineplot(
        data=mean_effect_sz_df,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        marker=marker,
        markersize=markersize,
        palette=palette,
    )
    ax.set_ylabel("Effect size")
    fig.suptitle(title, y=0.9)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    return ax


def convert_entity_uri_to_entity(row: pd.Series, yago_qec):
    try:
        q_id = row["q_id"]
        entity_uris = yago_qec[q_id]["entity_uris"]
        eu_index = entity_uris.index(row["entity_uri"])
        entities = yago_qec[q_id]["entities"]

        return entities[eu_index]
    except (KeyError, ValueError):
        return None
