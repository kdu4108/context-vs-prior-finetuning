from collections import defaultdict
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def format_query(query: str, entity: Tuple[str], context: str, prefix="", answer=None):
    """
    Number of elements in entity must match the number of format {} things in query.
    This is to handle for multiple-entity entities (e.g. friend enemy pairs)
    """
    if not isinstance(entity, tuple):
        raise ValueError("entity must be of type tuple.")
    if "{entity}" in query:
        if "{answer}" in query:
            if answer is None:
                raise ValueError("Expected answer to be provided because query contains {answer} but none was given.")
            concrete_query = query.format(entity=entity[0], answer=answer)
        else:
            concrete_query = query.format(entity=entity[0])
    else:
        if answer is not None:
            concrete_query = query.format(*entity, answer=answer)
        else:
            concrete_query = query.format(*entity)
    return prefix + context + concrete_query


def convert_fakepedia_dict_to_df(dataset: List[Dict[str, Any]]) -> pd.DataFrame:
    my_dataset = defaultdict(list)
    for d in dataset:
        # add fake
        my_dataset["context"] += [d["fact_paragraph"]]
        my_dataset["query"] += [d["query"]]
        my_dataset["weight_context"] += [1.0]
        my_dataset["answer"] += [d["object"]]

        # add real
        my_dataset["context"] += [d["fact_paragraph"]]
        my_dataset["query"] += [d["query"]]
        my_dataset["weight_context"] += [0.0]
        my_dataset["answer"] += [d["fact_parent"]["object"]]

        # Add metadata shared between both examples
        my_dataset["subject"] += [d["subject"]] * 2
        my_dataset["object"] += [d["object"]] * 2
        my_dataset["factparent_obj"] += [d["fact_parent"]["object"]] * 2
        my_dataset["ctx_answer"] += [d["object"]] * 2
        my_dataset["prior_answer"] += [d["fact_parent"]["object"]] * 2
        my_dataset["rel_p_id"] += [d["rel_p_id"]] * 2

    return pd.DataFrame.from_dict(my_dataset)


def tuple_df(df: pd.DataFrame) -> List[tuple]:
    """
    Convert df into a list of tuples.
    """
    return list(df.itertuples(index=False, name=None))


def partition_df(
    df: pd.DataFrame,
    columns: List[str],
    train_keys_df: Optional[pd.DataFrame] = None,
    val_keys_df: Optional[pd.DataFrame] = None,
    test_keys_df: Optional[pd.DataFrame] = None,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition dataset so that no two examples in train/val/test share the same value in each of `columns`.

    Args:
        df - the dataframe containing each example. Must contain `columns` columns.
        columns - the columns that cannot be shared in value across the train/val/test splits.
        train_keys_df - a df containing the unique keys according to the columns for the train split. Must be distinct from the keys in the other keys_dfs.
        val_keys_df - a df containing the unique keys according to the columns for the val split. Must be distinct from the keys in the other keys_dfs.
        test_keys_df - a df containing the unique keys according to the columns for the test split. Must be distinct from the keys in the other keys_dfs.

    Returns:
        the train_df, val_df, test_df
    """
    if train_keys_df is None or val_keys_df is None or test_keys_df is None:
        keys_df = df[columns].drop_duplicates()
        train_keys_df, test_keys_df = train_test_split(keys_df, test_size=test_frac, random_state=seed)
        train_keys_df, val_keys_df = train_test_split(train_keys_df, test_size=val_frac, random_state=seed)

    train_df = df.merge(train_keys_df, on=columns, how="inner")
    val_df = df.merge(val_keys_df, on=columns, how="inner")
    test_df = df.merge(test_keys_df, on=columns, how="inner")

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert not set(tuple_df(train_df[columns])).intersection(tuple_df(val_df[columns]))
    assert not set(tuple_df(train_df[columns])).intersection(tuple_df(test_df[columns]))

    return train_df, val_df, test_df


def split_dataset(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    columns_to_partition: List[str] = None,
    seed=0,
):
    """
    Partition df into two dfs such that the unique values of the columns in `columns_to_partition` are disjoint between the two dfs.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Get unique values for each column and create sets of values
    unique_values = {col: df[col].unique() for col in columns_to_partition}
    # Shuffle and split unique values for each column
    partitioned_values = {}
    for col, values in unique_values.items():
        np.random.shuffle(values)
        train_sz = int(len(values) * (1 - test_frac))
        # test_sz = int(len(values) * test_frac)
        partitioned_values[col] = (values[:train_sz], values[train_sz:])

    # Create masks for filtering the DataFrame
    masks = []
    for col, (part1, part2) in partitioned_values.items():
        masks.append((df[col].isin(part1), df[col].isin(part2)))

    # Combine masks to ensure no overlap
    mask1 = masks[0][0]
    mask2 = masks[0][1]
    for i in range(1, len(masks)):
        mask1 &= masks[i][0]
        mask2 &= masks[i][1]

    # Create two DataFrames based on the masks
    train_df = df[mask1]
    test_df = df[mask2]

    # Check to ensure no overlap
    overlap = train_df.merge(test_df, how="inner", on=columns_to_partition)
    print("No overlap?:", overlap.empty)  # Should be True if there is no overlap

    return train_df, test_df


def partition_df_disjoint_any_cols(
    df: pd.DataFrame, columns: List[str], val_frac=0.3, test_frac=0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition df into train/val/test dfs such that the unique values of the columns in `columns_to_partition` are disjoint between all dfs.
    """
    train_df, test_df = split_dataset(
        df,
        test_frac=test_frac,
        columns_to_partition=columns,
    )
    train_df, val_df = split_dataset(
        train_df,
        test_frac=val_frac,
        columns_to_partition=columns,
    )
    print(len(df), len(train_df), len(val_df), len(test_df))

    # Check the overlaps
    assert not set(train_df["subject"].unique()).intersection(val_df["subject"].unique())
    assert not set(train_df["subject"].unique()).intersection(test_df["subject"].unique())

    assert not set(train_df["rel_p_id"].unique()).intersection(val_df["rel_p_id"].unique())
    assert not set(train_df["rel_p_id"].unique()).intersection(test_df["rel_p_id"].unique())

    assert not set(train_df["object"].unique()).intersection(val_df["object"].unique())
    assert not set(train_df["object"].unique()).intersection(test_df["object"].unique())

    return train_df, val_df, test_df
