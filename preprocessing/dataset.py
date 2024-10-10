import numpy as np
import json
import os
import pandas as pd
import pickle
import random
import torch
from transformers.tokenization_utils_base import BatchEncoding
from typing import Iterable, List, Optional
from enum import Enum
import re
from datasets import Dataset


class ContextQueryDataset:
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        train_size: int = None,
        seed: Optional[int] = None,
    ) -> None:
        self.train_data: ContextQueryDataset = None
        self.val_data: ContextQueryDataset = None
        self.test_data: ContextQueryDataset = None
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.train_size = train_size
        self.seed = seed
        self._set_seeds()
        self.name = None
        self.answer_format = "word"

    def _set_seeds(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def get_name(self):
        return self.name

    def get_answer_format(self):
        return self.answer_format

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data

    def _set_train_data(self) -> None:
        """Set the self.train_data field to the dataset."""
        train_df = load_dataset_from_path(self.train_path)
        self.train_data = Dataset.from_pandas(train_df[: self.train_size], split="train", preserve_index=False)

    def _set_val_data(self) -> None:
        """Set the self.val_data field to the dataset."""
        val_df = load_dataset_from_path(self.val_path)
        self.val_data = Dataset.from_pandas(val_df, split="val", preserve_index=False)

    def _set_test_data(self) -> None:
        """Set the self.test_data field to the dataset."""
        test_df = load_dataset_from_path(self.test_path)
        self.test_data = Dataset.from_pandas(test_df, split="test", preserve_index=False)


class BaseFakepedia(ContextQueryDataset):
    def __init__(
        self,
        train_path: str = "data/BaseFakepedia/splits/nodup_relpid/train.csv",
        val_path: str = "data/BaseFakepedia/splits/nodup_relpid/val.csv",
        test_path: str = "data/BaseFakepedia/splits/nodup_relpid/test.csv",
        train_size: int = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            seed=seed, train_size=train_size, train_path=train_path, val_path=val_path, test_path=test_path
        )
        self.name = "BaseFakepedia"
        self._set_train_data()
        self._set_val_data()
        self._set_test_data()

    def is_response_correct(self, prediction, label):
        return prediction.startswith(label)


class MultihopFakepedia(BaseFakepedia):
    def __init__(
        self,
        train_path: str = "data/MultihopFakepedia/splits/nodup_relpid/train.csv",
        val_path: str = "data/MultihopFakepedia/splits/nodup_relpid/val.csv",
        test_path: str = "data/MultihopFakepedia/splits/nodup_relpid/test.csv",
        train_size: int = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            seed=seed, train_size=train_size, train_path=train_path, val_path=val_path, test_path=test_path
        )
        self.name = "MultihopFakepedia"


class Arithmetic(ContextQueryDataset):
    def __init__(
        self,
        train_path: str = "data/Arithmetic/splits/nodup_relpid/train.csv",
        val_path: str = "data/Arithmetic/splits/nodup_relpid/val.csv",
        test_path: str = "data/Arithmetic/splits/nodup_relpid/test.csv",
        train_size: int = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            seed=seed, train_size=train_size, train_path=train_path, val_path=val_path, test_path=test_path
        )
        self.name = "Arithmetic"
        self._set_train_data()
        self._set_val_data()
        self._set_test_data()
        self.answer_format = "number"

    def _set_train_data(self) -> None:
        """Set the self.train_data field to the dataset."""
        try:
            train_df = load_dataset_from_path(self.train_path)
            train_df["answer"] = train_df["answer"].apply(str)
            train_df["prior_answer"] = train_df["prior_answer"].apply(str)
            train_df["ctx_answer"] = train_df["ctx_answer"].apply(str)
            self.train_data = Dataset.from_pandas(train_df[: self.train_size], split="train", preserve_index=False)
        except:  # noqa
            print("Couldn't load and set train data for Arithmetic.")

    def _set_val_data(self) -> None:
        """Set the self.val_data field to the dataset."""
        try:
            val_df = load_dataset_from_path(self.val_path)
            val_df["answer"] = val_df["answer"].apply(str)
            val_df["prior_answer"] = val_df["prior_answer"].apply(str)
            val_df["ctx_answer"] = val_df["ctx_answer"].apply(str)
            self.val_data = Dataset.from_pandas(val_df, split="val", preserve_index=False)
        except:  # noqa
            print("Couldn't load and set val data for Arithmetic.")

    def _set_test_data(self) -> None:
        """Set the self.test_data field to the dataset."""
        try:
            test_df = load_dataset_from_path(self.test_path)
            test_df["answer"] = test_df["answer"].apply(str)
            test_df["prior_answer"] = test_df["prior_answer"].apply(str)
            test_df["ctx_answer"] = test_df["ctx_answer"].apply(str)
            self.test_data = Dataset.from_pandas(test_df, split="test", preserve_index=False)
        except:  # noqa
            print("Couldn't load and set test data for Arithmetic.")

    def is_response_correct(self, prediction, label):
        # Convert prediction and label to lowercase for case-insensitive comparison
        prediction = prediction.lower().strip()
        label = label.lower().strip()

        # Dictionary to convert English number words to digits
        number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        }

        # Convert English number words to digits in both prediction and label
        for word, digit in number_words.items():
            prediction = prediction.replace(word, digit).replace(word.capitalize(), digit)

        return (
            prediction.startswith(label)
            or prediction.endswith(label)
            or label in [x.strip().strip(",.;/") for x in re.split("=|\n", prediction)]
        )


class Yago(ContextQueryDataset):
    def __init__(
        self,
        train_path: str = "data/Yago/train.csv",
        val_path: str = "data/Yago/val.csv",
        test_path: str = "data/Yago/test.csv",
        train_size: int = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            seed=seed, train_size=train_size, train_path=train_path, val_path=val_path, test_path=test_path
        )
        self.name = "Yago"
        self._set_train_data()
        self._set_val_data()
        self._set_test_data()


class YagoLlama2(Yago):
    def __init__(
        self,
        train_path: str = "data/YagoLlama2/train.csv",
        val_path: str = "data/YagoLlama2/val.csv",
        test_path: str = "data/YagoLlama2/test.csv",
        train_size: int = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            seed=seed, train_size=train_size, train_path=train_path, val_path=val_path, test_path=test_path
        )
        self.name = "YagoLlama2"


def load_dataset_from_path(path: str, **kwargs):
    """
    Loads a dataset from the path.
    """
    supported_filetypes = {".pickle", ".pt", ".csv", ".tsv", ".json"}
    _, path_suffix = os.path.splitext(path)

    if path_suffix not in supported_filetypes:
        raise ValueError(
            f"load_dataset_from_path currently only loads files of type {supported_filetypes}. Instead received a file with path suffix {path_suffix}."
        )
    else:
        if path_suffix == ".pickle":
            try:
                return pd.read_pickle(path)
            except FileNotFoundError as e:
                print(f"WARNING: unable to read pickle with pandas, instead just loading. Full error: {e}")
                with open(path, "rb") as f:
                    return pickle.load(f)
        elif path_suffix == ".pt":
            return torch.load(path)
        elif path_suffix == ".csv":
            return pd.read_csv(path, **kwargs)
        elif path_suffix == ".tsv":
            return pd.read_csv(path, sep="\t", **kwargs)
        elif path_suffix == ".json":
            with open(path, "r") as f:
                return json.load(f)


def balance_df_by_label_column(df, label_col, random_state):
    """Resample the df so that there's an equal number of instances in each class for label_col (to the min for each class)"""
    g = df.groupby(label_col)
    return g.apply(lambda x: x.sample(g.size().min(), random_state=random_state).reset_index(drop=True))
