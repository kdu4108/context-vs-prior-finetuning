from typing import Callable, List, Optional, Tuple


def convert_text_to_qa_pair(
    df,
    question: str,
    answers: Tuple[str, str],
    label_to_int_mapper: Optional[Callable] = None,
    text_col: str = "hard_text_untokenized",
    label_col="g",
    make_lowercase=False,
) -> List[Tuple[str, str, int]]:
    """
    Map each row (sentence) of a dataframe to a tuple of (sentence with positive answer, sentence with negative answer, label)

    Args:
        df - a dataframe containing at least the columns `text_col` and `label_col`.
        question - the question to append to the end of each example. Ex: "Is this person male?"
        answers - the 2 candidate answers to append to the each of each question.
            For ease, default to the "positive" answer at index 0 and "negative" answer at index 1, because positivity first
            Ex: ("Yes", "No")
        text_col - the name of the column in df containing the main sentence/text.
        label_col - the name of the column in df containing the true label of the concept of interest for this example (e.g. for gender, "m" or "f")
        label_to_int_mapper - function mapping from an element in df[label_col] to 0 or 1 depending on which answer is True.

    Returns:
        Tuple of (pos_example, neg_example, label={0, 1}) where label is an int (0 or 1) as determined by the provided label_to_int_mapper function.
    """
    pos_neg_label_triple = []
    for text, label in zip(list(df[text_col]), list(df[label_col])):
        text_with_question = text + f" {question}"
        pos_example = text_with_question + f" {answers[0]}."
        neg_example = text_with_question + f" {answers[1]}."
        int_label = label_to_int_mapper(label) if label_to_int_mapper is not None else label
        pos_neg_label_triple.append(
            (
                pos_example.lower() if make_lowercase else pos_example,
                neg_example.lower() if make_lowercase else neg_example,
                int_label,
            )
        )

    return pos_neg_label_triple
