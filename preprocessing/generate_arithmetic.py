import random
import operator
import re
import pandas as pd
import numpy as np
import os
from utils import partition_df

ROOT_DATA_DIR = "data/Arithmetic"


def generate_dataset(num_examples=1050, mod=None, max_depth=2, upper_answer_bound=99):
    operators = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        # "%": operator.mod,
        "/": operator.truediv,  # Using floor division for integer results
        "**": pow,
    }

    dataset = []
    unique_expressions = set()

    def evaluate_expression(expr, operators_dict):
        if mod is not None:
            return eval(expr, {}, operators_dict) % mod
        else:
            return eval(expr, {}, operators_dict)

    def get_subexpressions(expr):
        subexpressions = [expr]
        # Find all subexpressions within parentheses
        paren_subexprs = re.findall(r"\([^()]+\)", expr)
        subexpressions.extend(paren_subexprs)
        # Add individual numbers
        numbers = re.findall(r"\d+", expr)
        subexpressions.extend(numbers)
        return subexpressions

    def count_operators(expr):
        count = 0
        i = 0
        while i < len(expr):
            if expr[i : i + 2] == "**":
                count += 1
                i += 2
            elif expr[i] in "+-*/":
                count += 1
                i += 1
            else:
                i += 1
        return count

    while len(dataset) < num_examples:
        num_ops = random.randint(1, max_depth)
        expression = []

        for _ in range(num_ops):
            op = random.choice(list(operators.keys()))
            num1 = random.randint(0, 9)
            num2 = random.randint(0 if op != "/" else 1, 9)
            expression.append((op, num1, num2))

        random.shuffle(expression)

        problem = ""
        for i, (op, num1, num2) in enumerate(expression):
            if i == 0:
                problem += f"{num1} {op} {num2}"
            else:
                problem = f"({problem}) {op} {num2}"

        # Add "(mod 10)" to the problem
        problem_with_mod = f"({problem}) (mod {mod}) =" if mod is not None else f"{problem} ="

        # Check if this expression is unique
        if problem_with_mod in unique_expressions:
            continue

        unique_expressions.add(problem_with_mod)

        try:
            original_answer = evaluate_expression(problem, operators)
            if int(original_answer) != original_answer:
                continue
            original_answer = int(original_answer)
            if original_answer < 0 or original_answer > upper_answer_bound:
                continue

            original_op_count = count_operators(problem)

            # Get all possible subexpressions
            subexpressions = get_subexpressions(problem)

            # Choose a random subexpression
            subexpr = random.choice(subexpressions)
            subexpr_op_count = count_operators(subexpr)

            # Generate a new random value for the subexpression
            new_value = random.randint(0, 9)
            while evaluate_expression(subexpr, operators) == new_value:
                new_value = random.randint(0, 9)

            # Create the assignment string without stripping parentheses
            assignment = f"{subexpr} = {new_value}"

            # Replace the subexpression in the problem with the new value
            new_problem = problem.replace(subexpr, str(new_value))
            # new_problem_with_mod = f"({new_problem}) (mod 10)"

            # Evaluate the new problem
            new_answer = evaluate_expression(new_problem, operators)
            if int(new_answer) != new_answer:
                continue
            new_answer = int(new_answer)
            if new_answer < 0 or new_answer > upper_answer_bound or new_answer == original_answer:
                continue

            dataset.append(
                (
                    problem,
                    problem_with_mod,
                    original_answer,
                    assignment,
                    new_answer,
                    original_op_count,
                    subexpr_op_count,
                )
            )

        except Exception as e:
            print(f"Error processing: {problem}")
            print(f"Error message: {str(e)}")
            continue

    return pd.DataFrame(
        dataset, columns=["query_no_mod", "query", "prior_answer", "context", "ctx_answer", "query_depth", "ctx_depth"]
    )


def preprocess_dataset(df, split):
    def interleave_datasets(df1, df2):
        # Ensure df1 and df2 have the same number of rows
        min_rows = min(len(df1), len(df2))
        df1 = df1.iloc[:min_rows]
        df2 = df2.iloc[:min_rows]

        # Create a temporary key for sorting
        df1["temp_key"] = np.arange(len(df1)) * 2
        df2["temp_key"] = np.arange(len(df2)) * 2 + 1

        # Concatenate and sort
        result = pd.concat([df1, df2]).sort_values("temp_key").reset_index(drop=True)

        # Remove the temporary key
        return result.drop("temp_key", axis=1)

    df_prior = df.copy()
    df_prior["weight_context"] = 0
    df_prior["answer"] = df_prior["prior_answer"]

    df_ctx = df.copy()
    df_ctx["weight_context"] = 1
    df_ctx["answer"] = df_ctx["ctx_answer"]

    df_all = interleave_datasets(df_prior, df_ctx)

    df_all["answer"] = df_all["answer"].apply(str)
    df_all["prior_answer"] = df_all["prior_answer"].apply(str)
    df_all["ctx_answer"] = df_all["ctx_answer"].apply(str)

    train_df, val_df, test_df = partition_df(df_all, columns=["query"], val_frac=0.01, test_frac=0.9)

    full_dir = os.path.join(ROOT_DATA_DIR, "splits", split)
    os.makedirs(full_dir, exist_ok=True)
    train_df.to_csv(
        os.path.join(full_dir, "train.csv"),
        index=False,
    )
    val_df.to_csv(
        os.path.join(full_dir, "val.csv"),
        index=False,
    )
    test_df.to_csv(
        os.path.join(full_dir, "test.csv"),
        index=False,
    )
    print(f"Preprocessed and saved dataset to {full_dir}")


def generate_and_preprocess_dataset(split, num_examples, max_depth, upper_answer_bound, mod=None):
    # Generate the dataset
    SEED = 0
    random.seed(SEED)

    arithmetic_dataset = generate_dataset(
        num_examples=num_examples, mod=mod, max_depth=max_depth, upper_answer_bound=upper_answer_bound
    )

    if (arithmetic_dataset["prior_answer"] == arithmetic_dataset["ctx_answer"]).any():
        print(arithmetic_dataset[arithmetic_dataset["prior_answer"] == arithmetic_dataset["ctx_answer"]])
        raise ValueError("prior and context answer are the same for at least one element.")

    print(arithmetic_dataset.head())
    os.makedirs(ROOT_DATA_DIR, exist_ok=True)
    save_path = os.path.join(ROOT_DATA_DIR, "arithmetic_dataset.csv")
    arithmetic_dataset.to_csv(save_path, sep=",", index=None)
    print(f"Dataset with {len(arithmetic_dataset)} examples generated and saved to '{save_path}'.")

    preprocess_dataset(arithmetic_dataset, split=split)


generate_and_preprocess_dataset("d2ub99", num_examples=1050, max_depth=2, upper_answer_bound=99)
generate_and_preprocess_dataset("d1ub99", num_examples=250, max_depth=1, upper_answer_bound=99)
generate_and_preprocess_dataset("d2ub9", num_examples=1050, max_depth=2, upper_answer_bound=9)
generate_and_preprocess_dataset("d1ub9", num_examples=120, max_depth=1, upper_answer_bound=9)
