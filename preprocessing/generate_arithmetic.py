import random
import operator
import re
import pandas as pd
import os


def generate_dataset(num_examples=10000):
    operators = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        # '/': operator.floordiv,  # Using floor division for integer results
        "**": pow,
    }

    dataset = []
    unique_expressions = set()

    def evaluate_expression(expr, operators_dict):
        return eval(expr, {}, operators_dict) % 10

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
        num_ops = random.randint(1, 4)
        expression = []

        for _ in range(num_ops):
            op = random.choice(list(operators.keys()))
            num1 = random.randint(0, 9)
            num2 = random.randint(0, 9)
            expression.append((op, num1, num2))

        random.shuffle(expression)

        problem = ""
        for i, (op, num1, num2) in enumerate(expression):
            if i == 0:
                problem += f"{num1} {op} {num2}"
            else:
                problem = f"({problem}) {op} {num2}"

        # Add "(mod 10)" to the problem
        problem_with_mod = f"({problem}) (mod 10)"

        # Check if this expression is unique
        if problem_with_mod in unique_expressions:
            continue

        unique_expressions.add(problem_with_mod)

        try:
            original_answer = evaluate_expression(problem, operators)
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
            assignment = f"{subexpr}={new_value}"

            # Replace the subexpression in the problem with the new value
            new_problem = problem.replace(subexpr, str(new_value))
            # new_problem_with_mod = f"({new_problem}) (mod 10)"

            # Evaluate the new problem
            new_answer = evaluate_expression(new_problem, operators)

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


# Generate the dataset
SEED = 0
random.seed(SEED)

arithmetic_dataset = generate_dataset()

print(arithmetic_dataset.head())
DATA_DIR = "data/Arithmetic"
os.makedirs(DATA_DIR, exist_ok=True)
save_path = os.path.join(DATA_DIR, "arithmetic_dataset.csv")
arithmetic_dataset.to_csv(save_path, sep=",", index=None)

print(f"Dataset with {len(arithmetic_dataset)} examples generated and saved to '{save_path}'.")