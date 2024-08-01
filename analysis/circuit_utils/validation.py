from transformers import pipeline
import pandas as pd
from torch import Tensor


def get_true_false_ids(df, tokenizer):
    answers = df.answer.apply(lambda x: "\n" + x).tolist()
    answers = tokenizer(answers, add_special_tokens=False)["input_ids"]
    answers = [
        el[2] for el in answers
    ]  # first two tokens are "\n", we are only interested in the first token of answer -> idx 2
    counterfactuals = df.counterfactual.apply(lambda x: "\n" + x).tolist()
    counterfactuals = tokenizer(counterfactuals, add_special_tokens=False)["input_ids"]
    counterfactuals = [
        el[2] for el in counterfactuals
    ]  # first two tokens are "\n", we are only interested in the first token of answer -> idx 2
    return answers, counterfactuals


def generate(model, tokenizer, df, batch_size=10):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=5,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=batch_size,
    )

    texts = [row.text for i, row in df.iterrows()]
    out = pipe(texts)
    
    return out


def accuracy(generations, data):
    accuracy = 0
    for i, o in enumerate(generations):
        gen = o[0]["generated_text"][len(data.iloc[i].text) :]
        accuracy += 1 if data.iloc[i].answer in gen else 0
    return float(accuracy) / len(data)


def validate(model, tokenizer, df, batch_size=10):
    generations = generate(model, tokenizer, df, batch_size)
    return accuracy(generations, df)


def iou(a, b):
    if isinstance(a, Tensor):
        a = a.tolist()
    if isinstance(b, Tensor):
        b = b.tolist()
    a = [tuple(i) for i in a]
    b = [tuple(i) for i in b]
    intersection = len(set(a).intersection(set(b)))
    union = len(set(a).union(set(b)))
    return intersection / union
