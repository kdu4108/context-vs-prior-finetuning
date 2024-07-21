QUERY_TEMPLATE_INT=  """Context: {}
Context Weight: {:.2f}
Query: {}"""

#
QUERY_TEMPLATE_STR= """Context: {}
Instruction: {}
Query: {}"""

TEMPLATES = {}
# LLAMA3 INSTRUCT
TEMPLATES["unsloth/llama-3-8b-Instruct-bnb-4bit"] = {
    "SYSTEM": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{}<|eot_id|>""",
    "ROUND":  """<|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>


{}""",
    "END_OF_ROUND": """<|eot_id|>"""
}

# MISTRAL INSTRUCT
TEMPLATES["unsloth/mistral-7b-instruct-v0.2-bnb-4bit"] = {
    "SYSTEM": """<s>[INST] {} \n""",
    "ROUND":  """{}[/INST]{}""",
    "END_OF_ROUND": """</s>[INST]""",
}

# LLAMA2 CHAT
TEMPLATES["unsloth/llama-2-7b-chat-bnb-4bit"] = {
    "SYSTEM": """<s>[INST] <<SYS>> {} <</SYS>> \n""",
    "ROUND":  """{}[/INST]{}""",
    "END_OF_ROUND": """[INST]""",
}

context_int_to_str = [
    "Ignore the context in answering the query.",
    "Only consider the context in answering the query.",
]

def generate_few_shot_prompts(model_name, data, val_context, val_query, context_weight=1.0, context_weight_as_int=True):
    if context_weight_as_int:
        system = TEMPLATES[model_name]["SYSTEM"].format("Answer the following query considering the provided context. ")
        rounds = []
        for i, row in data.iterrows():
            query = QUERY_TEMPLATE_INT.format(row["context"], row["weight_context"], row["query"])
            rounds.append(TEMPLATES[model_name]["ROUND"].format(query, row["answer"]) + TEMPLATES[model_name]["END_OF_ROUND"])
        query = QUERY_TEMPLATE_INT.format(val_context, context_weight, val_query)
        out = system + "".join(rounds) + TEMPLATES[model_name]["ROUND"].format(query, "")
        return out
    else:
        system = TEMPLATES[model_name]["SYSTEM"].format("Answer the following query considering the provided context. ")
        rounds = []
        for i, row in data.iterrows():
            query = QUERY_TEMPLATE_STR.format(row["context"], context_int_to_str[int(row["weight_context"])], row["query"])
            rounds.append(TEMPLATES[model_name]["ROUND"].format(query, row["answer"]) + TEMPLATES[model_name]["END_OF_ROUND"])
        query = QUERY_TEMPLATE_STR.format(val_context, context_int_to_str[int(context_weight)], val_query)
        return system + "".join(rounds) + TEMPLATES[model_name]["ROUND"].format(query, "")