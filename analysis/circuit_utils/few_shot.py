QUERY_TEMPLATE= """Context: {}
Context Weight: {:.2f}
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

def generate_few_shot_prompts(model_name, data, val_context, val_query, context_weight=1.0):
    system = TEMPLATES[model_name]["SYSTEM"].format("Answer the following query considering the provided context. A context weight of 1.0 means the context is fully relevant to the query, while a weight of 0.0 means the context is not relevant.")
    rounds = []
    for i, row in data.iterrows():
        query = QUERY_TEMPLATE.format(row["context"], row["weight_context"], row["query"])
        rounds.append(TEMPLATES[model_name]["ROUND"].format(query, row["answer"]) + TEMPLATES[model_name]["END_OF_ROUND"])
    query = QUERY_TEMPLATE.format(val_context, context_weight, val_query)
    return system + "".join(rounds) + TEMPLATES[model_name]["ROUND"].format(query, "")