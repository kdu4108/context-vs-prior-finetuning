from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch

def load_model(path, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(path,
                                            device_map=device, torch_dtype=torch.bfloat16)
    return model


def load_peft(path, base_path="/dlabscratch1/public/llm_weights/llama2_hf/Llama-2-7b-chat-hf", device="cuda"):
    base_model = load_model(base_path, device)
    peft_model = PeftModel.from_pretrained(base_model,
            model_id = path,
            device_map = device,
            torch_dtype=torch.bfloat16
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "left"
    
    return peft_model, tokenizer

    
def merge_save_peft(peft_model, tokenizer, path):
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    tokenizer.padding_side = "left"

    return merged_model, tokenizer

def load_merged(path, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "left"
    return model, tokenizer


    