import unittest as ut
from datasets import Dataset
from model_utils.utils import ALPACA_PROMPT, GEMMA_PROMPT, format_prompts


class TestFormatPrompt(ut.TestCase):
    def test_format_prompt_alpaca(self):
        dataset = Dataset.from_dict(
            {
                "context": ["The Beatles wrote Tiny Dancer.", "The Beatles wrote Tiny Dancer."],
                "query": ["Who wrote Tiny Dancer?", "Who wrote Tiny Dancer?"],
                "weight_context": [1.0, 0.0],
                "answer": ["The Beatles", "Elton John"],
            }
        )
        actual_train = format_prompts(dataset, "<EOS>", ALPACA_PROMPT, do_eval=False)
        expected_train = [
            "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAnswer the following query considering the provided context.\n\n### Input:\nContext: The Beatles wrote Tiny Dancer. \nContext weight: 1.00\nQuery: Who wrote Tiny Dancer?\n\n### Response:\nThe Beatles<EOS>",
            "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAnswer the following query considering the provided context.\n\n### Input:\nContext: The Beatles wrote Tiny Dancer. \nContext weight: 0.00\nQuery: Who wrote Tiny Dancer?\n\n### Response:\nElton John<EOS>",
        ]
        assert expected_train == actual_train

        actual_eval = format_prompts(dataset, "<EOS>", ALPACA_PROMPT, do_eval=True)
        expected_eval = [
            "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAnswer the following query considering the provided context.\n\n### Input:\nContext: The Beatles wrote Tiny Dancer. \nContext weight: 1.00\nQuery: Who wrote Tiny Dancer?\n\n### Response:\n",
            "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAnswer the following query considering the provided context.\n\n### Input:\nContext: The Beatles wrote Tiny Dancer. \nContext weight: 0.00\nQuery: Who wrote Tiny Dancer?\n\n### Response:\n",
        ]
        assert expected_eval == actual_eval

    def test_format_prompt_gemma(self):
        dataset = Dataset.from_dict(
            {
                "context": ["The Beatles wrote Tiny Dancer.", "The Beatles wrote Tiny Dancer."],
                "query": ["Who wrote Tiny Dancer?", "Who wrote Tiny Dancer?"],
                "weight_context": [1.0, 0.0],
                "answer": ["The Beatles", "Elton John"],
            }
        )

        actual_train = format_prompts(dataset, "<EOS>", GEMMA_PROMPT, do_eval=False)
        expected_train = [
            "<start_of_turn>user\nAnswer the following query considering the provided context.\n\nContext: The Beatles wrote Tiny Dancer. \nContext weight: 1.00\nQuery: Who wrote Tiny Dancer?<end_of_turn>\n<start_of_turn>model\nThe Beatles<EOS>",
            "<start_of_turn>user\nAnswer the following query considering the provided context.\n\nContext: The Beatles wrote Tiny Dancer. \nContext weight: 0.00\nQuery: Who wrote Tiny Dancer?<end_of_turn>\n<start_of_turn>model\nElton John<EOS>",
        ]
        assert expected_train == actual_train

        actual_eval = format_prompts(dataset, "<EOS>", GEMMA_PROMPT, do_eval=True)
        expected_eval = [
            "<start_of_turn>user\nAnswer the following query considering the provided context.\n\nContext: The Beatles wrote Tiny Dancer. \nContext weight: 1.00\nQuery: Who wrote Tiny Dancer?<end_of_turn>\n<start_of_turn>model\n",
            "<start_of_turn>user\nAnswer the following query considering the provided context.\n\nContext: The Beatles wrote Tiny Dancer. \nContext weight: 0.00\nQuery: Who wrote Tiny Dancer?<end_of_turn>\n<start_of_turn>model\n",
        ]
        assert expected_eval == actual_eval
