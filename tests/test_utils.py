import unittest as ut
from datasets import Dataset
from utils import ALPACA_PROMPT, format_prompts


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
        print(actual_train)
        # expected =

        # assert expected == actual
