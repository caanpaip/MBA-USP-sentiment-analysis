import json
import os

import tiktoken
from loguru import logger
from openai import OpenAI

from src.utils.openaiwrapper.get_openai_cost import (
    extract_and_save_model_cost_data,
    get_data,
)

file_name_path = __file__
file_path = os.path.dirname(os.path.abspath(file_name_path))
output_cost_py = os.path.join(file_path, "model_costs.json")

logger.info("Update cost from langchain")
# https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/callbacks/openai_info.py
data = get_data()
extract_and_save_model_cost_data(data, output_cost_py)


class OpenAIWrapper(OpenAI):
    def __init__(self):
        self.openai_client = OpenAI()

    @staticmethod
    def calculate_llm_cost(model: str, tokens_cost: dict):
        MODEL_COST_PER_1K_TOKENS = json.load(open(output_cost_py, "r", encoding="utf-8"))
        model_completion = model + "-completion"
        tokens = tokens_cost

        # compute
        input_tokens = tokens["prompt_tokens"] / 1000
        output_tokens = tokens["completion_tokens"] / 1000

        cost_input = MODEL_COST_PER_1K_TOKENS[model]
        cost_output = MODEL_COST_PER_1K_TOKENS[model_completion]

        total_cost = input_tokens * cost_input + output_tokens * cost_output

        # add value
        tokens_cost["total_cost"] = total_cost

        return tokens_cost

    def embedding(self, text: str, model: str = "text-embedding-ada-002"):
        result = self.openai_client.embeddings.create(input=text, model=model)

        result_emb = [emb.embedding for emb in result.data]
        result_usage = result.usage.__dict__

        return result_emb, result_usage

    def chat_completion(self, *args, **kwargs):
        result = self.openai_client.chat.completions.create(*args, **kwargs)

        result_chat = result.choices[0].message.content
        result_usage = result.usage.__dict__

        result_usage = self.calculate_llm_cost(result.model, result_usage)

        self.full_chat_result = result

        return result_chat, result_usage

    def completion(self, *args, **kwargs):
        result = self.openai_client.completions.create(*args, **kwargs)
        result_chat = result.choices[0].text
        result_usage = result.usage.__dict__

        result_usage = self.calculate_llm_cost(result.model, result_usage)

        return result_chat, result_usage


def token_counter(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
