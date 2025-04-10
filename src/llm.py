from src.utils.geminiwrapper.gemini_wrapper import GeminiWrapper
from src.utils.openaiwrapper.openai_wrapper import OpenAIWrapper


class multi_llm:
    def __init__(self, temperature: int = 0, top_p: int = 0, top_k: int = 0, max_tokens: int = 852):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens

    def response(self, model: str, user_message: str, system_message: str = "", json_output: bool = False):
        if "gpt" in model:
            call_openai = OpenAIWrapper()

            response_format = {"type": "json_object"} if json_output else {"type": "text"}

            result, usage = call_openai.chat_completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                model=model,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                response_format=response_format,
            )

            return result, usage

        else:
            call_gemini = GeminiWrapper(self.temperature, self.top_p, self.top_k, self.max_tokens)

            result = call_gemini.response(model, user_message, system_message, json_output)
            usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0, "total_cost": 0}

            return result, usage
