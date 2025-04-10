import os

import google.generativeai as genai
from google.auth.exceptions import DefaultCredentialsError


class GeminiWrapper:
    def __init__(
        self,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: float = 0.0,
        max_tokens: int = 8192,
    ):
        genai.configure(api_key=os.getenv("GEMINI_KEY"))

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens

    def response(self, model: str, user_message: str, system_message: str, json_output: bool = False):
        response_mime_type = "application/json" if json_output else "text/plain"

        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_tokens,
            "response_mime_type": response_mime_type,
        }

        if not system_message.strip():
            model_gem = genai.GenerativeModel(model_name=model, generation_config=generation_config)

        else:
            model_gem = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                system_instruction=system_message,
            )

        try:
            result = model_gem.generate_content(user_message).text
            return result

        except DefaultCredentialsError:
            raise "Authentication failed. Define enviroment variable GEMINI_KEY "
