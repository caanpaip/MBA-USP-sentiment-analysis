import json

import requests
from bs4 import BeautifulSoup
from loguru import logger


def get_data():
    """Retrieves the raw lines of the `openai_info.py` file from GitHub.

    This function scrapes the GitHub page hosting the specified Python file,
    extracts the relevant JavaScript content containing the raw file data,
    and returns the data as a list of strings.

    Returns:
        list: A list of strings, where each string represents a line from the
              `openai_info.py` file.  Returns None if an error occurs during
              the retrieval or parsing process.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request to GitHub fails.
        IndexError: If the expected JavaScript content or raw lines data is not found on the scraped page.
        json.JSONDecodeError: If the extracted JavaScript content is not valid JSON.
    """
    link = "https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/callbacks/openai_info.py"
    response = requests.get(link)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    scripts = soup.find_all("script")
    novo = [script.string for script in scripts if script.string]

    cost = [x for x in novo if "MODEL_COST_PER_1K_TOKENS" in x and "" in x]
    data = json.loads(cost[0])["payload"]["blob"]["rawLines"]

    return data


def extract_model_cost_data(data):
    """Extracts the MODEL_COST_PER_1K_TOKENS dictionary from a list of strings.

    Args:
        data: A list of strings representing Python code.

    Returns:
        A dictionary representing MODEL_COST_PER_1K_TOKENS, or None if not found.
    """

    start_index = -1
    end_index = -1

    for i, line in enumerate(data):
        if line.startswith("MODEL_COST_PER_1K_TOKENS = {"):
            start_index = i
        elif start_index != -1 and line == "}":
            end_index = i
            break

    if start_index == -1 or end_index == -1:
        return None

    model_cost_data = {}
    for line in data[start_index + 1 : end_index]:
        line_ = line.strip()
        if not line_ or line_.startswith("#"):
            continue  # Skip empty lines and comments
        key, value = line_.split(":", 1)
        key = key.strip().replace('"', "")
        value = value.strip().replace(",", "")
        model_cost_data[key] = value

    # Tirando aquele que tem '#' com o modelo na linha
    model_cost = {}
    for x, y in model_cost_data.items():
        model_cost[x] = y.split("#")[0].strip()

    return model_cost


def extract_and_save_model_cost_data(data, output_filename="model_costs.py"):
    """Extracts MODEL_COST_PER_1K_TOKENS and saves it to a .py file.

    Args:
        data: A list of strings representing the Python code.
        output_filename: The name of the output .py file.
    """

    model_costs = extract_model_cost_data(data)  # Use the function from the previous response

    if model_costs:
        total_items = len(model_costs)
        with open(output_filename, "w", encoding="utf-8") as f:
            # f.write("# https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/callbacks/openai_info.py")
            # f.write("\n\n")
            # f.write("MODEL_COST_PER_1K_TOKENS = {\n")
            f.write("{\n")
            for n, items in enumerate(model_costs.items(), start=1):
                key, value = items
                if n != total_items:
                    f.write(f'    "{key}": {value},\n')
                else:
                    f.write(f'    "{key}": {value}\n')

            f.write("}\n")

        logger.info(f"Model cost data saved to {output_filename}")
    else:
        logger.info("MODEL_COST_PER_1K_TOKENS not found in the data.")
