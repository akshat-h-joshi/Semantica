import re

QUERY_EXPANSIONS = {
    "llm": [
        "large language model",
        "large language models",
        "foundation model",
        "transformer language model",
    ],
    "nlp": ["natural language processing"],
    "rlhf": ["reinforcement learning from human feedback"],
    "vit": ["vision transformer"],
    "cv": ["computer vision"],
}

def expand_query(query: str) -> list[str]:
    query = query.lower().strip()
    tokens = re.findall(r"\w+", query)

    expanded = set(tokens)

    for token in tokens:
        if token in QUERY_EXPANSIONS:
            expanded.update(QUERY_EXPANSIONS[token])

    return list(expanded)