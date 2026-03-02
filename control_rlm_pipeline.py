from pydantic import BaseModel

from docling.document_converter import DocumentConverter

import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

class OutputFormat(BaseModel):
    """ desired output format """
    policy_name: str
    category: str
    details: str
    reasoning: str
    confidence_level: float

PROMPT_TEMPLATE = """You are a domain expert at climate change policy extraction, using the following domain specific information. extract the policy_name, category, and details from the below document in a valid json schema. DO NOT use the examples within the <example> </examples> tags


EXAMPLES:

<examples>

policy_name | category | details

EV Fleets Initiative | 	Housing and Infrastructure |	Adapting to the age of electric vehicles. The governor's EV Fleets Initiative requires that 20 percent of all new passenger vehicles in our state fleet are electric.

100 percent carbon-neutral | Environmental and Resources | 	Adapting to carbon-neutral policies. Eighty percent of their power must come from "nonemitting electric generation and electricity from renewable resources.

Clean Buildings Bill | Housing and Infrastructure | Adoption of an energy code that will require new large buildings to be fossil fuel free by winding down the gas system and transitioning to clean, electric alternatives.
</examples>

EXTRACTION CRITERIA:

{domain_knowledge}

DOCUMENT:

{document}

YOUR ANSWER (in valid json):
"""
ALL_DOCUMENTS = False

OUTPUT_FORMAT = OutputFormat

DIRECTORY = "../document_parsing/pdfs/"
CACHE_DIR = "./cache/"


def get_cache_path(file_path):
    """Get the cache file path for a given document."""
    basename = os.path.splitext(os.path.basename(file_path))[0]
    return os.path.join(CACHE_DIR, f"{basename}.txt")


def parse_document(file_path, use_cache=True):
    """Parse a single document and return its markdown content. Uses caching if available."""
    cache_path = get_cache_path(file_path)

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return f.read()

    converter = DocumentConverter()
    result = converter.convert(file_path)
    mrk_down = result.document.export_to_markdown()

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(mrk_down)

    return mrk_down


def parse_documents(directory):
    results = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            mrk_down = parse_document(file_path)
            results.append(mrk_down)
    return results



def extract_policy(document, expert_eliciation, output_format=None):

    """

    input: document, output format
    output: output format filled 

    """

  

    logger = RLMLogger(log_dir="./logs")

    rlm = RLM(
        backend="openai",  # or "portkey", etc.
        backend_kwargs={
            "model_name": "gpt-5.2",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        other_backends=["openai"],
        other_backend_kwargs=[
        {"model_name": "gpt-5-mini"},
        ],
        environment="local",
        environment_kwargs={},
        max_depth=1,
        logger=logger,
        verbose=True,  # For printing to console with rich, disabled by default.
    )

    prompt = PROMPT_TEMPLATE.format(
        domain_knowledge=expert_eliciation,
        document=document
    )

    result = rlm.completion(prompt)

    return result


if __name__ == "__main__":

    # ingest expert elicitation

    expert_eliciation = parse_document("./RLM_proc_instr.pdf")

    # ingest document as markdown
    document = parse_document("./Glasgows_Climate_Plan.pdf")

    result = extract_policy(document, expert_eliciation)



    print(result)
