import json
import os

from langchain import SagemakerEndpoint
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.prompts import PromptTemplate
from langchain.retrievers import AmazonKendraRetriever

# ***** CONFIGURABLE PARAMETERS *****
REGION_NAME = os.environ.get("REGION_NAME", "us-east-1")
ENDPOINT_NAME = os.environ["FALCON_ENDPOINT"]
TEMPERATURE = os.environ.get("TEMPERATURE", 0.3)
MAX_NEW_TOKENS = os.environ.get("MAX_NEW_TOKENS", 512)
TOP_K = os.environ.get("TOP_K", 250)
TOP_P = os.environ.get("TOP_P", .3)
STOP_SEQUENCES = os.environ.get("STOP_SEQUENCES", ["\nUser:", "<|endoftext|>", "</s>"])
KENDRA_INDEX_ID = os.environ["KENDRA_INDEX_ID"]
MAX_HISTORY_LENGTH = 5
# ******************************************************************


def build_chain():
    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            prompt = prompt[:1023]
            input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
            print("input_str", input_str)
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            print(response_json)
            return response_json[0]["generated_text"]

    content_handler = ContentHandler()

    llm = SagemakerEndpoint(
        endpoint_name=ENDPOINT_NAME,
        region_name=REGION_NAME,
        model_kwargs={
            "temperature": TEMPERATURE,
            "max_length": 10000,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": True,
            "top_p": TOP_P,
            "repetition_penalty": 1.03,
            "stop": STOP_SEQUENCES,
        },
        content_handler=content_handler,
    )

    retriever = AmazonKendraRetriever(index_id=KENDRA_INDEX_ID)

    prompt_template = """
    The following is a friendly conversation between a human and an AI. 
    The AI is talkative and provides lots of specific details from its context.
    If the AI does not know the answer to a question, it truthfully says it 
    does not know.
    {context}
    Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
    if not present in the document. 
    Solution:"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    condense_qa_template = """
    Given the following conversation and a follow up question, rephrase the follow up question 
    to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=standalone_question_prompt,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    return qa


def run_chain(chain, prompt: str, history=None):
    if history is None:
        history = []
    return chain({"question": prompt, "chat_history": history})
