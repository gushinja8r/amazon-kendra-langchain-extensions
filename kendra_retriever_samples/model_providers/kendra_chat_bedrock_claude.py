import os

from langchain.chains import ConversationalRetrievalChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.retrievers import AmazonKendraRetriever

# ***** CONFIGURABLE PARAMETERS *****
REGION_NAME = os.environ.get("REGION_NAME", "us-east-1")
MODEL_NAME = os.environ.get("MODEL_NAME", "anthropic.claude-instant-v1")
TEMPERATURE = os.environ.get("TEMPERATURE",1e-10)
MAX_TOKENS_TO_SAMPLE = os.environ.get("MAX_TOKENS_TO_SAMPLE", 512)
TOP_K = os.environ.get("TOP_K", 250)
TOP_P = os.environ.get("TOP_P", 1)
STOP_SEQUENCES = os.environ.get("STOP_SEQUENCES", ["\n\nHuman"])
KENDRA_INDEX_ID = os.environ["KENDRA_INDEX_ID"]
MAX_HISTORY_LENGTH = 5
# ******************************************************************


def build_chain():
    parameters = {
        "max_tokens_to_sample": MAX_TOKENS_TO_SAMPLE,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "stop_sequences": STOP_SEQUENCES,
    }

    llm = Bedrock(
        region_name=REGION_NAME,
        model_id=MODEL_NAME,
        model_kwargs=parameters,
        verbose=True,
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

    # print(f"prompt: {prompt}\n")
    # print(f"standalone_question_prompt: {standalone_question_prompt}\n")

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
