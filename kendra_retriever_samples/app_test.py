# Conversational Retrieval QA Chatbot LLM Response Evaluation
# Author: Gary A. Stafford
# Date: 2023-08-21
# Requirements: pip install -r requirements.txt -U
# Usage: Usage: python3 ./app_test.py <flanxl|flanxxl|llama2chat|falcon|bedrockclaude|bedrocktitan|bedrockai21labs|openai|cohere>

import os
import sys

from model_providers import (
    kendra_chat_bedrock_anthropic_claude as bedrockclaude,
    kendra_chat_bedrock_amazon_titan as bedrocktitan,
    kendra_chat_bedrock_ai21_labs as bedrockai21labs,
    kendra_chat_open_ai as openai,
    kendra_chat_flan_xxl as flanxxl,
    kendra_chat_flan_xl as flanxl,
    kendra_chat_llama2_chat as llama2chat,
    kendra_chat_falcon as falcon,
    kendra_chat_cohere as cohere,
)


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "flanxl":
            llm_app = flanxl
            llm_chain = flanxl.build_chain()
        elif sys.argv[1] == "flanxxl":
            llm_app = flanxxl
            llm_chain = flanxxl.build_chain()
        elif sys.argv[1] == "openai":
            llm_app = openai
            llm_chain = openai.build_chain()
        elif sys.argv[1] == "llama2chat":
            llm_app = llama2chat
            llm_chain = llama2chat.build_chain()
        elif sys.argv[1] == "falcon":
            llm_app = falcon
            llm_chain = falcon.build_chain()
        elif sys.argv[1] == "bedrockclaude":
            llm_app = bedrockclaude
            llm_chain = bedrockclaude.build_chain()
        elif sys.argv[1] == "bedrocktitan":
            llm_app = bedrocktitan
            llm_chain = bedrocktitan.build_chain()
        elif sys.argv[1] == "bedrockai21labs":
            llm_app = bedrockai21labs
            llm_chain = bedrockai21labs.build_chain()
        elif sys.argv[1] == "cohere":
            llm_app = cohere
            llm_chain = cohere.build_chain()
        else:
            raise Exception("Unsupported LLM: ", sys.argv[1])
    else:
        raise Exception(
            "Models: flanxl|flanxxl|llama2chat|falcon|bedrockclaude|bedrocktitan|bedrockai21labs|openai|cohere"
        )

    chain = llm_app

    prompt = os.environ.get("INPUT", "")
    model_ref_id = os.environ.get("MODEL_REFERENCE_ID", "")
    answer_id = os.environ.get("ANSWER_ID", 0)

    result = chain.run_chain(llm_chain, prompt, [])

    answer = f'{model_ref_id} = """\n{result["answer"]}\n"""\n\n\n'

    print(answer)
    with open(f"answer_{answer_id}.txt", "a") as file1:
        file1.write(answer)


if __name__ == "__main__":
    main()
