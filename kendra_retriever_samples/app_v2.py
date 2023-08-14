# Conversational Retrieval QA Chatbot, built using Langflow and Streamlit
# Author: Gary A. Stafford
# Date: 2023-08-13
# Usage: streamlit run app_v2.py <anthropic|flanxl|flanxxl|openai|llama2chat> --server.runOnSave true

import sys

import streamlit as st

import kendra_chat_anthropic as anthropic
import kendra_chat_flan_xl as flanxl
import kendra_chat_flan_xxl as flanxxl
import kendra_chat_llama2_chat as llama2chat
import kendra_chat_open_ai as openai

# ****** CONFIGURABLE PARAMETERS ******
USER_ICON = "images/user-icon.png"
AI_ICON = "images/ai-icon.png"
HEADER_TITLE = "An AI App powered by Amazon Kendra"
HEADER_LOGO = "images/ai-icon.png"
PAGE_TITLE = "AI Chatbot"
PAGE_FAVICON = "images/ai-icon.png"
SHOW_DOC_SOURCES = True
TEXT_INPUT_PROMPT = "You are talking to an AI, ask any question."
TEXT_INPUT_PLACEHOLDER = "What is Amazon SageMaker?"
SHOW_SAMPLE_QUESTIONS = False
MAX_HISTORY_LENGTH = 5
PROVIDER_MAP = {
    "openai": "Open AI",
    "anthropic": "Anthropic",
    "flanxl": "Flan XL",
    "flanxxl": "Flan XXL",
    "llama2chat": "Llama-2 13B Chat",
}


def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_FAVICON,
    )

    if "llm_chain" not in st.session_state:
        if len(sys.argv) > 1:
            if sys.argv[1] == "anthropic":
                st.session_state["llm_app"] = anthropic
                st.session_state["llm_chain"] = anthropic.build_chain()
            elif sys.argv[1] == "flanxl":
                st.session_state["llm_app"] = flanxl
                st.session_state["llm_chain"] = flanxl.build_chain()
            elif sys.argv[1] == "flanxxl":
                st.session_state["llm_app"] = flanxxl
                st.session_state["llm_chain"] = flanxxl.build_chain()
            elif sys.argv[1] == "openai":
                st.session_state["llm_app"] = openai
                st.session_state["llm_chain"] = openai.build_chain()
            elif sys.argv[1] == "llama2chat":
                st.session_state["llm_app"] = llama2chat
                st.session_state["llm_chain"] = llama2chat.build_chain()
            else:
                raise Exception("Unsupported LLM: ", sys.argv[1])
        else:
            raise Exception(
                "Usage: streamlit run app.py <anthropic|flanxl|flanxxl|openai|llama2chat>"
            )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "chats" not in st.session_state:
        st.session_state.chats = [{"id": 0, "question": "", "answer": ""}]

    if "questions" not in st.session_state:
        st.session_state.questions = []

    if "answers" not in st.session_state:
        st.session_state.answers = []

    if "input" not in st.session_state:
        st.session_state.input = ""

    st.markdown(
        """
            <style>
                   .block-container {
                        padding-top: 32px;
                        padding-bottom: 32px;
                        padding-left: 0;
                        padding-right: 0;
                    }
                    .element-container img {
                        background-color: #000000;
                    }
                    .main-header {
                        font-size: 24px;
                    }
                    #MainMenu {
                        visibility: visible;
                    }
                    footer {
                        visibility: visible;
                    }
                    header {
                        visibility: visible;
                    }
            </style>
            """,
        unsafe_allow_html=True,
    )

    clear = write_top_bar()

    if clear:
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.input = ""
        st.session_state["chat_history"] = []

    with st.container():
        for q, a in zip(st.session_state.questions, st.session_state.answers):
            write_user_message(q)
            write_chat_message(a, q)

    st.markdown("---")

    if SHOW_SAMPLE_QUESTIONS:
        with st.expander("Click here for sample questions..."):
            st.markdown(
                """
                    - What is Amazon SageMaker?
                    - What are some of its major features?
                    - How do I get started using it?
                    - Tell me about Amazon SageMaker Feature Store.
                    - What does the Inference Recommender do?
                    - What is Autopilot?
                    - How much does SageMaker cost?
                """
            )
        st.markdown(" ")

    st.text_input(
        TEXT_INPUT_PROMPT,
        placeholder=TEXT_INPUT_PLACEHOLDER,
        key="input",
        on_change=handle_input,
    )


def write_top_bar():
    col1, col2, col3 = st.columns([1, 10, 2])
    with col1:
        st.image(HEADER_LOGO, use_column_width="always")
    with col2:
        selected_provider = sys.argv[1]
        if selected_provider in PROVIDER_MAP:
            provider = PROVIDER_MAP[selected_provider]
        else:
            provider = selected_provider.capitalize()
        st.markdown(f"#### {HEADER_TITLE} and # {provider}!")
    with col3:
        clear = st.button("Clear Chat")
    return clear


def handle_input():
    input = st.session_state.input
    question_with_id = {"question": input, "id": len(st.session_state.questions)}
    st.session_state.questions.append(question_with_id)

    chat_history = st.session_state["chat_history"]
    if len(chat_history) == MAX_HISTORY_LENGTH:
        chat_history = chat_history[:-1]

    llm_chain = st.session_state["llm_chain"]
    chain = st.session_state["llm_app"]
    result = chain.run_chain(llm_chain, input, chat_history)
    answer = result["answer"]
    chat_history.append((input, answer))

    document_list = []
    if "source_documents" in result:
        for d in result["source_documents"]:
            if not (d.metadata["source"] in document_list):
                document_list.append((d.metadata["source"]))

    st.session_state.answers.append(
        {
            "answer": result,
            "sources": document_list,
            "id": len(st.session_state.questions),
        }
    )
    st.session_state.input = ""


def write_user_message(md):
    col1, col2 = st.columns([1, 12])

    with col1:
        st.image(USER_ICON, use_column_width="always")
    with col2:
        st.warning(md["question"])


def render_answer(answer):
    col1, col2 = st.columns([1, 12])
    with col1:
        st.image(AI_ICON, use_column_width="always")
    with col2:
        st.info(answer["answer"])


def render_sources(sources):
    col1, col2 = st.columns([1, 12])
    with col2:
        with st.expander("Sources"):
            for s in sources:
                st.write(s)


def write_chat_message(md, q):
    chat = st.container()
    with chat:
        render_answer(md["answer"])
        if SHOW_DOC_SOURCES:
            render_sources(md["sources"])


if __name__ == "__main__":
    main()
