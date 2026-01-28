import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def ask_question(query, vector_db, chat_history):
    # --- AUTHENTICATION & CONFIGURATION ---
    my_api_token = st.secrets["HF_TOKEN"]

    # --- INITIALIZE MODEL ---
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token=my_api_token,
        max_new_tokens=1024,
    )

    chat_model = ChatHuggingFace(llm=llm)

    # --- VECTOR SEARCH ---
    docs = vector_db.max_marginal_relevance_search(
        query, k=7, fetch_k=20, lambda_param=0.5
    )

    # --- CONTEXT PREPARATION ---
    context_with_metadata = ""

    for doc in docs:
        page_num = doc.metadata.get("page", "Tidak diketahui")
        context_with_metadata += f"\n[Halaman {page_num}]: {doc.page_content}\n"

    # --- SYSTEM PROMPT DEFINITION ---
    messages = [
        SystemMessage(
            content=f"""Adopt the role of a Meta-Cognitive Reasoning Expert. Your task is to answer questions based STRICTLY on the provided PDF {context_with_metadata}.

                        For complex queries, follow this Recursive Reasoning Loop:

                        1. DECOMPOSE: Break the question into logical sub-steps.
                        2. RETRIEVE & SOLVE: For each sub-step, find exact quotes from the context. Assign a confidence score (0.0-1.0) based on how clearly the PDF supports the claim.
                        3. CRITICIZE: Act as a strict skeptic. 
                            - Is this information explicitly written in the PDF? 
                            - If the query is unrelated to the PDF content (e.g., cooking, coding, or general chat), you MUST state: "Sorry the information aren't included in the docs."
                            - DO NOT use your internal knowledge about the world. 
                            - ABSOLUTE RULE: Do not explain why you are declining; just state that the information is missing. This prevents the model from hallucinating a "general" version of the answer.
                        4. REFLECT & RETRY: If any sub-step confidence is < 0.85, RE-SCAN the context for better evidence. If evidence is truly missing, admit it.
                        5. SYNTHESIZE: Combine the verified facts into a final, cohesive answer.

                        For simple questions (e.g., greetings or single facts), provide a direct, concise answer.

                        OUTPUT FORMAT:
                        ---
                        [Final Answer]
                        (Your synthesized response here)

                        [Reasoning Metrics]
                        - Confidence: (0.0 - 1.0)
                        - Verification: (State if fact-checked against PDF)
                        - Caveats: (Any missing data or assumptions)
                        --- 
            """
        ),
        HumanMessage(content=query),
    ]

    # --- CHAT HISTORY INTEGRATION ---
    for msg in chat_history[-5:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # --- APPEND CURRENT QUERY ---
    messages.append(HumanMessage(content=query))

    # --- EXECUTE ---
    response = chat_model.invoke(messages)

    return response.content, docs
