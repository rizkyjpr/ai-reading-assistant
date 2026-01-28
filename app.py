import streamlit as st
from processor import process_document
from logic import ask_question

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Reading Assistant", page_icon="📚")

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- SIDEBAR SETUP ---
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        if st.session_state.get("last_file") != uploaded_file.name:
            with st.spinner("⏳ Memproses buku..."):
                st.session_state.vector_db = process_document(uploaded_file)
                st.session_state.last_file = uploaded_file.name
            st.success("✅ Berhasil dimuat!")

# --- MAIN INTERFACE ---
st.title("📚 AI Reading Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT ---
if query := st.chat_input("Tanya sesuatu..."):
    if st.session_state.vector_db is None:
        st.error("Upload buku dulu ya!")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                history = st.session_state.messages[:-1]
                answer, sources = ask_question(
                    query, st.session_state.vector_db, history
                )

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
