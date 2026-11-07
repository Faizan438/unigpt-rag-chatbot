import streamlit as st
import chromadb
import os
import glob

from rag import ask
import rag
st.write("rag file:", rag.__file__, "| model:", rag.GEMINI_MODEL)

st.set_page_config(page_title="UniGPT")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db")

pdf_files = glob.glob(os.path.join(BASE_DIR, "*.pdf"))

db = chromadb.PersistentClient(path=DB_PATH)
col = db.get_or_create_collection(name="unigpt")
if col.count() == 0:
    import subprocess
    import sys
    st.write("Database khaali hai, ingest.py chala raha hun...")
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "ingest.py")], check=False)
    col = db.get_or_create_collection(name="unigpt")
    st.write(f"Ingest ke baad chunks: {col.count()}")

st.title("🎓 UniGPT Chatbot")

st.write(f"PDF files found: {len(pdf_files)}")
st.write(f"Total chunks in database: {col.count()}")

for pdf in pdf_files:
    st.write(f"📄 {os.path.basename(pdf)}")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, message in st.session_state.chat:
    st.chat_message(role).write(message)

if q := st.chat_input("Ask about admissions, fees, visa process or exchange programs..."):
    st.session_state.chat.append(("user", q))
    st.chat_message("user").write(q)

    answer, sources = ask(q)

    st.session_state.chat.append(("assistant", answer))
    st.chat_message("assistant").write(answer)

    with st.expander("Sources"):
        for s in sources:
            st.write(f"{s['source']} - page {s['page']}")
