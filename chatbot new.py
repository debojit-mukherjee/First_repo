import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import PyPDF2

# -------------------------------
# 1. Knowledge Base (initial notes)
# -------------------------------
knowledge_base = [
    {"text": "Python is a beginner-friendly programming language.", "source": "notes"},
    {"text": "SQL is used to query and manage databases.", "source": "notes"},
    {"text": "APIs let applications talk to each other. REST APIs use GET, POST, PUT, DELETE.", "source": "notes"},
    {"text": "SDLC has phases: Requirements, Design, Implementation, Testing, Deployment, Maintenance.", "source": "notes"}
]

# -------------------------------
# 2. Embeddings + FAISS
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode([item["text"] for item in knowledge_base]).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# -------------------------------
# 3. Hugging Face Q&A model
# -------------------------------
try:
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
except Exception as e:
    qa_model = None
    st.warning(f"⚠️ Q&A model not loaded: {e}")

# -------------------------------
# 4. Streamlit App with Multi-Document Upload + Page-Level Citations
# -------------------------------
st.title("Debojit's RAG Chatbot")
st.header("Week 6: Retrieval-Augmented Generation")
st.write("Upload multiple documents (TXT, MD, PDF) and ask questions across all of them!")

# Initialize session state for memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Helper: split text into chunks
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Upload multiple documents
uploaded_files = st.file_uploader("Upload documents", type=["txt", "md", "pdf"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            # Extract text from PDF with page numbers
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text() or ""
                chunks = chunk_text(text, chunk_size=300)
                for chunk in chunks:
                    knowledge_base.append({
                        "text": chunk,
                        "source": f"{uploaded_file.name}, page {page_num}"
                    })
                doc_embeddings = embedder.encode(chunks).astype("float32")
                index.add(doc_embeddings)
        else:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            chunks = chunk_text(text, chunk_size=300)
            for chunk in chunks:
                knowledge_base.append({"text": chunk, "source": uploaded_file.name})
            doc_embeddings = embedder.encode(chunks).astype("float32")
            index.add(doc_embeddings)

    st.success(f"✅ Added {len(uploaded_files)} documents to knowledge base!")

# Toggle for citation style
show_inline = st.checkbox("Show citations inline in answers", value=True)

# Question input
question = st.text_input("Ask me a question:")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question!")
    else:
        # Encode question
        q_embed = embedder.encode([question]).astype("float32")

        # Retrieve top 3 chunks
        distances, indices = index.search(q_embed, k=3)
        retrieved_notes = [knowledge_base[i] for i in indices[0]]

        # Build context with citations
        context_parts = []
        for note in retrieved_notes:
            citation = f"[Source: {note['source']}]"
            context_parts.append(note["text"] + " " + citation)

        context = " ".join(context_parts)

        # Generate answer
        if qa_model:
            try:
                response = qa_model(question=question, context=context)
                polished_answer = response['answer']
                if show_inline:
                    sources_inline = ", ".join(set([note["source"] for note in retrieved_notes]))
                    polished_answer += f"  [Sources: {sources_inline}]"
            except Exception as e:
                polished_answer = f"⚠️ Error generating answer: {e}"
        else:
            polished_answer = "⚠️ Q&A model unavailable."

        # Save to session memory with sources
        st.session_state.chat_history.append({
            "question": question,
            "answer": polished_answer,
            "sources": [note["source"] for note in retrieved_notes]
        })

# Display chat history
if st.session_state.chat_history:
    st.subheader("Conversation History")
    for i, chat in enumerate(st.session_state.chat_history, 1):
        st.write(f"**Q{i}:** {chat['question']}")
        st.write(f"**A{i}:** {chat['answer']}")
        if not show_inline:
            st.write(f"📖 Sources: {', '.join(set(chat['sources']))}")
        st.write("---")
else:
    st.info("No questions asked yet. Start by typing above!")
