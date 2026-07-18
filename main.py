import re

import requests
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OLLAMA_URL = "http://localhost:11434/api/generate"


def extract_pages(pdf_file):
    reader = PdfReader(pdf_file)
    return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]


def make_chunks(pages, chunk_size=1200, overlap=200):
    chunks = []
    for page_number, text in pages:
        clean_text = re.sub(r"\s+", " ", text).strip()
        start = 0
        while start < len(clean_text):
            chunk = clean_text[start : start + chunk_size]
            if chunk:
                chunks.append({"page": page_number, "text": chunk})
            start += chunk_size - overlap
    return chunks


def retrieve(question, chunks, top_k=4):
    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    document_vectors = vectorizer.fit_transform(texts)
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, document_vectors)[0]
    best_indices = scores.argsort()[::-1][:top_k]
    return [(chunks[index], float(scores[index])) for index in best_indices if scores[index] > 0]


def ask_ollama(question, matches, model):
    context = "\n\n".join(
        f"[Page {chunk['page']}] {chunk['text']}" for chunk, _ in matches
    )
    prompt = f"""Answer the question using only the PDF context below.
If the answer is not in the context, say: "I could not find this in the PDF."
Keep the answer clear and cite page numbers like [Page 3].

PDF context:
{context}

Question: {question}
Answer:"""
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


st.set_page_config(page_title="AI PDF Question Answering", page_icon="📄")
st.title("📄 AI PDF Question Answering")
st.caption("Upload a text-based PDF and ask questions. Everything runs locally with Ollama.")

with st.sidebar:
    model = st.text_input("Ollama model", value="llama3.2:3b")
    top_k = st.slider("Retrieved passages", 2, 8, 4)

pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    try:
        pages = extract_pages(pdf_file)
        chunks = make_chunks(pages)
        if not chunks:
            st.error("No text was found. This may be a scanned PDF that needs OCR.")
            st.stop()
        st.success(f"Processed {len(pages)} pages into {len(chunks)} passages.")

        question = st.text_input("Ask a question about the PDF")
        if st.button("Get answer", type="primary", disabled=not question):
            matches = retrieve(question, chunks, top_k)
            if not matches:
                st.warning("I could not find a relevant passage in the PDF.")
            else:
                with st.spinner("Thinking..."):
                    answer = ask_ollama(question, matches, model)
                st.subheader("Answer")
                st.write(answer)
                with st.expander("Retrieved passages"):
                    for chunk, score in matches:
                        st.markdown(f"**Page {chunk['page']} · relevance {score:.2f}**")
                        st.write(chunk["text"])
    except requests.ConnectionError:
        st.error("Ollama is not running. Start it with `ollama serve`.")
    except requests.HTTPError as error:
        st.error(f"Ollama error: {error}. Make sure the selected model is installed.")
    except Exception as error:
        st.error(f"Could not process the PDF: {error}")

