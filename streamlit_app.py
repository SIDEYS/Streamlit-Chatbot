import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import numpy as np
import requests

# ---------- Page Config ----------
st.set_page_config(page_title="Mankir Chatbot", page_icon="🌱", layout="centered")

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Roboto Slab', serif;
            background-color: #121212;
            color: #e0ffe0;
        }

        .stTextInput input {
            border: 2px solid #00b894;
            border-radius: 8px;
            padding: 10px;
            background-color: #1f1f1f;
            color: #fff;
        }

        .stButton button {
            background-color: #00b894;
            color: white;
            font-weight: bold;
            border-radius: 6px;
            padding: 0.5rem 1.2rem;
            border: none;
        }

        .chatbox {
            background-color: #2d3436;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 10px;
        }

        .bot-response {
            color: #dfe6e9;
            font-size: 16px;
            line-height: 1.6;
        }

        h1, h2 {
            color: #55efc4;
        }

        .expander {
            background-color: #2c2c2c !important;
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# ---------- Extract Q&A Pairs ----------
def extract_qa_from_pdf(filepath):
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    lines = [line.strip() for line in full_text.split('\n') if line.strip()]
    qa_list = []
    current_q = None
    current_a = []

    for line in lines:
        if line.lower().startswith("q:"):
            if current_q and current_a:
                qa_list.append((current_q, " ".join(current_a)))
                current_a = []
            current_q = line[2:].strip()
        elif line.lower().startswith("a:"):
            current_a.append(line[2:].strip())
        else:
            current_a.append(line.strip())

    if current_q and current_a:
        qa_list.append((current_q, " ".join(current_a)))

    return qa_list

# ---------- Build KB ----------
@st.cache_data
def build_knowledge_base():
    pdf_path = "./pdfs/QuestionAnswers.pdf"
    qa_pairs = extract_qa_from_pdf(pdf_path)
    questions = [q for q, _ in qa_pairs]
    embeddings = embed_model.encode(questions)
    return qa_pairs, embeddings

qa_pairs, question_embeddings = build_knowledge_base()

# ---------- Rephrase LLM ----------
def rephrase_with_llm(user_question, base_answer):
    prompt = (
        f"You are a helpful assistant for a plant store. "
        f"Answer the user's question in a friendly tone using the answer below.\n\n"
        f"User Question: {user_question}\n"
        f"Reference Answer: {base_answer}\n\n"
        f"Response:"
    )
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "tinyllama", "prompt": prompt, "stream": False}
        )
        return response.json()["response"].strip() if response.status_code == 200 else base_answer
    except:
        return base_answer

# ---------- Title & Tagline ----------
st.markdown("<h1>🌿 Mankir Chatbot</h1>", unsafe_allow_html=True)
st.markdown("💬 *Ask me anything about indoor plants, FRP decor, bamboo accessories, or garden planning!*")

# ---------- Brand Description ----------
with st.expander("🌍 About MANKIR"):
    st.markdown("""
    **At MANKIR**, we are dedicated to driving sustainability and innovation across industries.  
    Our expertise spans creating eco-friendly environments, offering advanced agrotech solutions,  
    and providing comprehensive veterinary services.

    We transform spaces with our state-of-the-art green living designs, including **Live Green Walls**,  
    **Vertical Gardens**, and **durable planters**, all aimed at promoting a vibrant, sustainable lifestyle.

    🌱 *Join us in our mission to make the world a greener, healthier place.*
    """)

# ---------- Chatbot Interaction ----------
user_input = st.text_input("🧠 Type your question here...")

if user_input:
    query_embedding = embed_model.encode([user_input])
    scores = cosine_similarity(query_embedding, question_embeddings)[0]
    top_idx = int(np.argmax(scores))
    top_score = float(scores[top_idx])
    matched_q, matched_a = qa_pairs[top_idx]

    SIMILARITY_THRESHOLD = 0.65

    if top_score >= SIMILARITY_THRESHOLD:
        llm_answer = rephrase_with_llm(user_input, matched_a)
        st.markdown("🤖 **Bot Response**", unsafe_allow_html=True)
        st.markdown(f"<div class='chatbox bot-response'>{llm_answer}</div>", unsafe_allow_html=True)
    else:
        st.warning("😕 I couldn’t find a close match. Try rephrasing your question.")
        st.markdown("🌟 **Here's what I can help with:**")
        for i, (q, _) in enumerate(qa_pairs[:5]):
            st.markdown(f"🔹 {q}")
