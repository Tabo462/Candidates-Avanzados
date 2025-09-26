from ollama import generate
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd

emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

kb1 = pd.read_csv("C:/Dev/Roborregos/Candidates Avanzados/PrimerKB.csv")
kb2 = pd.read_csv("C:/Dev/Roborregos/Candidates Avanzados/SegundoKB.csv")

kb = pd.concat([kb1, kb2], ignore_index=True)


def embed(texts):
    return emb_model.encode(texts)

kb_embeddings = embed(kb['question'].tolist())


def top_k(query_emb, kb_emb, k=3):
    sims = cosine_similarity(query_emb, kb_emb)[0]
    top = sims.argsort()[-k:][::-1]
    return [kb['answer'].iloc[i] for i in top]

while True:
    question = input("What is your question? or 'quit'\n")
    if question.lower() == 'quit':
        break
    
    enc_question = embed([question])
    retrieved = top_k(enc_question, kb_embeddings)

    context = "\n".join(retrieved)
    prompt = f"Use this as context:\n{context}\n\nQuestion: {question}\n If no enough context is given, don't say you don't have enough context."

    response = generate('mistral', prompt)
    print(response['response'])