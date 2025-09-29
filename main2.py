import os
import glob
from pathlib import Path
from typing import List
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def qa_system(folder: str, question: str) -> str:
    texts = [Path(f).read_text(encoding='utf-8') for f in glob.glob(f"{folder}/**/*.txt", recursive=True)]
    chunks = [text[i:i+500] for text in texts for i in range(0, len(text), 450)]
    embeddings = [client.embeddings.create(input=chunk, model="text-embedding-ada-002").data[0].embedding for chunk in chunks]
    q_emb = client.embeddings.create(input=question, model="text-embedding-ada-002").data[0].embedding
    similarities = [np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb)) for emb in embeddings]
    context = "\n".join([chunks[i] for i in sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Use the following pieces of context to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n" + context},
                  {"role": "user", "content": question}],
        temperature=0
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print(qa_system(input("Folder: "), input("Question: ")))