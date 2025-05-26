import faiss
import numpy
import numpy as np
import pickle
from langchain_community.embeddings import OpenAIEmbeddings
import os

INDEX_PATH="vector_store/index.faiss"
META_PATH="vector_store/meta.pkl"

def save_job_description(text: str):
    os.makedirs("vector_store", exist_ok=True)
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    vector = embedder.embed_query(text=text)
    index = faiss.IndexFlatL2(len(vector))
    index.add(np.array([vector]).astype("float32"))
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"text": text}, f)
    return True

def load_job_description():
    if os.path.exists(META_PATH):
        with open(META_PATH, 'rb') as f:
            meta = pickle.load(f)
            return meta
    else:
        return "not found"