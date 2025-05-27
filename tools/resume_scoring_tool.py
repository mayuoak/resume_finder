from langchain.tools import tool
from tools.vector_store import load_job_description
from langchain_openai import ChatOpenAI
import pandas as pd
import ast
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
import faiss
import os
import numpy as np

load_dotenv()

INDEX_PATH = "vector_store/index.faiss"
META_PATH = "vector_store/meta.pkl"

embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    

@tool
def score_resumes_tool(resumes_str: str):
    """
    Accepts resumes as a list of dictionaries in string format:

    Returns a markdown table of resume scores based on the stored job description.
    """

    try:
        resumes = ast.literal_eval(resumes_str)
    except Exception as e:
        return f"Failed to parse resumes input: {e}"
    
    if not os.path.exists(INDEX_PATH):
        return "FAISS index not found. Please save the job description first."
    
    # Load job description vector from FAISS
    index = faiss.read_index(INDEX_PATH)
    job_vector = index.reconstruct(0).reshape(1, -1).astype("float32")  # single vector

    llm = ChatOpenAI(model_name="gpt-4o")
    # Embed resumes
    resume_vectors = []
    names = []
    for r in resumes:
        names.append(r.get("name", "Unknown"))
        vec = embedder.embed_query(r.get("description", ""))
        resume_vectors.append(vec)
    resume_vectors = np.array(resume_vectors).astype("float32")

    # Compare using L2 distance
    dim = resume_vectors.shape[1]
    scoring_index = faiss.IndexFlatL2(dim)
    scoring_index.add(resume_vectors)
    distances, indices = scoring_index.search(job_vector, len(resumes))

    # Convert to similarity scores
    max_dist = np.max(distances)
    min_dist = np.min(distances)
    scores = 100 - 100 * (distances[0] - min_dist) / (max_dist - min_dist + 1e-6)

    # Generate markdown table
    markdown = "| Rank | Name | Score |\n|------|------|-------|\n"
    for rank, idx in enumerate(indices[0]):
        name = names[idx]
        score = round(scores[rank], 2)
        markdown += f"| {rank + 1} | {name} | {score} |\n"

    return markdown