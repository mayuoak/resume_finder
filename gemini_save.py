# ai_resume_matcher_agent.py

import os
import pickle
import faiss
import numpy as np
import ast
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import fitz
import google.generativeai as genai
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import json
import re

# Load environment
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Constants
INDEX_PATH = "vector_store/gemini.index.faiss"
META_PATH = "vector_store/gemini.meta.pkl"


class Resume(BaseModel):
    name: str
    description: str

class ResumeList(BaseModel):
    resumes: list[Resume]

class Score(BaseModel):
    score: float = Field(..., ge=0, le=100, description="score in float format")
    metric: str = Field(..., description="metric name")


# ============================= EMBEDDING =============================
class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str):
        return self.model.encode(text)

# ============================= STORAGE =============================
class IndexStore:
    def __init__(self, index_path=INDEX_PATH, meta_path=META_PATH):
        self.index_path = index_path
        self.meta_path = meta_path

    def save(self, text, vector):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        index = faiss.IndexFlatL2(len(vector))
        index.add(np.array([vector]).astype("float32"))
        faiss.write_index(index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({"text": text, "embedding": vector}, f)

    def load(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'rb') as f:
                meta = pickle.load(f)
                return meta.get("text", ""), meta.get("embedding", None)
        return "", None

# ============================= LLM WRAPPER =============================
class GeminiLLM:
    def __init__(self, model_name="models/gemini-1.5-pro"):
        self.model = genai.GenerativeModel(model_name)

    def predict(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text.strip()

# ============================= PDF =============================
class PDFReader:
    @staticmethod
    def extract_text(file_path):
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        return text.strip()

    @staticmethod
    def read_resumes(dir_path):
        resumes = []
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(dir_path, filename)
                name = os.path.splitext(filename)[0]
                description = PDFReader.extract_text(full_path)
                resumes.append({"name": name, "description": description})
        return resumes

# ============================= TOOLS =============================
embedder = Embedder()
index_store = IndexStore()
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.3,
    convert_system_message_to_human=True
)

@tool
def save_job_description_tool(filepath: str) -> str:
    """Save a job description from a text file and generate its embedding."""
    with open(filepath, 'r') as f:
        text = f.read()
    vector = embedder.encode(text)
    index_store.save(text, vector)
    return "Job description saved."


@tool
def read_resumes_from_directory(resume_dir: str) -> str:
    """Reads all PDF resumes from a directory and returns them as a JSON string."""
    resumes = PDFReader.read_resumes(resume_dir)
    return json.dumps(resumes)

@tool
def extract_text_from_pdf_tool(filepath: str) -> str:
    """Extract and return text content from a PDF file."""
    return PDFReader.extract_text(filepath)


def clean_llm_response(text: str) -> str:
    """
    Cleans LLM responses by removing:
    - Markdown-style code block markers (e.g., ```json, ```)
    - Leading/trailing backticks (` or ```)
    - Triple quotes if present
    - Stray whitespace and newlines
    - JSON string inside quoted string
    """
    text = text.strip()

    # Remove enclosing triple quotes or single quotes
    text = text.strip('"""').strip("'''").strip("'").strip('"')

    # Remove markdown code fences like ```json or ```
    lines = text.splitlines()
    lines = [line for line in lines if not line.strip().startswith("```")]
    text = "\n".join(lines)

    # Remove any leading/trailing backticks or triple backticks
    text = re.sub(r"^`{1,3}", "", text)
    text = re.sub(r"`{1,3}$", "", text)

    # Remove any surrounding quotes again
    text = text.strip('"""').strip("'''").strip("'").strip('"').strip()

    return text


@tool
def score_resumes_tool(resume_list_str: str) -> str:
    """Score a list of resumes (JSON string) against the stored job description using Gemini and cosine similarity."""

    jd_text, jd_vector = index_store.load()
    if jd_vector is None:
        return "Job description not found."

    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    try:
        cleaned = clean_llm_response(resume_list_str)
        parsed = json.loads(cleaned)
        resume_list = ResumeList(resumes=[Resume(**r) for r in parsed])
    except Exception as e:
        return f"Error parsing resume json: {e}"

    results = []
    for resume in resume_list.resumes:
        name = resume.name
        content = resume.description
        try:
            resume_vector = embedder.encode(content)
            similarity = cosine_similarity(jd_vector, resume_vector)
            cosine_score = Score(metric="cosine", score=round(similarity * 100, 2))
        except Exception as e:
            score = f"Error: {e}"

        prompt = f"""
        You are a technical recruiter.
        Job Description:
        {jd_text}

        Resume:
        {content}

        Task:
        Based on the job description, rate the resume on a scale of 0 to 100.
        Give output in clear json structure as follows:
        {{
        "score": 87
        }} 
        Do not add any commentary.
        """
        try:
            response = gemini_llm.predict(prompt)
            temp = json.loads(clean_llm_response(response))
            gemini_score = Score(metric="gemini", score=float(temp.get("score", 0)))
        except Exception as e:
            raise e
        # final_score = f"{gemini_score.score} LLM, {cosine_score.score} cosine"
        final_score = 0.7 * gemini_score.score + 0.3 * cosine_score.score
        results.append((name, final_score))

    df = pd.DataFrame(results, columns=["Name", "Score"])
    df = df.sort_values(by="Score", ascending=False)
    return df.to_markdown(index=False)

# ============================= AGENT =============================
tools = [
    save_job_description_tool,
    extract_text_from_pdf_tool,
    score_resumes_tool,
    read_resumes_from_directory,
]

agent_executor = initialize_agent(
    tools,
    gemini_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ============================= MAIN =============================
def main():
    jd_path = "/Users/mayureshoak/Documents/git_repos/resume_finder/data/job_description.txt"
    resume_dir = "/Users/mayureshoak/Documents/git_repos/resume_finder/data/resumes"

    result = agent_executor.run(f"""
    Please perform the following steps:
    1. Load and save the job description from the file: {jd_path}
    2. Extract and score all resumes found in the folder: {resume_dir}
    3. Use the score_resumes_tool and return the result as a markdown table. 
    job description is loaded internally in score resumes tool.
    """)
    print(result)

if __name__ == "__main__":
    main()
