from langchain.tools import tool
from tools.vector_store import save_job_description, load_job_description

@tool
def store_job_description_tool(filepath: str):
    """Reads the job description from filepath and stores it to vector db"""
    with open(filepath, 'r') as f:
        jd = f.read()
    save_job_description(jd)
    return "Job description stored"

@tool
def fetch_job_description(description: str):
    """Fetches job description from vector store"""
    return load_job_description()