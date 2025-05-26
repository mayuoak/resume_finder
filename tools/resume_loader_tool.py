from langchain.tools import tool
import os
import fitz

def extract_text_from_pdf(file_path: str):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text.strip()


@tool
def load_resume_tool(dir_path: str):
    """
    Loads all PDF resumes from the specified folder and returns a list of dictionaries with:
    - name: name of the candidate
    - description: full text of the resume
    Input: folder path containing PDF resumes (e.g. 'data/resumes')
    """
    resumes = []
    for file_path in os.listdir(dir_path):
        if file_path.lower().endswith(".pdf"):
            full_path = os.path.join(dir_path, file_path)
            name = os.path.splitext(file_path)[0]
            description = extract_text_from_pdf(full_path)
            resumes.append({"name": name, "description": description})

    return resumes