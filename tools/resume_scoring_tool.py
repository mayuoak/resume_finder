from langchain.tools import tool
from tools.vector_store import load_job_description
from langchain_openai import ChatOpenAI
import pandas as pd
import ast
from dotenv import load_dotenv

load_dotenv()
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
    
    jd = load_job_description()['text']
    llm = ChatOpenAI(model_name="gpt-4o")
    print(jd)

    results = []

    for resume in resumes:
        candidate = resume.get("name", "Unknown")
        content = resume.get("description", "")

        prompt = f"""
                    You are a technical recruiter. 
                    job Description:
                    {jd}
                    Resume:
                    {content}

                    Task:
                    Score this resume out of 100 based on its match to the job description.
                    Only return the score as a number (e.g., 83).
                    """
        try:
            raw = llm.invoke(prompt)
            score = float(raw.content.strip())
        except Exception as e:
            score = 0.0

        results.append((candidate, score))
    df = pd.DataFrame(results, columns=["Name", "Match Score"])
    df = df.sort_values(by="Match Score", ascending=False)

    return df.to_markdown(index=False)