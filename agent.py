from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from tools.job_description_tool import store_job_description_tool, fetch_job_description
from tools.resume_loader_tool import load_resume_tool
from tools.vector_store import load_job_description
from tools.resume_scoring_tool import score_resumes_tool
import json

def run_agent(mode: str):
    load_dotenv()
    llm = ChatOpenAI(model_name='gpt-4o')

    tools = [store_job_description_tool, 
             load_resume_tool,
             score_resumes_tool,
             ]

    agent_executor = initialize_agent(
        llm = llm,
        tools=tools,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    if mode == "initialize":
        job_desc_filepath = input("Enter path of job description_file: ").strip()
        agent_executor.run(f"store job description from this file: {job_desc_filepath}")
    elif mode == "find":
        resumes = load_resume_tool.invoke('data/resumes')
        resumes_str = json.dumps(resumes, indent=2)
        prompt = f"""
        Use score_resumes_tool to evaluate the following resumes against the stored job description.

        Return a markdown table with Name and Match Score.

        Here is the resume data:
        '''{resumes_str}'''
        """

        # ✅ Now pass the prompt properly
        response = agent_executor.invoke({"input": prompt})
        print(response)
    else:
        print(f"Valid commands are 'initialize' and 'find'")