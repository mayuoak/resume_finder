# Resume Finder

A smart AI-powerd resume matching tool using both OpenAI and Google Gemini LLMs.

## Overview

**Resume Finder** automatically ranks candidate resumes based on how well they match a job description. It uses two techniques:
- **Cosine similarity** between embeddings
- **LLM-based evaluation** using OpenAI or Gemini

It's designed for recruiters, hiring tools, or anyone automating resume screening.

## 🧪 Sample Output

| Name                                        | Score                  |
|:--------------------------------------------|:-----------------------|
| BobSmith_resume_MachineLearningEngineer     | 70.0 LLM, 52.02 cosine |
| AliceJohnson_resume_MachineLearningEngineer | 65.0 LLM, 50.0 cosine  |
| GraceHill_resume_DataEngineer               | 40.0 LLM, 37.66 cosine |
| HenryAdams_resume_DataEngineer              | 40.0 LLM, 36.23 cosine |
| CarolWhite_resume_MachineLearningEngineer   | 20.0 LLM, 51.97 cosine |
| IvyScott_resume_DataEngineer                | 20.0 LLM, 37.67 cosine |
| JackWilson_resume_DevOpsEngineer            | 20.0 LLM, 32.19 cosine |
| KarenLee_resume_DevOpsEngineer              | 20.0 LLM, 29.72 cosine |
| LeoMartin_resume_DevOpsEngineer             | 10.0 LLM, 29.95 cosine |
| EllaBrown_resume_FrontendEngineer           | 10.0 LLM, 28.51 cosine |
| FrankTurner_resume_FrontendEngineer         | 10.0 LLM, 25.11 cosine |
| DavidGreen_resume_FrontendEngineer          | 10.0 LLM, 22.38 cosine |


## Features

- Parse job descriptions from `.txt` files
- Extract text from PDF resumes
- Score using:
    - OpenAI GPT models (via `main.py`)
    - Google Gemini LLM (via `gemini_save.py`)
- Cosine similarity score with FAISS + SentenceTransformers
- Modular LangChain agent integration
- Stores embedding index for faster reruns


## 🔀 Scoring Pipelines

| Script           | LLM Provider | Description                          |
|------------------|--------------|--------------------------------------|
| `main.py`        | OpenAI       | Uses GPT-3.5/4 via LangChain         |
| `gemini_save.py` | Gemini       | Uses Gemini 1.5 Pro for scoring      |

## Project Structure

resume_finder/  
./agent.py  
./data  
├── job_description.txt  
└── resumes  
    ├── *pdf  
./gemini_save.py  
./gemini.index.faiss  
./gemini.meta.pkl  
./main.py  
./README.md  
./requirements.txt  
./tools  
├── job_description_tool.py  
├── resume_loader_tool.py  
├── resume_scoring_tool.py  
└── vector_store.py  
./utils.py  
./vector_store  
├── gemini.index.faiss  
├── gemini.meta.pkl  
├── index.faiss  
└── meta.pkl  

## Setup Instructins

1. **Clone the repo**

```bash
git clone https://github.com/mayuoak/resume_finder.git
cd resume_finder
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set your .env file**
```bash
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
```

5. **OpenAI**
```bash
python main.py initialize # Read job description
python main.py find # find score for each resume
```

6. **Gemini**
```bash
python gemini_save.py
```

7. Inputs and Outputs
## 📥 Input Format

- `data/job_description.txt` → plain text
- `data/resumes/*.pdf` → resumes to rank

## 📤 Output Format

- Markdown table with:
  - LLM Score (0–100)
  - Cosine similarity score

## 🧰 Tech Stack

- Python 3.12+
- FAISS
- SentenceTransformers
- LangChain
- Google Generative AI
- OpenAI GPT
- Pydantic
- PyMuPDF (for PDF parsing)



---

#### 👤 Author and 📜 License

End the file with credits and licensing terms:

## Author

**Mayuresh Oak**  
[GitHub](https://github.com/mayuoak)

## 📜 License

Currently private use only. Please contact the author before reusing this code commercially or publicly.
