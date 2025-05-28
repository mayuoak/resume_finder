# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import json
import pandas as pd
from typing import List
from gemini_save import agent_executor, PDFReader
import io


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    return """
    <html>
    <head>
        <title>Resume Matcher</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h2 {
                color: #333;
            }
            input[type="file"] {
                margin: 10px 0;
                display: block;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            }
            button:hover {
                background-color: #45a049;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                background: white;
            }
            th, td {
                border: 1px solid #ccc;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <h2>Upload Job Description</h2>
        <input type="file" id="jdFile" />

        <h2>Upload Resumes (Multiple PDFs)</h2>
        <input type="file" id="resumeFiles" multiple />

        <button onclick="executeScoring()">Run Matching</button>
        <button onclick="downloadCSV()">Download CSV</button>

        <div id="result"></div>

        <script>
            let tableData = [];

            async function executeScoring() {
                const jdFile = document.getElementById('jdFile').files[0];
                const resumeFiles = document.getElementById('resumeFiles').files;
                if (!jdFile || resumeFiles.length === 0) {
                    alert("Please upload both job description and resumes.");
                    return;
                }

                try {
                    const jdForm = new FormData();
                    jdForm.append("file", jdFile);
                    const jdRes = await fetch("/upload-jd", { method: "POST", body: jdForm });
                    if (!jdRes.ok) throw new Error("Failed to upload JD");

                    const resumeForm = new FormData();
                    for (let file of resumeFiles) {
                        resumeForm.append("files", file);
                    }
                    const resumeRes = await fetch("/upload-resumes", { method: "POST", body: resumeForm });
                    if (!resumeRes.ok) throw new Error("Failed to upload resumes");

                    const resumeJson = await resumeRes.json();
                    sessionId = resumeJson.session_id;

                    const res = await fetch(`/score-results?format=json&session_id=${sessionId}`);
                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || "Failed to get scores");
                    }

                    const data = await res.json();
                    tableData = data.result;
                    renderTable(tableData);
                } catch (err) {
                    alert("Error: " + err.message);
                    console.error(err);
                }
            }

            function renderTable(data) {
                if (!data || data.length === 0) return;
                const table = `<table><thead><tr><th>Name</th><th>Score</th><th>Download</th></tr></thead><tbody>
                    ${data.map(row => `
                        <tr>
                            <td>${row.Name}</td>
                            <td>${row.Score}</td>
                            <td><a href="/download-resume/${row.Filename}" target="_blank">Download</a></td>
                        </tr>`).join('')}
                </tbody></table>`;
                document.getElementById("result").innerHTML = table;
            }
        </script>

    </body>
    </html>
    """

@app.post("/upload-jd")
def upload_job_description(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, "jd.txt")
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": "Job description uploaded."}

@app.post("/upload-resumes")
def upload_resumes(files: List[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())  # unique folder
    resumes_dir = os.path.join(UPLOAD_DIR, "resumes", session_id)
    os.makedirs(resumes_dir, exist_ok=True)

    saved_filenames = []

    for file in files:
        filename = file.filename.replace(" ", "_")
        full_path = os.path.join(resumes_dir, filename)
        with open(full_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_filenames.append(filename)

    return {"message": f"Uploaded {len(files)} resumes.", "session_id": session_id}



def parse_markdown_with_pandas(md_text):
    # Remove the markdown alignment row (e.g., |:-----|)
    lines = md_text.strip().splitlines()
    clean_lines = [line for line in lines if not set(line.strip()).issubset({':', '-', '|'})]

    # Join back the cleaned lines into a single string
    clean_markdown = "\n".join(clean_lines)

    # Read using pandas
    df = pd.read_table(io.StringIO(clean_markdown), sep='|', engine='python')

    # Strip column names and values
    df.columns = [col.strip() for col in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop unnamed columns created by extra pipes
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    return df

@app.get("/download-resume/{filename}")
def download_resume(filename: str):
    file_path = os.path.join(UPLOAD_DIR, "resumes", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type='application/pdf')


@app.get("/score-results")
def score_results(format: str = "markdown", session_id: str = ""):
    jd_path = os.path.join(UPLOAD_DIR, "jd.txt")
    resume_dir = os.path.join(UPLOAD_DIR, "resumes", session_id)

    if not os.path.exists(resume_dir):
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    try:
        result = agent_executor.run(f"""
        Please perform the following steps:
        1. Load and save the job description from the file: {jd_path}
        2. Extract and score all resumes found in the folder: {resume_dir}
        3. Use the score_resumes_tool and return the result as a markdown table. 
        """)

        if format == "json":
            df = parse_markdown_with_pandas(result)
            df["Filename"] = df["Name"].apply(lambda name: name.replace(" ", "_") + ".pdf")
            return JSONResponse(content={"result": df.to_dict(orient="records")})

        return JSONResponse(content={"result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)