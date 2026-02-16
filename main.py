from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import tempfile
import sys

from rag import ingest_document, query_document

print("===== MAIN.PY STARTUP DEBUG =====", file=sys.stderr)
print("Python version:", sys.version, file=sys.stderr)
print("Current working dir:", os.getcwd(), file=sys.stderr)
print("DATABASE_URL exists?", "DATABASE_URL" in os.environ, file=sys.stderr)
print("GEMINI_API_KEY exists?", "GEMINI_API_KEY" in os.environ, file=sys.stderr)
print("Trying to import rag...", file=sys.stderr)
try:
    print("rag imported successfully", file=sys.stderr)
except Exception as e:
    print("Failed to import rag:", str(e), file=sys.stderr)
    raise
print("===== END STARTUP DEBUG =====", file=sys.stderr)

app = FastAPI(title="Simple PDF RAG")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files allowed"}

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        ingest_document(tmp_path)
        return {"message": "PDF processed successfully! You can now ask questions."}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)  # clean up

@app.post("/query")
async def ask(question: str = Form(...)):
    try:
        answer = query_document(question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}