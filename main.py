import os
import sys
import tempfile

from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag import ingest_document, query_document

# ── Very early debug prints (appear even if app crashes during import) ──
print("===== MAIN.PY VERY EARLY STARTUP DEBUG =====", file=sys.stderr)
print("Python executable:", sys.executable, file=sys.stderr)
print("Python version:", sys.version, file=sys.stderr)
print("Current working directory:", os.getcwd(), file=sys.stderr)
print("Environment keys:", list(os.environ.keys()), file=sys.stderr)
print("DATABASE_URL present?", "DATABASE_URL" in os.environ, file=sys.stderr)
if "DATABASE_URL" in os.environ:
    url = os.environ["DATABASE_URL"]
    masked = url[:10] + "..." + url[-10:] if len(url) > 20 else url
    print("DATABASE_URL (masked):", masked, file=sys.stderr)
else:
    print("DATABASE_URL: MISSING", file=sys.stderr)
print("GEMINI_API_KEY present?", "GEMINI_API_KEY" in os.environ, file=sys.stderr)
print("GROQ_API_KEY present?", "GROQ_API_KEY" in os.environ, file=sys.stderr)
print("===== END EARLY DEBUG =====", file=sys.stderr)

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
        return {"error": f"Ingestion failed: {str(e)}"}
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass  # ignore cleanup error

@app.post("/query")
async def ask(question: str = Form(...)):
    try:
        answer = query_document(question)
        return {"answer": answer}
    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}