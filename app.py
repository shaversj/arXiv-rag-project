from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from query_engine import QueryEngine

app = FastAPI()
qe = QueryEngine()

class SearchRequest(BaseModel):
    query: str
    filters: dict = {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/search")
def search(req: SearchRequest):
    try:
        qe.initialize()
        results = qe.search(req.query, req.filters)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/paper/{paper_id}")
def get_paper(paper_id: str):
    qe.initialize()
    paper = qe.store.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.on_event("shutdown")
def shutdown_event():
    qe.close()