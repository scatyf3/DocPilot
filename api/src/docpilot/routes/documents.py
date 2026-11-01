"""Document library: upload, list, view, delete. Plus /search for debugging retrieval."""

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from docpilot import store
from docpilot.config import RerankerName, RetrievalMode
from docpilot.ingest.pipeline import ingest_file
from docpilot.security import rate_limit

router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@router.post("/documents", status_code=201, dependencies=[Depends(rate_limit("upload"))])
async def upload_document(request: Request, file: UploadFile) -> dict:
    s = request.app.state
    # Multipart bodies are already spooled to disk here; don't pull an oversized one into memory.
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large (max 20 MB)")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:  # not every client declares a size
        raise HTTPException(413, "file too large (max 20 MB)")
    try:
        doc_id, n_chunks = await ingest_file(
            s.pool, s.embedder, s.settings, filename=file.filename or "upload.md", data=data
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"id": doc_id, "chunks": n_chunks}


@router.get("/documents")
async def list_documents(request: Request) -> list[dict]:
    return await store.list_documents(request.app.state.pool)


@router.get("/documents/{doc_id}")
async def get_document(request: Request, doc_id: int) -> dict:
    doc = await store.get_document(request.app.state.pool, doc_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    return doc


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(request: Request, doc_id: int) -> None:
    if not await store.delete_document(request.app.state.pool, doc_id):
        raise HTTPException(404, "document not found")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(5, ge=1, le=50)
    mode: RetrievalMode | None = None
    reranker: RerankerName | None = None


@router.post("/search", dependencies=[Depends(rate_limit("search"))])
async def search(request: Request, body: SearchRequest) -> list[dict]:
    """Retrieval only, no LLM. The eval suite uses this to score recall@k and MRR."""
    chunks = await request.app.state.retriever.search(
        body.query, top_k=body.top_k, mode=body.mode, reranker=body.reranker
    )
    return [c.to_dict() for c in chunks]
