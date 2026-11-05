"""Document library: upload, list, view, delete. Plus /search for debugging retrieval.

Everything here acts on the caller's tenant. Another tenant's document gets the same 404 as
one that doesn't exist, so ids can't be probed.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from docpilot import jobs, store
from docpilot.config import RerankerName, RetrievalMode
from docpilot.ingest.pipeline import ingest_file
from docpilot.security import CurrentPrincipal, rate_limit

log = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


async def read_upload(file: UploadFile) -> bytes:
    # Multipart bodies are already spooled to disk here; don't pull an oversized one into memory.
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large (max 20 MB)")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:  # not every client declares a size
        raise HTTPException(413, "file too large (max 20 MB)")
    return data


@router.post("/documents", status_code=201, dependencies=[Depends(rate_limit("upload"))],
             responses={202: {"description": "Queued (INGEST_MODE=queue): poll the job"}})
async def upload_document(request: Request, file: UploadFile, principal: CurrentPrincipal):
    """Inline mode: index the file now, 201 {id, chunks}. Queue mode: 202 {job_id}."""
    s = request.app.state
    data = await read_upload(file)
    filename = file.filename or "upload.md"
    if s.settings.ingest_mode == "queue":
        job_id = await jobs.create_job(s.pool, tenant_id=principal.tenant_id,
                                       created_by=principal.user_id, filename=filename, data=data)
        try:
            await jobs.enqueue(s.arq, job_id, principal.tenant_id)
        except (RedisError, OSError) as e:
            # The job is safe in Postgres; the worker's sweeper enqueues it once Redis is back.
            log.warning("could not enqueue ingest job %s: %s", job_id, type(e).__name__)
        return JSONResponse({"job_id": str(job_id), "status": "queued"}, status_code=202,
                            headers={"Location": f"/jobs/{job_id}"})
    try:
        doc_id, n_chunks = await ingest_file(
            s.pool, s.embedder, s.settings, tenant_id=principal.tenant_id,
            filename=filename, data=data, created_by=principal.user_id,
            cache=s.cache,
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"id": doc_id, "chunks": n_chunks}


@router.get("/documents")
async def list_documents(request: Request, principal: CurrentPrincipal) -> list[dict]:
    return await store.list_documents(request.app.state.pool, principal.tenant_id)


@router.get("/documents/{doc_id}")
async def get_document(request: Request, doc_id: int,
                       principal: CurrentPrincipal) -> dict:
    doc = await store.get_document(request.app.state.pool, principal.tenant_id, doc_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    return doc


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(request: Request, doc_id: int,
                          principal: CurrentPrincipal) -> None:
    s = request.app.state
    if not await store.delete_document(s.pool, principal.tenant_id, doc_id):
        raise HTTPException(404, "document not found")
    if s.cache is not None:
        await s.cache.invalidate(principal.tenant_id)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(5, ge=1, le=50)
    mode: RetrievalMode | None = None
    reranker: RerankerName | None = None
    document_ids: list[int] | None = Field(None, min_length=1, max_length=100)


@router.post("/search", dependencies=[Depends(rate_limit("search"))])
async def search(request: Request, body: SearchRequest,
                 principal: CurrentPrincipal) -> list[dict]:
    """Retrieval only, no LLM. The eval suite uses this to score recall@k and MRR."""
    chunks = await request.app.state.retriever.search(
        body.query, tenant_id=principal.tenant_id, top_k=body.top_k, mode=body.mode,
        reranker=body.reranker, document_ids=body.document_ids,
    )
    return [c.to_dict() for c in chunks]
