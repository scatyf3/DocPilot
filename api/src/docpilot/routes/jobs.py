"""Progress of background ingest jobs (INGEST_MODE=queue), for the client to poll."""

import uuid

from fastapi import APIRouter, HTTPException, Query, Request

from docpilot import jobs
from docpilot.security import CurrentPrincipal

router = APIRouter()


def _view(job: dict) -> dict:
    total = job["chunks_total"]
    return {**job, "progress": round(job["chunks_done"] / total, 3) if total else None}


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: uuid.UUID, principal: CurrentPrincipal) -> dict:
    job = await jobs.get_job(request.app.state.pool, principal.tenant_id, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return _view(job)


@router.get("/jobs")
async def list_jobs(request: Request, principal: CurrentPrincipal,
                    limit: int = Query(20, ge=1, le=100)) -> list[dict]:
    return [_view(j) for j in await jobs.list_jobs(request.app.state.pool, principal.tenant_id, limit)]
