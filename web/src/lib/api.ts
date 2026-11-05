// Thin typed client for the FastAPI backend, reached through the server-side proxy route.
import type { DocumentDetail, DocumentSummary, IngestJob, TraceDetail, UploadResult } from "./types";

export const API_URL = "/api/backend";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const listDocuments = () => request<DocumentSummary[]>("/documents");

export const getDocument = (id: number) => request<DocumentDetail>(`/documents/${id}`);

export const deleteDocument = (id: number) =>
  request<void>(`/documents/${id}`, { method: "DELETE" });

export function uploadDocument(file: File) {
  const form = new FormData();
  form.append("file", file);
  return request<UploadResult>("/documents", { method: "POST", body: form });
}

export const getJob = (id: string) => request<IngestJob>(`/jobs/${id}`);

/** Poll a queued upload until it finishes, reporting each state on the way. */
export async function waitForJob(id: string, onUpdate: (job: IngestJob) => void, intervalMs = 1000) {
  for (;;) {
    const job = await getJob(id);
    onUpdate(job);
    if (job.status === "succeeded" || job.status === "failed") return job;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

export const getTrace = (messageId: string) => request<TraceDetail>(`/traces/${messageId}`);
