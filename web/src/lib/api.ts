// Thin typed client for the FastAPI backend. NEXT_PUBLIC_API_URL is inlined at build time.
import type { DocumentDetail, DocumentSummary, TraceDetail } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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
  return request<{ id: number; chunks: number }>("/documents", { method: "POST", body: form });
}

export const getTrace = (messageId: string) => request<TraceDetail>(`/traces/${messageId}`);
