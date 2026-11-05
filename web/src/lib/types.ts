import type { UIMessage } from "ai";

// Shapes returned by the FastAPI backend (see api/src/docpilot/store.py).

export type Chunk = {
  id: number;
  document_id: number;
  ord: number;
  title: string;
  text: string;
};

export type ToolCallLog = {
  name: string;
  args: Record<string, unknown> | null;
  summary: string;
  is_error: boolean;
  duration_ms: number;
};

export type TraceSummary = {
  message_id: string;
  model: string;
  retrieved_chunk_ids: number[];
  tool_calls: ToolCallLog[];
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  cost_usd: number;
  rejected_citation_ids?: number[];
  ungrounded?: boolean;
};

export type TraceDetail = TraceSummary & { retrieved_chunks: Chunk[] };

export type DocumentSummary = {
  id: number;
  title: string;
  source: string;
  kind: "upload" | "generated";
  created_at: string;
  chunk_count: number;
};

export type DocumentDetail = Omit<DocumentSummary, "chunk_count"> & {
  chunks: { id: number; ord: number; text: string }[];
};

// Custom `data-*` parts the backend streams alongside the answer text.
export type DocPilotDataParts = {
  citations: Chunk[];
  trace: TraceSummary;
};

export type ChatMessage = UIMessage<never, DocPilotDataParts>;

export type RetrievalMode = "hybrid" | "vector" | "fts";

// A background ingest job (API running with INGEST_MODE=queue). See api/src/docpilot/jobs.py.
export type IngestJob = {
  id: string;
  filename: string;
  status: "queued" | "running" | "succeeded" | "failed";
  stage: "parsing" | "embedding" | "storing" | null;
  chunks_done: number;
  chunks_total: number | null;
  progress: number | null;
  attempts: number;
  error: string | null;
  document_id: number | null;
};

// POST /documents: indexed at once (inline mode), or queued (queue mode).
export type UploadResult = { id: number; chunks: number } | { job_id: string; status: "queued" };
