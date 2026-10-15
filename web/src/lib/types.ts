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
