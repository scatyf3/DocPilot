"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useState } from "react";
import { API_URL } from "@/lib/api";
import type { ChatMessage, Chunk, RetrievalMode, TraceSummary } from "@/lib/types";
import { AnswerText } from "./AnswerText";
import { SourcePanel } from "./SourcePanel";
import { TracePanel } from "./TracePanel";

// The FastAPI /chat endpoint speaks the AI SDK UI message stream, so the stock transport works.
const transport = new DefaultChatTransport<ChatMessage>({ api: `${API_URL}/chat` });

const TOOL_LABELS: Record<string, string> = {
  search_docs: "Searching documents",
  read_doc: "Reading document",
  create_page: "Creating page",
};

/** Remounting the window with a new key starts a fresh conversation (useChat makes a new id). */
export function Chat() {
  const [session, setSession] = useState(0);
  return <ChatWindow key={session} onNewChat={() => setSession((s) => s + 1)} />;
}

function ChatWindow({ onNewChat }: { onNewChat: () => void }) {
  const { messages, sendMessage, status, error, stop } = useChat<ChatMessage>({ transport });
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [source, setSource] = useState<Chunk | null>(null);
  const busy = status === "submitted" || status === "streaming";

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || busy) return;
    // Extra body fields are merged into the POST; the backend reads `retrieval`.
    sendMessage({ text: input }, { body: { retrieval: { mode } } });
    setInput("");
  }

  return (
    <div className="flex min-h-0 flex-1">
      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-4">
        <div className="flex-1 space-y-6 overflow-y-auto py-6">
          {messages.length === 0 && (
            <p className="pt-24 text-center text-gray-500">
              Ask a question about your documents. Answers cite the chunks they came from.
            </p>
          )}
          {messages.map((m) => (
            <Message key={m.id} message={m} onCite={setSource} />
          ))}
          {status === "submitted" && <p className="text-sm text-gray-400">Thinking…</p>}
          {error && <p className="text-sm text-red-600">Error: {error.message}</p>}
        </div>

        <form onSubmit={submit} className="sticky bottom-0 flex gap-2 border-t bg-white py-3">
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as RetrievalMode)}
            className="rounded border px-2 text-sm"
            title="Retrieval mode"
          >
            <option value="hybrid">hybrid</option>
            <option value="vector">vector</option>
            <option value="fts">full-text</option>
          </select>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your documents…"
            className="flex-1 rounded border px-3 py-2"
          />
          {busy ? (
            <button type="button" onClick={stop} className="rounded bg-gray-200 px-4">
              Stop
            </button>
          ) : (
            <button type="submit" className="rounded bg-blue-600 px-4 text-white disabled:opacity-50" disabled={!input.trim()}>
              Send
            </button>
          )}
          <button type="button" onClick={onNewChat} className="rounded border px-3 text-sm" title="Start a new conversation">
            New
          </button>
        </form>
      </div>
      {source && <SourcePanel chunk={source} onClose={() => setSource(null)} />}
    </div>
  );
}

function Message({ message, onCite }: { message: ChatMessage; onCite: (c: Chunk) => void }) {
  if (message.role === "user") {
    const text = message.parts.map((p) => (p.type === "text" ? p.text : "")).join("");
    return <div className="ml-auto w-fit max-w-[80%] rounded-lg bg-blue-600 px-4 py-2 text-white">{text}</div>;
  }

  let citations: Chunk[] = [];
  let trace: TraceSummary | null = null;
  for (const part of message.parts) {
    if (part.type === "data-citations") citations = part.data;
    if (part.type === "data-trace") trace = part.data;
  }

  return (
    <div className="space-y-2">
      {message.parts.map((part, i) => {
        if (part.type === "text") {
          return <AnswerText key={i} text={part.text} citations={citations} onCite={onCite} />;
        }
        if (part.type === "dynamic-tool") {
          const done = part.state === "output-available";
          const failed = part.state === "output-error";
          return (
            <div key={i} className="w-fit rounded-full border px-3 py-0.5 text-xs text-gray-600">
              {failed ? "✗" : done ? "✓" : "…"} {TOOL_LABELS[part.toolName] ?? part.toolName}
              {typeof part.input === "object" && part.input && "query" in part.input
                ? `: "${String(part.input.query)}"`
                : ""}
            </div>
          );
        }
        return null;
      })}
      {trace?.ungrounded && (
        <p className="w-fit rounded bg-amber-50 px-2 py-1 text-xs text-amber-800">
          No valid source: this answer may not come from your documents.
        </p>
      )}
      {trace && <TracePanel trace={trace} />}
    </div>
  );
}
