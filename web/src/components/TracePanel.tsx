"use client";

import { useState } from "react";
import { getTrace } from "@/lib/api";
import type { TraceDetail, TraceSummary } from "@/lib/types";

/** Per-message trace: tokens, latency, cost, tool calls, and (on expand) the retrieved chunks. */
export function TracePanel({ trace }: { trace: TraceSummary }) {
  const [open, setOpen] = useState(false);
  const [detail, setDetail] = useState<TraceDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function toggle() {
    setOpen(!open);
    if (!detail && !open) {
      getTrace(trace.message_id).then(setDetail, (e: Error) => setError(e.message));
    }
  }

  return (
    <div className="mt-2 rounded border border-gray-200 bg-gray-50 text-xs text-gray-600">
      <button onClick={toggle} className="flex w-full flex-wrap gap-x-4 px-3 py-1.5 text-left">
        <span className="font-medium">{open ? "▾" : "▸"} Trace</span>
        <span>{trace.tool_calls.length} tool calls</span>
        <span>{trace.retrieved_chunk_ids.length} chunks</span>
        <span>
          {trace.input_tokens.toLocaleString()} in / {trace.output_tokens.toLocaleString()} out tokens
        </span>
        <span>{(trace.latency_ms / 1000).toFixed(2)} s</span>
        <span>${trace.cost_usd.toFixed(4)}</span>
        <span className="text-gray-400">{trace.model}</span>
      </button>
      {open && (
        <div className="space-y-3 border-t border-gray-200 px-3 py-2">
          <section>
            <h4 className="font-semibold">Tool calls</h4>
            <ol className="list-decimal pl-5">
              {trace.tool_calls.map((t, i) => (
                <li key={i} className={t.is_error ? "text-red-600" : ""}>
                  <code>{t.name}</code>({JSON.stringify(t.args)}) → {t.summary}{" "}
                  <span className="text-gray-400">{t.duration_ms} ms</span>
                </li>
              ))}
            </ol>
          </section>
          <section>
            <h4 className="font-semibold">Retrieved chunks</h4>
            {error && <p className="text-red-600">{error}</p>}
            {!detail && !error && <p>Loading…</p>}
            {detail?.retrieved_chunks.map((c) => (
              <details key={c.id} className="py-0.5">
                <summary className="cursor-pointer">
                  <code>c:{c.id}</code> {c.title} (chunk {c.ord})
                </summary>
                <p className="whitespace-pre-wrap py-1 pl-4 text-gray-700">{c.text}</p>
              </details>
            ))}
          </section>
        </div>
      )}
    </div>
  );
}
