"use client";

import Link from "next/link";
import type { Chunk } from "@/lib/types";

/** Side panel showing the exact chunk a citation points to. */
export function SourcePanel({ chunk, onClose }: { chunk: Chunk; onClose: () => void }) {
  return (
    <aside className="w-96 shrink-0 overflow-y-auto border-l bg-gray-50 p-4">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div>
          <h3 className="font-semibold">{chunk.title}</h3>
          <p className="text-xs text-gray-500">
            chunk {chunk.ord} · <code>c:{chunk.id}</code> ·{" "}
            <Link href={`/library?doc=${chunk.document_id}`} className="text-blue-700 underline">
              open document
            </Link>
          </p>
        </div>
        <button onClick={onClose} className="text-gray-500 hover:text-black" aria-label="Close">
          ✕
        </button>
      </div>
      <p className="whitespace-pre-wrap text-sm leading-relaxed">{chunk.text}</p>
    </aside>
  );
}
