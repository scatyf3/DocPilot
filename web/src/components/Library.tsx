"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { deleteDocument, getDocument, listDocuments, uploadDocument } from "@/lib/api";
import type { DocumentDetail, DocumentSummary } from "@/lib/types";

/** Upload, list and read documents, including pages the agent created with create_page. */
export function Library() {
  const router = useRouter();
  const selectedId = Number(useSearchParams().get("doc")) || null;
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [selected, setSelected] = useState<DocumentDetail | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const refresh = useCallback(() => {
    listDocuments().then(setDocs, (e: Error) => setStatus(`Could not load documents: ${e.message}`));
  }, []);

  useEffect(refresh, [refresh]);

  useEffect(() => {
    if (selectedId === null) return;
    getDocument(selectedId).then(setSelected, (e: Error) => setStatus(e.message));
  }, [selectedId]);

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    for (const file of files) {
      setStatus(`Uploading ${file.name}…`);
      try {
        const res = await uploadDocument(file);
        setStatus(`Indexed ${file.name} (${res.chunks} chunks)`);
      } catch (err) {
        setStatus(`Failed to upload ${file.name}: ${(err as Error).message}`);
      }
    }
    refresh();
  }

  async function onDelete(id: number) {
    if (!confirm("Delete this document and its chunks?")) return;
    await deleteDocument(id);
    if (selected?.id === id) {
      setSelected(null);
      router.push("/library");
    }
    refresh();
  }

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-1 gap-6 px-4 py-6">
      <section className="w-80 shrink-0 space-y-3">
        <label className="block cursor-pointer rounded border-2 border-dashed p-4 text-center text-sm text-gray-600 hover:bg-gray-50">
          Upload PDF, Markdown or HTML
          <input type="file" multiple accept=".pdf,.md,.markdown,.html,.htm,.txt" onChange={onUpload} className="hidden" />
        </label>
        {status && <p className="text-xs text-gray-500">{status}</p>}
        <ul className="divide-y rounded border">
          {docs.length === 0 && <li className="p-3 text-sm text-gray-500">No documents yet.</li>}
          {docs.map((d) => (
            <li key={d.id} className={`flex items-center gap-2 p-2 text-sm ${d.id === selectedId ? "bg-blue-50" : ""}`}>
              <button onClick={() => router.push(`/library?doc=${d.id}`)} className="flex-1 truncate text-left" title={d.source}>
                {d.title}
              </button>
              {d.kind === "generated" && (
                <span className="rounded bg-purple-100 px-1.5 text-xs text-purple-800">agent</span>
              )}
              <span className="text-xs text-gray-400">{d.chunk_count}</span>
              <button onClick={() => onDelete(d.id)} className="text-gray-400 hover:text-red-600" aria-label="Delete">
                ✕
              </button>
            </li>
          ))}
        </ul>
      </section>

      <article className="min-w-0 flex-1">
        {!selected && <p className="pt-12 text-center text-gray-500">Select a document to read it.</p>}
        {selected && (
          <>
            <h2 className="text-xl font-semibold">{selected.title}</h2>
            <p className="mb-4 text-xs text-gray-500">
              {selected.kind === "generated" ? "Created by the agent" : selected.source} ·{" "}
              {new Date(selected.created_at).toLocaleString()} · {selected.chunks.length} chunks
            </p>
            {selected.chunks.map((c) => (
              <div key={c.id} className="border-l-2 border-gray-200 py-2 pl-3 hover:border-blue-400">
                <p className="mb-1 text-xs text-gray-400">
                  chunk {c.ord} · <code>c:{c.id}</code>
                </p>
                <p className="whitespace-pre-wrap text-sm leading-relaxed">{c.text}</p>
              </div>
            ))}
          </>
        )}
      </article>
    </div>
  );
}
