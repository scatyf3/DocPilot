"use client";

import type { Chunk } from "@/lib/types";

const CITATION = /\[c:(\d+)\]/g;

type Props = {
  text: string;
  citations: Chunk[];
  onCite: (chunk: Chunk) => void;
};

/** Renders answer text, replacing `[c:<id>]` markers with numbered, clickable chips. */
export function AnswerText({ text, citations, onCite }: Props) {
  const byId = new Map(citations.map((c, i) => [c.id, { chunk: c, n: i + 1 }]));
  const nodes: React.ReactNode[] = [];
  let last = 0;

  for (const match of text.matchAll(CITATION)) {
    nodes.push(text.slice(last, match.index));
    const id = Number(match[1]);
    const hit = byId.get(id);
    nodes.push(
      hit ? (
        <button
          key={`${match.index}-${id}`}
          onClick={() => onCite(hit.chunk)}
          title={hit.chunk.title}
          className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-blue-100 px-1 align-baseline text-xs font-semibold text-blue-800 hover:bg-blue-200"
        >
          {hit.n}
        </button>
      ) : (
        // Not resolved (yet): citations arrive at the end of the stream; unknown ids stay grey.
        <span key={`${match.index}-${id}`} className="mx-0.5 rounded bg-gray-100 px-1 text-xs text-gray-500">
          c:{id}
        </span>
      ),
    );
    last = match.index + match[0].length;
  }
  nodes.push(text.slice(last));

  return <div className="whitespace-pre-wrap leading-relaxed">{nodes}</div>;
}
