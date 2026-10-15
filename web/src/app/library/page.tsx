import { Suspense } from "react";
import { Library } from "@/components/Library";

export default function LibraryPage() {
  // Library reads ?doc= from the URL, which is only known at request time.
  return (
    <Suspense fallback={<p className="p-6 text-gray-500">Loading…</p>}>
      <Library />
    </Suspense>
  );
}
