import { Suspense } from "react";
import { Chat } from "@/components/Chat";

export default function ChatPage() {
  // useChat generates a random chat id, so the chat renders per request, not at build time.
  return (
    <Suspense fallback={null}>
      <Chat />
    </Suspense>
  );
}
