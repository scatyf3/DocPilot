import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "DocPilot",
  description: "Chat with your documents. Answers cite their sources.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex h-full flex-col bg-white text-gray-900">
        <header className="flex items-center gap-6 border-b px-4 py-3">
          <Link href="/" className="font-semibold">
            DocPilot
          </Link>
          <nav className="flex gap-4 text-sm text-gray-600">
            <Link href="/" className="hover:text-black">
              Chat
            </Link>
            <Link href="/library" className="hover:text-black">
              Library
            </Link>
          </nav>
        </header>
        <main className="flex min-h-0 flex-1 flex-col">{children}</main>
      </body>
    </html>
  );
}
