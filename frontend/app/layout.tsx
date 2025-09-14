import React from "react";
import "../styles/globals.css";
import { ThemeProvider } from "../components/theme-provider";
import Sidebar from "../components/sidebar";
import Header from "../components/header";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-background font-sans antialiased" suppressHydrationWarning>
        <ThemeProvider>
          <div className="flex min-h-screen">
            <Sidebar />
            <div className="flex-1 flex flex-col">
              <Header />
              <main className="flex-1 p-6 bg-muted/40">{children}</main>
            </div>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
