import type { Metadata } from "next";
import { Plus_Jakarta_Sans, IBM_Plex_Mono, Source_Serif_4 } from "next/font/google";
import Nav from "@/components/Nav";
import "./globals.css";

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-jakarta",
  display: "swap",
});

const plex = IBM_Plex_Mono({
  weight: ["400", "500", "600"],
  subsets: ["latin"],
  variable: "--font-plex",
  display: "swap",
});

const sourceSerif = Source_Serif_4({
  subsets: ["latin"],
  variable: "--font-source-serif",
  display: "swap",
});

export const metadata: Metadata = {
  title: "PyroSight — Wildfire Risk Intelligence",
  description: "Next-day fire spread prediction with evidential deep learning and Rothermel physics.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${jakarta.variable} ${plex.variable} ${sourceSerif.variable}`}>
      <body className="font-sans antialiased bg-bg min-h-screen">
        <Nav />
        {children}
      </body>
    </html>
  );
}
