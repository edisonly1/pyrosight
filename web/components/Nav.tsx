"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import SearchBar from "./SearchBar";

const LINKS = [
  { href: "/", label: "Dashboard" },
  { href: "/map", label: "Map" },
];

export default function Nav() {
  const pathname = usePathname();

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  };

  return (
    <nav className="flex items-center justify-between px-6 lg:px-10 h-16 bg-bg-white border-b border-border sticky top-0 z-50 gap-4">
      <Link href="/" className="flex items-center gap-2.5 no-underline flex-shrink-0">
        <div className="w-[30px] h-[30px] rounded-[7px] bg-gradient-to-br from-orange-400 to-fire flex items-center justify-center shadow-[0_2px_6px_rgba(234,88,12,0.2)]">
          <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" className="w-4 h-4">
            <path d="M12 2c0 6-8 10-8 16a8 8 0 0016 0c0-6-8-10-8-16z" />
          </svg>
        </div>
        <span className="font-bold text-[17px] tracking-tight text-text hidden sm:inline">PyroSight</span>
      </Link>

      <div className="flex gap-1 flex-shrink-0">
        {LINKS.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            className={`px-3 lg:px-4 py-2 rounded-lg text-sm font-medium transition-colors no-underline ${
              isActive(href)
                ? "text-fire bg-bg-fire"
                : "text-text-2 hover:text-text hover:bg-bg-warm"
            }`}
          >
            {label}
          </Link>
        ))}
      </div>

      <SearchBar />
    </nav>
  );
}
