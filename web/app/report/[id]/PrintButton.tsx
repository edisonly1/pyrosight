"use client";

export default function PrintButton() {
  return (
    <button
      onClick={() => window.print()}
      className="px-5 py-2 bg-fire text-white rounded-lg text-sm font-semibold hover:bg-fire-dark transition-colors"
    >
      Print Report
    </button>
  );
}
