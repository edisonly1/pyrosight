"use client";

import { useState } from "react";
import { hasPin, addPin, removePin, getPins } from "@/lib/pins";

export default function PinButton({ id }: { id: number }) {
  const [pinned, setPinned] = useState(() => hasPin(id));
  const [count, setCount] = useState(() => getPins().length);

  const toggle = () => {
    if (pinned) {
      removePin(id);
      setPinned(false);
    } else {
      addPin(id);
      setPinned(true);
    }
    setCount(getPins().length);
    window.dispatchEvent(new Event("pinsUpdated"));
  };

  return (
    <button
      onClick={toggle}
      className={`px-5 py-2.5 rounded-lg text-sm font-semibold border transition-colors ${
        pinned
          ? "bg-bg-fire text-fire border-fire/20"
          : "bg-white text-text border-border hover:bg-bg-warm"
      }`}
    >
      {pinned ? `Pinned (${count}/3)` : "Pin for Comparison"}
    </button>
  );
}
