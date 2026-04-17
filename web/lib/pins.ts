export function getPins(): number[] {
  if (typeof window === "undefined") return [];
  try { return JSON.parse(localStorage.getItem("pyro_pins") || "[]"); }
  catch { return []; }
}

export function addPin(id: number) {
  const p = getPins();
  if (p.length < 3 && !p.includes(id)) {
    p.push(id);
    localStorage.setItem("pyro_pins", JSON.stringify(p));
  }
}

export function removePin(id: number) {
  const p = getPins().filter((i) => i !== id);
  localStorage.setItem("pyro_pins", JSON.stringify(p));
}

export function hasPin(id: number): boolean {
  return getPins().includes(id);
}

export function clearPins() {
  localStorage.removeItem("pyro_pins");
}
