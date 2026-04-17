const THERMAL_STOPS: [number, [number, number, number]][] = [
  [0.0, [10, 10, 20]],
  [0.15, [60, 30, 120]],
  [0.3, [140, 40, 130]],
  [0.5, [214, 51, 100]],
  [0.65, [255, 107, 53]],
  [0.8, [255, 200, 60]],
  [1.0, [255, 252, 240]],
];

const TLUT = new Uint8Array(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  let lo = 0;
  for (let s = 1; s < THERMAL_STOPS.length; s++) {
    if (t <= THERMAL_STOPS[s][0]) { lo = s - 1; break; }
  }
  const [t0, c0] = THERMAL_STOPS[lo];
  const [t1, c1] = THERMAL_STOPS[lo + 1] || THERMAL_STOPS[lo];
  const f = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
  TLUT[i * 3] = Math.round(c0[0] + (c1[0] - c0[0]) * f);
  TLUT[i * 3 + 1] = Math.round(c0[1] + (c1[1] - c0[1]) * f);
  TLUT[i * 3 + 2] = Math.round(c0[2] + (c1[2] - c0[2]) * f);
}

export function drawThermal(canvas: HTMLCanvasElement, data: number[][]) {
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(64, 64);
  for (let y = 0; y < 64; y++)
    for (let x = 0; x < 64; x++) {
      const p = (y * 64 + x) * 4;
      const idx = Math.max(0, Math.min(255, Math.round(data[y][x] * 255)));
      img.data[p] = TLUT[idx * 3];
      img.data[p + 1] = TLUT[idx * 3 + 1];
      img.data[p + 2] = TLUT[idx * 3 + 2];
      img.data[p + 3] = 255;
    }
  ctx.putImageData(img, 0, 0);
}

export function drawUncertainty(canvas: HTMLCanvasElement, data: number[][]) {
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(64, 64);
  for (let y = 0; y < 64; y++)
    for (let x = 0; x < 64; x++) {
      const p = (y * 64 + x) * 4;
      const v = data[y][x];
      img.data[p] = Math.round(40 + 215 * v);
      img.data[p + 1] = Math.round(60 + 195 * v);
      img.data[p + 2] = Math.round(180 + 75 * v);
      img.data[p + 3] = Math.round(180 * v);
    }
  ctx.putImageData(img, 0, 0);
}

export function drawViridis(canvas: HTMLCanvasElement, data: number[][]) {
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(64, 64);
  for (let y = 0; y < 64; y++)
    for (let x = 0; x < 64; x++) {
      const p = (y * 64 + x) * 4;
      const v = data[y][x];
      let r: number, g: number, b: number;
      if (v < 0.5) {
        const t = v * 2;
        r = Math.round(68 * (1 - t) + 33 * t);
        g = Math.round(1 * (1 - t) + 145 * t);
        b = Math.round(84 * (1 - t) + 140 * t);
      } else {
        const t = (v - 0.5) * 2;
        r = Math.round(33 * (1 - t) + 253 * t);
        g = Math.round(145 * (1 - t) + 231 * t);
        b = Math.round(140 * (1 - t) + 37 * t);
      }
      img.data[p] = r;
      img.data[p + 1] = g;
      img.data[p + 2] = b;
      img.data[p + 3] = 255;
    }
  ctx.putImageData(img, 0, 0);
}

export function drawHist(canvas: HTMLCanvasElement, data: number[][]) {
  const ctx = canvas.getContext("2d")!;
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const flat = data.flat();
  const nB = 25;
  const bins = new Array(nB).fill(0);
  for (const v of flat) bins[Math.min(nB - 1, Math.floor(v * nB))]++;
  const mx = Math.max(...bins, 1);
  const bw = w / nB;
  for (let i = 0; i < nB; i++) {
    const bh = (bins[i] / mx) * (h - 4);
    ctx.fillStyle = `rgba(37,99,235,${0.25 + (i / nB) * 0.45})`;
    ctx.fillRect(i * bw + 1, h - bh - 2, bw - 2, bh);
  }
}

export function drawGrid(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext("2d")!;
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(0,0,0,0.06)";
  ctx.lineWidth = 1;
  const cw = w / 64, ch = h / 64;
  for (let i = 0; i <= 64; i += 8) {
    const px = Math.round(i * cw) + 0.5;
    const py = Math.round(i * ch) + 0.5;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(w, py); ctx.stroke();
  }
}
