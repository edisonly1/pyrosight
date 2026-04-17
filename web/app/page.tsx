import Link from "next/link";

export default function HomePage() {
  return (
    <main>
      {/* ── Hero ─────────────────────────────────────────────── */}
      <section
        className="min-h-screen flex flex-col items-center justify-center text-center px-6"
        style={{ background: "linear-gradient(180deg, #FEFDFB 0%, #F8F7F4 100%)" }}
      >
        <p className="font-mono text-xs uppercase tracking-widest text-text-3 mb-6">
          Real-Time Wildfire Risk Intelligence
        </p>
        <h1 className="font-serif text-7xl md:text-8xl font-semibold tracking-tighter text-text mb-6">
          PyroSight
        </h1>
        <p className="text-lg text-text-2 max-w-xl leading-relaxed mb-10">
          Click anywhere on the US map to get a next-day fire spread prediction
          powered by live satellite weather data, evidential deep learning, and
          Rothermel fire physics.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-4 mb-16">
          <Link
            href="/map"
            className="px-8 py-3.5 bg-fire text-white text-sm font-semibold rounded-lg hover:bg-fire-dark transition-colors no-underline"
          >
            Open Risk Map &rarr;
          </Link>
        </div>

        <div className="flex items-center gap-6 font-mono text-xs text-text-4">
          <span>Live GRIDMET Weather</span>
          <span>&middot;</span>
          <span>NASA FIRMS Fire Detection</span>
          <span>&middot;</span>
          <span>64&times;64 km Coverage</span>
        </div>
      </section>

      {/* ── What It Does ─────────────────────────────────────── */}
      <section className="max-w-4xl mx-auto px-6 md:px-10 py-24">
        <div className="w-12 h-[3px] bg-text mb-6" />
        <p className="font-mono text-xs uppercase tracking-[2px] text-text-3 mb-3">
          How It Works
        </p>
        <h2 className="font-serif text-3xl md:text-4xl font-semibold tracking-tight mb-4">
          From Click to Risk Assessment
        </h2>
        <p className="text-lg text-text-2 max-w-2xl leading-relaxed mb-16">
          When you click a location on the map, PyroSight fetches real environmental
          data and runs a physics-informed deep learning model to predict where fire
          will spread in the next 24 hours.
        </p>

        <div className="grid md:grid-cols-3 gap-8 mb-16">
          {[
            {
              step: "01",
              title: "Fetch Live Data",
              desc: "12 environmental channels are assembled from GRIDMET (weather), NASA FIRMS (active fires), MODIS (vegetation), SRTM (elevation), and GPWv4 (population) — all for a 64×64 km area centered on your click.",
            },
            {
              step: "02",
              title: "Run Inference",
              desc: "A dual-branch U-Net processes fuel data (terrain, vegetation) and weather data (wind, temperature, humidity) through cross-attention modules, then fuses with Rothermel fire-spread physics equations.",
            },
            {
              step: "03",
              title: "Assess Risk",
              desc: "The model outputs a per-pixel fire probability map with calibrated uncertainty via Evidential Deep Learning. Risk is classified as Critical, High, Moderate, or Low based on peak probability and area coverage.",
            },
          ].map((item) => (
            <div key={item.step} className="bg-bg-white border border-border rounded-xl p-6">
              <p className="font-mono text-xs text-fire font-semibold mb-3">{item.step}</p>
              <h3 className="font-serif text-lg font-semibold mb-2">{item.title}</h3>
              <p className="text-sm text-text-2 leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Data Sources ─────────────────────────────────────── */}
      <section className="bg-bg-white border-y border-border">
        <div className="max-w-4xl mx-auto px-6 md:px-10 py-20">
          <div className="w-12 h-[3px] bg-text mb-6" />
          <p className="font-mono text-xs uppercase tracking-[2px] text-text-3 mb-3">
            Real Data Sources
          </p>
          <h2 className="font-serif text-3xl font-semibold tracking-tight mb-12">
            12 Environmental Channels
          </h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              ["GRIDMET", "Wind, Temp, Humidity, Precip, ERC", "Daily, 4km"],
              ["NASA FIRMS", "Active Fire Detections", "Hourly, 375m"],
              ["SRTM", "Terrain Elevation", "Static, 30m"],
              ["MODIS", "Vegetation Index (NDVI)", "8-day, 500m"],
              ["GPWv4", "Population Density", "Static, 1km"],
              ["GRIDMET", "Drought Index (PDSI)", "5-day, 4km"],
              ["GRIDMET", "Fire Energy (ERC)", "Daily, 4km"],
              ["MODIS", "Prior Fire Mask", "Daily, 1km"],
            ].map(([source, desc, res], i) => (
              <div key={i} className="bg-bg-warm rounded-lg p-4">
                <p className="font-mono text-xs text-fire font-semibold mb-1">{source}</p>
                <p className="text-sm font-medium text-text mb-1">{desc}</p>
                <p className="text-xs text-text-3">{res}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Model Architecture ───────────────────────────────── */}
      <section className="max-w-4xl mx-auto px-6 md:px-10 py-20">
        <div className="w-12 h-[3px] bg-text mb-6" />
        <p className="font-mono text-xs uppercase tracking-[2px] text-text-3 mb-3">
          The Science
        </p>
        <h2 className="font-serif text-3xl font-semibold tracking-tight mb-12">
          Model Architecture
        </h2>

        <div className="grid md:grid-cols-2 gap-12">
          <div>
            <h3 className="font-serif text-xl font-semibold mb-3">Dual-Branch U-Net</h3>
            <p className="text-text-2 leading-relaxed mb-6">
              The fuel branch (3&times;3 convolutions) processes terrain, vegetation,
              population, and prior fire data. The weather branch (5&times;5 depthwise-separable
              convolutions) handles wind, temperature, humidity, and precipitation.
              Cross-Attentive Feature Interaction Modules (CAFIM) fuse the branches at
              three encoder scales, learning where weather amplifies fuel risk.
            </p>
            <h3 className="font-serif text-xl font-semibold mb-3">Rothermel Physics</h3>
            <p className="text-text-2 leading-relaxed">
              A deterministic physics branch computes fire spread rate from the Rothermel
              equations — incorporating slope gradients (Sobel filters on elevation),
              wind direction vectors, and fuel moisture proxies. These physics features
              are fused with the neural network output, ensuring predictions respect
              physical fire behavior: fire goes downwind, uphill, and through dry vegetation.
            </p>
          </div>
          <div>
            <h3 className="font-serif text-xl font-semibold mb-3">Evidential Deep Learning</h3>
            <p className="text-text-2 leading-relaxed mb-6">
              Instead of a standard softmax, the model outputs Dirichlet distribution
              parameters (&alpha;) for each pixel. This provides calibrated epistemic
              uncertainty in a single forward pass — no ensembles or Monte Carlo dropout
              needed. The model can say &ldquo;90% fire probability with high confidence&rdquo;
              or &ldquo;60% probability but I&rsquo;m very uncertain&rdquo; — a critical
              distinction for operational decision-making.
            </p>
            <h3 className="font-serif text-xl font-semibold mb-3">Risk Classification</h3>
            <p className="text-text-2 leading-relaxed">
              <span className="inline-block px-2 py-0.5 rounded text-xs font-semibold bg-red text-white mr-1">CRITICAL</span> Peak &gt;80%, &gt;5% high-risk area.
              <span className="inline-block px-2 py-0.5 rounded text-xs font-semibold bg-fire text-white mx-1">HIGH</span> Peak &gt;50%, &gt;2% area.
              <span className="inline-block px-2 py-0.5 rounded text-xs font-semibold bg-amber text-white mx-1">MODERATE</span> Peak &gt;30%.
              <span className="inline-block px-2 py-0.5 rounded text-xs font-semibold bg-green text-white mx-1">LOW</span> Minimal risk.
              Confidence is derived from the model&rsquo;s evidential uncertainty.
            </p>
          </div>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────────── */}
      <section className="bg-bg-white border-y border-border">
        <div className="max-w-4xl mx-auto px-6 md:px-10 py-16 text-center">
          <h2 className="font-serif text-3xl font-semibold tracking-tight mb-4">
            Try It Now
          </h2>
          <p className="text-text-2 mb-8 max-w-lg mx-auto">
            Click anywhere on the US map to assess wildfire risk using real-time
            satellite and weather data. No account needed.
          </p>
          <Link
            href="/map"
            className="inline-block px-8 py-3.5 bg-fire text-white text-sm font-semibold rounded-lg hover:bg-fire-dark transition-colors no-underline"
          >
            Open the Risk Map &rarr;
          </Link>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────── */}
      <footer className="border-t border-border">
        <div className="max-w-4xl mx-auto px-6 md:px-10 py-10 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="font-serif text-lg font-semibold tracking-tight">PyroSight</span>
            <span className="text-xs text-text-4">Wildfire Risk Intelligence</span>
          </div>
          <p className="font-mono text-xs text-text-4">
            DualBranchUNetEDL &middot; 1.86M params &middot; Evidential Deep Learning &middot; Rothermel Physics
          </p>
        </div>
      </footer>
    </main>
  );
}
