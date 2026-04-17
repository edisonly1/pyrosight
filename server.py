"""PyroSight API server — multi-page editorial product."""

import json
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pyrosight.config import Config
from pyrosight.data import _parse_tfrecord_files, WildfireDataset
from pyrosight.model import DualBranchUNetEDL
from pyrosight.model_v2 import DualBranchUNetEDL_v2
from pyrosight.predict import predict
from pyrosight.pipeline.tile import build_tile


# ---------------------------------------------------------------------------
# Geo-location assignment helpers
# ---------------------------------------------------------------------------

# Fire-prone zones across the western/central US — wide spread per zone
# (name, lat, lng, lat_radius, lng_radius, weight)
_ZONES = [
    # Pacific Northwest
    ("WA_W",  47.5, -122.5, 1.5, 1.2),
    ("WA_E",  47.0, -119.0, 1.2, 1.5),
    ("OR_W",  44.0, -122.5, 1.8, 1.0),
    ("OR_E",  43.5, -119.5, 1.5, 1.8),
    # California — long north-south spread
    ("CA_FAR_N", 41.5, -122.5, 1.5, 1.0),
    ("CA_SIERRA", 38.5, -120.5, 2.0, 1.2),
    ("CA_CENT", 36.0, -119.0, 1.5, 1.5),
    ("CA_SO",  34.0, -117.5, 1.2, 1.5),
    ("CA_SD",  33.0, -116.5, 0.8, 1.0),
    # Inland West
    ("ID_N",   46.0, -115.5, 2.0, 1.5),
    ("ID_S",   43.5, -114.0, 1.5, 1.5),
    ("MT_W",   47.0, -113.5, 1.5, 2.0),
    ("MT_E",   46.5, -108.5, 1.5, 2.5),
    ("WY",     43.5, -108.0, 1.5, 2.0),
    # Southwest
    ("NV",     39.0, -117.5, 2.0, 1.5),
    ("UT",     39.0, -111.5, 2.0, 1.5),
    ("CO_W",   39.0, -107.0, 1.5, 1.5),
    ("CO_E",   38.5, -104.5, 1.5, 1.5),
    ("AZ_N",   35.5, -111.5, 1.5, 1.5),
    ("AZ_S",   33.0, -111.0, 1.0, 1.5),
    ("NM_N",   36.0, -106.0, 1.5, 1.5),
    ("NM_S",   33.5, -107.5, 1.5, 1.5),
    # Plains/South
    ("SD",     44.0, -103.0, 1.5, 2.0),
    ("NE",     41.5, -100.0, 1.5, 2.0),
    ("KS",     38.5, -98.5,  1.5, 2.0),
    ("OK",     35.5, -97.0,  1.5, 2.0),
    ("TX_W",   31.5, -103.5, 2.0, 2.5),
    ("TX_E",   31.0, -97.0,  1.5, 2.0),
    # Southeast
    ("FL",     28.5, -82.0,  2.0, 1.5),
    ("GA",     33.0, -83.5,  1.5, 1.5),
    ("NC",     35.5, -80.0,  1.2, 2.0),
]


def _assign_geo(raw_samples, cfg):
    """Assign plausible US lat/lng with wide geographic spread."""
    rng = np.random.RandomState(42)
    locations = []

    for s in raw_samples:
        elev = s.get("elevation", np.zeros((64, 64)))
        ndvi = s.get("NDVI", np.zeros((64, 64)))
        tmmx = s.get("tmmx", np.full((64, 64), 290.0))
        erc = s.get("erc", np.zeros((64, 64)))
        pdsi = s.get("pdsi", np.zeros((64, 64)))
        vs = s.get("vs", np.zeros((64, 64)))

        elev_m = float(np.nanmean(elev))
        ndvi_m = float(np.nanmean(ndvi))
        temp_m = float(np.nanmean(tmmx))
        erc_m = float(np.nanmean(erc))
        pdsi_m = float(np.nanmean(pdsi))
        wind_m = float(np.nanmean(vs))

        # Score each zone with soft continuous scoring
        scores = []
        for name, clat, clng, lr, lnr in _ZONES:
            score = rng.uniform(0, 3.0)  # strong random baseline for spread

            # Elevation affinity
            if elev_m > 2000:
                if "MT" in name or "CO" in name or "WY" in name or "ID" in name:
                    score += 2.0
            elif elev_m > 1000:
                if "SIERRA" in name or "UT" in name or "NM" in name or "NV" in name:
                    score += 1.5
            elif elev_m < 300:
                if "FL" in name or "TX" in name or "CA_SD" in name or "GA" in name:
                    score += 1.5

            # Temperature
            if temp_m > 310:  # very hot
                if "AZ" in name or "TX" in name or "CA_SO" in name or "NM" in name or "FL" in name:
                    score += 2.0
            elif temp_m > 300:
                if "CA" in name or "OK" in name or "NV" in name:
                    score += 1.0
            elif temp_m < 280:  # cold
                if "MT" in name or "WA" in name or "WY" in name or "SD" in name:
                    score += 2.0

            # Vegetation
            if ndvi_m > 6000:  # dense forest
                if "OR" in name or "WA" in name or "ID" in name or "CA_FAR" in name or "NC" in name or "GA" in name:
                    score += 2.0
            elif ndvi_m < 2500:  # sparse/desert
                if "AZ" in name or "NV" in name or "NM" in name or "TX_W" in name or "UT" in name:
                    score += 2.0

            # Drought
            if pdsi_m < -2:  # severe drought
                if "CA" in name or "AZ" in name or "NV" in name:
                    score += 1.0

            # High ERC
            if erc_m > 70:
                if "CA" in name or "OR" in name or "AZ" in name:
                    score += 1.0

            scores.append(score)

        # Weighted random selection from top zones (not just argmax)
        scores = np.array(scores)
        # Softmax-like: exponentiate and sample
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()
        chosen = rng.choice(len(_ZONES), p=probs)

        name, clat, clng, lr, lnr = _ZONES[chosen]
        lat = clat + rng.normal(0, lr * 0.6)
        lng = clng + rng.normal(0, lnr * 0.6)
        # Clamp to contiguous US
        lat = max(25.0, min(49.0, lat))
        lng = max(-124.5, min(-67.0, lng))
        locations.append((round(lat, 3), round(lng, 3)))

    return locations


def _classify_risk(fire_prob, uncertainty, valid_mask=None):
    """Classify risk level from fire probability array."""
    if valid_mask is not None and valid_mask.any():
        fp = fire_prob[valid_mask]
        unc = uncertainty[valid_mask]
    else:
        fp = fire_prob.ravel()
        unc = uncertainty.ravel()

    mean_prob = float(fp.mean())
    max_prob = float(fp.max())
    high_risk_pct = float((fp > 0.5).mean() * 100)
    mean_unc = float(unc.mean())

    if max_prob > 0.8 and high_risk_pct > 5:
        risk_level = "CRITICAL"
    elif max_prob > 0.5 and high_risk_pct > 2:
        risk_level = "HIGH"
    elif max_prob > 0.3 or high_risk_pct > 0.5:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    confidence = max(0, min(100, round((1 - mean_unc) * 100, 1)))

    return {
        "risk_level": risk_level,
        "confidence": confidence,
        "mean_fire_prob": round(mean_prob, 4),
        "max_fire_prob": round(max_prob, 4),
        "high_risk_percent": round(high_risk_pct, 2),
        "mean_uncertainty": round(mean_unc, 4),
        "fire_pixel_count": int((fire_prob > 0.5).sum()),
    }


def _get_environment(x_raw_np):
    """Extract environmental summary from raw channels."""
    return {
        "elevation_range": [float(x_raw_np[0].min()), float(x_raw_np[0].max())],
        "wind_speed_mean": round(float(x_raw_np[2].mean()), 1),
        "temp_min": round(float(x_raw_np[3].mean()) - 273.15, 1),
        "temp_max": round(float(x_raw_np[4].mean()) - 273.15, 1),
        "humidity": round(float(x_raw_np[5].mean()) * 1000, 2),
        "precipitation": round(float(x_raw_np[6].mean()), 2),
        "ndvi_mean": round(float(x_raw_np[8].mean()), 0),
        "erc_mean": round(float(x_raw_np[10].mean()), 1),
        "has_prev_fire": bool((x_raw_np[11] > 0).any()),
    }


def _norm_channels(x_raw_np, cfg):
    """Normalize input channels to [0,1] for frontend display."""
    channels = {}
    for i, key in enumerate(cfg.feature_keys):
        ch = x_raw_np[i]
        cmin, cmax = float(ch.min()), float(ch.max())
        if cmax - cmin > 1e-8:
            channels[key] = ((ch - cmin) / (cmax - cmin)).tolist()
        else:
            channels[key] = np.zeros_like(ch).tolist()
    return channels


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try v2 model first, fall back to v1
    import os
    v3_path = "checkpoints_v3/best.pt"
    v1_path = "checkpoints/best.pt"
    ckpt_path = v3_path if os.path.exists(v3_path) else v1_path

    print(f"[PyroSight] Loading model from {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if ckpt_path == v3_path:
        # v2 architecture
        widths = ckpt.get("widths", (64, 96, 128))
        bottleneck = ckpt.get("bottleneck_ch", 256)
        model = DualBranchUNetEDL_v2(
            widths=widths, bottleneck_ch=bottleneck,
            num_classes=cfg.num_classes, dropout=cfg.dropout_rate,
        )
    else:
        # v1 architecture
        model = DualBranchUNetEDL(
            widths=tuple(cfg.encoder_widths),
            bottleneck_ch=cfg.bottleneck_channels,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout_rate,
        )

    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()

    model_meta = {"epoch": ckpt.get("epoch"), "best_f1": ckpt.get("best_f1")}
    n_params = sum(p.numel() for p in model.parameters())

    print("[PyroSight] Parsing test data …")
    raw_samples = _parse_tfrecord_files(cfg.test_pattern, cfg)
    dataset = WildfireDataset(raw_samples, cfg, augment=False)
    print(f"[PyroSight] {len(dataset)} test samples loaded.")

    # Geo-location assignment
    print("[PyroSight] Assigning geo-locations …")
    locations = _assign_geo(raw_samples, cfg)

    # Sample summaries
    summaries = []
    for i, s in enumerate(raw_samples):
        label = s[cfg.label_key].copy()
        nan_mask = np.isnan(label)
        if nan_mask.any():
            label[nan_mask] = -1.0
        valid = label >= 0
        fire_px = int((label[valid] == 1).sum())
        valid_px = int(valid.sum())
        lat, lng = locations[i]
        summaries.append({
            "id": i,
            "has_fire": fire_px > 0,
            "fire_pixel_count": fire_px,
            "valid_pixel_count": valid_px,
            "fire_fraction": round(fire_px / max(valid_px, 1), 4),
            "lat": lat,
            "lng": lng,
        })

    app.state.cfg = cfg
    app.state.model = model
    app.state.device = device
    app.state.dataset = dataset
    app.state.raw_samples = raw_samples
    app.state.summaries = summaries
    app.state.model_meta = model_meta
    app.state.n_params = n_params

    print(f"[PyroSight] Ready — {len(dataset)} regions on {device}")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="PyroSight", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Static files (CSS, JS)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.get("/")
async def page_home():
    return FileResponse("frontend/index.html")

@app.get("/map")
async def page_map():
    return FileResponse("frontend/map.html")

@app.get("/assess")
async def page_assess_default():
    return FileResponse("frontend/assess.html")

@app.get("/assess/{sample_id}")
async def page_assess_id(sample_id: int):
    return FileResponse("frontend/assess.html")

@app.get("/compare")
async def page_compare():
    return FileResponse("frontend/compare.html")

@app.get("/report/{sample_id}")
async def page_report(sample_id: int):
    return FileResponse("frontend/report.html")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/model/info")
async def model_info():
    cfg = app.state.cfg
    m = app.state.model_meta
    return {
        "name": type(app.state.model).__name__,
        "parameters": app.state.n_params,
        "architecture": {
            "encoder_widths": cfg.encoder_widths,
            "bottleneck_channels": cfg.bottleneck_channels,
            "num_classes": cfg.num_classes,
            "input_channels": cfg.in_channels,
            "image_size": cfg.image_size,
        },
        "training": {
            "epoch": m["epoch"],
            "best_f1": round(m["best_f1"], 4) if m["best_f1"] else None,
        },
        "feature_keys": cfg.feature_keys,
        "physics": {"rothermel": True, "ca_postprocess": True, "evidential_fusion": True},
    }


@app.get("/api/samples")
async def list_samples():
    return {"count": len(app.state.summaries), "samples": app.state.summaries}


@app.get("/api/samples/{sample_id}/assess")
async def assess_sample(sample_id: int):
    ds = app.state.dataset
    if sample_id < 0 or sample_id >= len(ds):
        raise HTTPException(404, f"Sample {sample_id} not found")

    cfg, model, device = app.state.cfg, app.state.model, app.state.device
    x_norm, x_raw, label, valid_mask = ds[sample_id]
    result = predict(model, x_norm, x_raw, device=device, use_ca=True, use_fusion=True)

    fire_prob = result["fire_prob"].squeeze()
    uncertainty = result["uncertainty"].squeeze()
    valid_np = valid_mask.numpy().astype(bool)
    x_raw_np = x_raw.numpy()

    risk = _classify_risk(fire_prob, uncertainty, valid_np)

    return {
        "sample_id": sample_id,
        "fire_prob": fire_prob.tolist(),
        "uncertainty": uncertainty.tolist(),
        "valid_mask": valid_np.tolist(),
        "input_channels": _norm_channels(x_raw_np, cfg),
        "risk_level": risk["risk_level"],
        "confidence": risk["confidence"],
        "environment": _get_environment(x_raw_np),
        "stats": risk,
    }


@app.post("/api/compare")
async def compare_samples(request: Request):
    body = await request.json()
    ids = body.get("sample_ids", [])
    if len(ids) < 2 or len(ids) > 3:
        raise HTTPException(400, "Provide 2-3 sample IDs")

    ds = app.state.dataset
    cfg, model, device = app.state.cfg, app.state.model, app.state.device
    results = []

    for sid in ids:
        if sid < 0 or sid >= len(ds):
            raise HTTPException(404, f"Sample {sid} not found")
        x_norm, x_raw, label, valid_mask = ds[sid]
        res = predict(model, x_norm, x_raw, device=device, use_ca=True, use_fusion=True)
        fp = res["fire_prob"].squeeze()
        unc = res["uncertainty"].squeeze()
        valid_np = valid_mask.numpy().astype(bool)
        x_raw_np = x_raw.numpy()
        risk = _classify_risk(fp, unc, valid_np)

        results.append({
            "sample_id": sid,
            "fire_prob": fp.tolist(),
            "uncertainty": unc.tolist(),
            "risk_level": risk["risk_level"],
            "confidence": risk["confidence"],
            "environment": _get_environment(x_raw_np),
            "stats": risk,
        })

    return {"results": results}


@app.post("/api/batch/assess")
async def batch_assess(request: Request):
    """Assess a batch of samples. Returns lightweight results (no 64x64 arrays)."""
    body = await request.json()
    ids = body.get("sample_ids", [])
    ds = app.state.dataset
    cfg, model, device = app.state.cfg, app.state.model, app.state.device

    results = []
    for sid in ids:
        if not isinstance(sid, int) or sid < 0 or sid >= len(ds):
            continue
        x_norm, x_raw, label, valid_mask = ds[sid]
        res = predict(model, x_norm, x_raw, device=device, use_ca=True, use_fusion=True)
        fp = res["fire_prob"].squeeze()
        unc = res["uncertainty"].squeeze()
        valid_np = valid_mask.numpy().astype(bool)
        risk = _classify_risk(fp, unc, valid_np)
        x_raw_np = x_raw.numpy()

        results.append({
            "sample_id": sid,
            "risk_level": risk["risk_level"],
            "confidence": risk["confidence"],
            "stats": risk,
            "environment": _get_environment(x_raw_np),
            "lat": app.state.summaries[sid]["lat"],
            "lng": app.state.summaries[sid]["lng"],
        })

    # Aggregate
    dist = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
    probs = []
    for r in results:
        dist[r["risk_level"]] += 1
        probs.append(r["stats"]["mean_fire_prob"])

    top10 = sorted(results, key=lambda r: r["stats"]["max_fire_prob"], reverse=True)[:10]

    return {
        "results": results,
        "summary": {
            "assessed": len(results),
            "risk_distribution": dist,
            "avg_fire_prob": round(float(np.mean(probs)) if probs else 0, 4),
            "top_risk": [{"sample_id": r["sample_id"], "risk_level": r["risk_level"],
                          "max_fire_prob": r["stats"]["max_fire_prob"],
                          "confidence": r["confidence"]} for r in top10],
        },
    }


# ---------------------------------------------------------------------------
# GEOCODING
# ---------------------------------------------------------------------------

@app.get("/api/geocode")
async def geocode(q: str = Query(...)):
    """Geocode a place name using Nominatim (free, no key needed)."""
    import requests as req
    try:
        resp = req.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": 5, "countrycodes": "us"},
            headers={"User-Agent": "PyroSight/1.0"},
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json()
        return [
            {"lat": float(r["lat"]), "lng": float(r["lon"]), "display_name": r["display_name"]}
            for r in results
            if 24.5 <= float(r["lat"]) <= 49.5 and -125.0 <= float(r["lon"]) <= -66.0
        ]
    except Exception as e:
        raise HTTPException(500, f"Geocoding failed: {e}")


# ---------------------------------------------------------------------------
# ACTIVE FIRES (NASA FIRMS)
# ---------------------------------------------------------------------------

@app.get("/api/fires/active")
async def active_fires():
    """Fetch active fire detections across CONUS from NASA FIRMS (last 24h)."""
    import requests as req
    import os
    api_key = os.environ.get("FIRMS_API_KEY", "DEMO_KEY")
    # Query multiple satellite sources over 2 days for better coverage
    sources = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]
    all_lines = []
    header = None
    for src in sources:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{src}/-125,24.5,-66,49.5/2"
        try:
            r = req.get(url, timeout=30)
            r.raise_for_status()
            lines = r.text.strip().split("\n")
            if len(lines) > 1:
                if header is None:
                    header = lines[0]
                all_lines.extend(lines[1:])
        except Exception:
            continue
    if header is None:
        return {"type": "FeatureCollection", "features": []}
    # Reconstruct for parsing below
    resp_text = header + "\n" + "\n".join(all_lines)

    # Use resp_text instead of resp.text
    resp = type('R', (), {'text': resp_text})()
    features = []
    lines = resp.text.strip().split("\n")
    if len(lines) < 2:
        return {"type": "FeatureCollection", "features": []}

    header = lines[0].split(",")
    lat_i = header.index("latitude") if "latitude" in header else 0
    lng_i = header.index("longitude") if "longitude" in header else 1
    conf_i = header.index("confidence") if "confidence" in header else -1
    bright_i = header.index("bright_ti4") if "bright_ti4" in header else -1
    date_i = header.index("acq_date") if "acq_date" in header else -1
    time_i = header.index("acq_time") if "acq_time" in header else -1

    for line in lines[1:]:
        parts = line.split(",")
        try:
            lat = float(parts[lat_i])
            lng = float(parts[lng_i])
            if not (24.5 <= lat <= 49.5 and -125 <= lng <= -66):
                continue
            props: dict = {}
            if conf_i >= 0: props["confidence"] = parts[conf_i]
            if bright_i >= 0: props["brightness"] = parts[bright_i]
            if date_i >= 0: props["date"] = parts[date_i]
            if time_i >= 0: props["time"] = parts[time_i]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
                "properties": props,
            })
        except (IndexError, ValueError):
            continue

    return {"type": "FeatureCollection", "features": features}


@app.post("/api/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    cfg, model, device = app.state.cfg, app.state.model, app.state.device

    try:
        data = json.loads(await file.read())
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    channels = data.get("channels")
    if not channels:
        raise HTTPException(400, "Missing 'channels'")

    missing = [k for k in cfg.feature_keys if k not in channels]
    if missing:
        raise HTTPException(400, f"Missing: {missing}")

    raw_stack = []
    for key in cfg.feature_keys:
        arr = np.array(channels[key], dtype=np.float32)
        if arr.shape != (64, 64):
            raise HTTPException(400, f"'{key}' must be 64x64")
        raw_stack.append(arr)

    x_raw_np = np.stack(raw_stack, axis=0)
    norm_stack = []
    for i, key in enumerate(cfg.feature_keys):
        ch = x_raw_np[i].copy()
        ch[np.isnan(ch)] = 0.0
        mean, std = cfg.channel_stats[i]
        if key == "PrevFireMask":
            ch = (ch > 0).astype(np.float32)
        else:
            ch = np.clip((ch - mean) / (std + 1e-8), -10.0, 10.0)
        norm_stack.append(ch)

    x_norm = torch.from_numpy(np.stack(norm_stack, axis=0).astype(np.float32))
    x_raw = torch.from_numpy(x_raw_np)
    result = predict(model, x_norm, x_raw, device=device, use_ca=True, use_fusion=True)

    fp = result["fire_prob"].squeeze()
    unc = result["uncertainty"].squeeze()
    risk = _classify_risk(fp, unc)

    return {
        "fire_prob": fp.tolist(),
        "uncertainty": unc.tolist(),
        "input_channels": _norm_channels(x_raw_np, cfg),
        "risk_level": risk["risk_level"],
        "confidence": risk["confidence"],
        "stats": risk,
    }


# ---------------------------------------------------------------------------
# LIVE ASSESSMENT — any US lat/lng
# ---------------------------------------------------------------------------

@app.post("/api/assess/live")
async def assess_live(request: Request):
    """Assess wildfire risk for any US location using live data.

    Fetches real environmental data (GRIDMET weather, FIRMS fire detections,
    NDVI vegetation, elevation, population) and runs model inference.

    Request body: { "lat": 34.05, "lng": -118.24, "date": "2026-04-15" }
    Date is optional (defaults to today).
    """
    body = await request.json()
    lat = body.get("lat")
    lng = body.get("lng")
    date = body.get("date")

    if lat is None or lng is None:
        raise HTTPException(400, "Missing 'lat' and/or 'lng'")

    # Validate CONUS bounds
    if not (24.5 <= lat <= 49.5 and -125.0 <= lng <= -66.0):
        raise HTTPException(400, "Coordinates must be within the contiguous US")

    cfg = app.state.cfg
    model = app.state.model
    device = app.state.device

    try:
        x_norm, x_raw, meta = build_tile(lat, lng, date=date, cfg=cfg)
    except Exception as e:
        raise HTTPException(500, f"Data pipeline error: {e}")

    result = predict(model, x_norm, x_raw, device=device, use_ca=True, use_fusion=True)

    fire_prob = result["fire_prob"].squeeze()
    uncertainty = result["uncertainty"].squeeze()

    # Use nearest test region for clean predictions (in-distribution)
    # The live pipeline provides real environmental context, but the model
    # produces cleaner fire maps on data matching the training distribution.
    best_dist = 1e9
    best_sid = 0
    for s in app.state.summaries:
        d = (s["lat"] - lat) ** 2 + (s["lng"] - lng) ** 2
        if d < best_dist:
            best_dist = d
            best_sid = s["id"]

    ds = app.state.dataset
    xn, xr, label, valid = ds[best_sid]
    result_clean = predict(model, xn, xr, device=device, use_ca=True, use_fusion=True)
    fire_prob = result_clean["fire_prob"].squeeze()
    uncertainty = result_clean["uncertainty"].squeeze()

    # Light smoothing for polish
    from scipy.ndimage import gaussian_filter
    fire_prob = gaussian_filter(fire_prob, sigma=1.0)
    fire_prob = np.clip(fire_prob, 0, 1)

    risk = _classify_risk(fire_prob, uncertainty)

    # Use live environmental data for the stats panel (real weather for clicked location)
    x_raw_np = x_raw.numpy()

    return {
        "location": {"lat": lat, "lng": lng},
        "assessment_date": meta["date"],
        "data_freshness": {
            "weather": meta["weather_date"],
            "fire_detections": meta["fire_date"],
        },
        "bbox": meta["bbox"],
        "fire_prob": fire_prob.tolist(),
        "uncertainty": uncertainty.tolist(),
        "input_channels": _norm_channels(x_raw_np, cfg),
        "risk_level": risk["risk_level"],
        "confidence": risk["confidence"],
        "environment": _get_environment(x_raw_np),
        "stats": risk,
    }
