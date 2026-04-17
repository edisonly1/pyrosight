# PyroSight Deployment Guide

## Architecture

```
Users → Vercel (Next.js frontend) → Railway/Render (FastAPI backend)
                                          ↓
                                   GRIDMET / NASA FIRMS APIs
```

- **Frontend**: Vercel (free tier — auto-deploys from GitHub)
- **Backend**: Railway or Render (Docker container with model + rasters)

---

## Step 1: Push to GitHub

```bash
cd /Users/Edison/Downloads/pyrosight

# Initialize git if not already
git init
git add -A
git commit -m "PyroSight: wildfire risk assessment with live data pipeline"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/pyrosight.git
git push -u origin main
```

**Important**: Add `.env` to `.gitignore` so your FIRMS API key isn't exposed:
```bash
echo ".env" >> .gitignore
```

---

## Step 2: Deploy Backend to Railway

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Click **New Project** → **Deploy from GitHub repo** → select `pyrosight`
3. Railway will detect the `Dockerfile` automatically
4. Add environment variable:
   - `FIRMS_API_KEY` = your NASA FIRMS API key
5. Railway assigns a public URL like `https://pyrosight-api-production.up.railway.app`
6. Note this URL — you'll need it for the frontend

**Settings to adjust:**
- Memory: at least 1 GB (model + rasters use ~500 MB)
- Startup time: ~30 seconds (parsing TFRecords + loading model)

### Alternative: Render

1. Go to [render.com](https://render.com) → New Web Service
2. Connect GitHub repo
3. Environment: **Docker**
4. Add env var: `FIRMS_API_KEY`
5. Instance: at least 1 GB RAM
6. Note the URL

---

## Step 3: Deploy Frontend to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. Click **Add New Project** → import `pyrosight` repo
3. Configure:
   - **Root Directory**: `web`
   - **Framework Preset**: Next.js (auto-detected)
   - **Build Command**: `npm run build` (default)
   - **Output Directory**: `.next` (default)
4. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = your Railway/Render backend URL (e.g. `https://pyrosight-api-production.up.railway.app`)
5. Click **Deploy**

Vercel will build the Next.js app and deploy it to a `.vercel.app` URL.

---

## Step 4: Custom Domain (Optional)

1. Buy a domain (Namecheap, Cloudflare, etc.)
2. In Vercel: Settings → Domains → Add your domain
3. Update DNS records as Vercel instructs
4. Done — `pyrosight.com` (or whatever you chose) is live

---

## Local Development

Run both servers locally:

```bash
# Terminal 1: Backend
cd /Users/Edison/Downloads/pyrosight
source .env
.venv/bin/uvicorn server:app --port 8000

# Terminal 2: Frontend
cd /Users/Edison/Downloads/pyrosight/web
npm run dev:next
```

Or use the combined command:
```bash
cd web
npm run dev  # starts both via concurrently
```

---

## Environment Variables Summary

| Variable | Where | Value |
|----------|-------|-------|
| `FIRMS_API_KEY` | Backend (Railway/Render) | Your NASA FIRMS API key |
| `NEXT_PUBLIC_API_URL` | Frontend (Vercel) | Backend URL (e.g. `https://pyrosight-api.up.railway.app`) |

---

## Costs

| Service | Free Tier | Notes |
|---------|-----------|-------|
| Vercel | 100 GB bandwidth/mo | More than enough |
| Railway | $5/mo credit | ~enough for light usage |
| Render | 750 hours/mo free | Sleeps after 15 min inactivity |
| NASA FIRMS | Free | 10 req/min |
| GRIDMET | Free | Public domain data |

**Total: $0–5/month** for a demo/hackathon deployment.
