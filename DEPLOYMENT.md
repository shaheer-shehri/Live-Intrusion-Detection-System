# Deployment Guide

Three pieces:

| Piece | What | Where |
|---|---|---|
| **Backend** | FastAPI + simulator | **Azure** (Container Apps recommended; uses your student credits) |
| **Frontend** | React + Vite static site | **Vercel** |
| **Local agent** | Tiny Python script that watches your local DNS / TCP and tells the cloud backend to fire an attack scenario when you visit a watched URL | **Your own laptop** |

> **Why a local agent?** Once the backend lives in Azure, it sees the container's network — not your browser's. So the in-process `DomainWatcher` cannot detect that *you* opened `hackthissite.org`. The local agent solves that: same detection logic, runs on your machine, and POSTs `/trigger/<scenario>` to the cloud backend whenever it spots a watched domain.

---

## 1. Local Docker run

```bash
docker compose up --build
```
- Backend  → http://localhost:8000  (Swagger at `/docs`)
- Frontend → http://localhost:5173

When everything is local, the in-process watcher works directly (no agent needed).

---

## 2. Deploy backend to Azure

You have **two good options** with student credits. Container Apps is the easiest; App Service for Containers is the next step up if you want more knobs.

### Option A — Azure Container Apps (recommended)

**Step 1.** Install Azure CLI and log in
```bash
az login                              # opens a browser
az account set --subscription "Azure for Students"
```

**Step 2.** Pick a region & names
```bash
az group create -n ids-rg -l eastus

az acr create -n idsregistry$RANDOM -g ids-rg --sku Basic --admin-enabled true
# note the registry name printed, e.g. idsregistry47291
```

**Step 3.** Build the image *inside* Azure Container Registry (so you don't push from your machine)
```bash
az acr build -r idsregistry47291 -t ids-backend:latest .
```
This uploads the build context and runs `docker build` server-side (~5–8 min).

**Step 4.** Create the Container Apps environment
```bash
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App

az containerapp env create -n ids-env -g ids-rg -l eastus
```

**Step 5.** Deploy the container
```bash
ACR_PASSWORD=$(az acr credential show -n idsregistry47291 --query "passwords[0].value" -o tsv)

az containerapp create \
  -n ids-backend -g ids-rg --environment ids-env \
  --image idsregistry47291.azurecr.io/ids-backend:latest \
  --registry-server idsregistry47291.azurecr.io \
  --registry-username idsregistry47291 \
  --registry-password "$ACR_PASSWORD" \
  --target-port 8000 --ingress external \
  --cpu 1.0 --memory 2.0Gi \
  --min-replicas 1 --max-replicas 1 \
  --env-vars IDS_DISABLE_LOCAL_WATCHER=1 IDS_TRIGGER_TOKEN=change-me-please
```

`--min-replicas 1` keeps the simulator thread alive (otherwise Container Apps scales to zero and the simulator state is lost).
`IDS_DISABLE_LOCAL_WATCHER=1` skips the in-container DNS/TCP watcher (useless in the cloud).
`IDS_TRIGGER_TOKEN` protects `/trigger/{scenario}` so only your local agent (which knows the token) can fire attacks.

**Step 6.** Get the public URL
```bash
az containerapp show -n ids-backend -g ids-rg --query properties.configuration.ingress.fqdn -o tsv
# e.g. ids-backend.bluestone-1234abcd.eastus.azurecontainerapps.io
```

**Step 7.** Verify
```bash
curl https://<that-fqdn>/health
curl -N https://<that-fqdn>/monitor/live          # SSE stream — should print events every 1 s
```

To redeploy after code changes:
```bash
az acr build -r idsregistry47291 -t ids-backend:latest .
az containerapp update -n ids-backend -g ids-rg --image idsregistry47291.azurecr.io/ids-backend:latest
```

### Option B — Azure App Service for Containers

```bash
az group create -n ids-rg -l eastus
az acr create -n idsregistry$RANDOM -g ids-rg --sku Basic --admin-enabled true
az acr build -r <registry> -t ids-backend:latest .

az appservice plan create -n ids-plan -g ids-rg --is-linux --sku B2
az webapp create -g ids-rg -p ids-plan -n ids-backend-<unique> \
  --deployment-container-image-name <registry>.azurecr.io/ids-backend:latest

az webapp config appsettings set -g ids-rg -n ids-backend-<unique> --settings \
  WEBSITES_PORT=8000 \
  IDS_DISABLE_LOCAL_WATCHER=1 \
  IDS_TRIGGER_TOKEN=change-me-please
```
URL is `https://ids-backend-<unique>.azurewebsites.net`.

> Either option gives **HTTPS by default** — required because Vercel is HTTPS and would otherwise block mixed content.

---

## 3. Deploy frontend to Vercel

1. Push repo to GitHub.
2. Vercel → **Add New → Project** → import repo.
3. Settings:
   - **Root Directory:** `frontend`
   - **Framework Preset:** Vite
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. Environment Variables:

| Name | Value |
|---|---|
| `VITE_API_BASE` | `https://<your-azure-fqdn>` (no trailing slash) |

5. Click **Deploy**. Vercel gives you `https://ids-yourname.vercel.app`.

---

## 4. Update CORS on the backend

Edit `api/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ids-yourname.vercel.app",
        "http://localhost:5173",
    ],
    ...
)
```
Rebuild & redeploy:
```bash
az acr build -r <registry> -t ids-backend:latest .
az containerapp update -n ids-backend -g ids-rg --image <registry>.azurecr.io/ids-backend:latest
```

---

## 5. Run the local agent (this is what makes the demo work)

The cloud backend can no longer see your DNS lookups, so the local agent does it for you.

**Step 1.** On your laptop:
```bash
pip install psutil scapy   # already in requirements.txt; skip if you've installed them
```

**Step 2.** Set env vars (PowerShell):
```powershell
$env:IDS_BACKEND_URL    = "https://<your-azure-fqdn>"
$env:IDS_TRIGGER_TOKEN  = "change-me-please"        # the value you set in Azure
python local_agent.py
```

Bash/macOS:
```bash
export IDS_BACKEND_URL=https://<your-azure-fqdn>
export IDS_TRIGGER_TOKEN=change-me-please
python local_agent.py
```

You'll see:
```
[agent] backend  : https://ids-backend.xxx.eastus.azurecontainerapps.io
[agent] auth     : enabled
[agent] watching : 9 domains
[agent] running. Press Ctrl-C to stop.
```

**Step 3.** Open the Vercel URL in your browser. Live Monitor shows Normal traffic.

**Step 4.** In a new tab visit `https://hackthissite.org` (or any of the 5 demo domains). Within ~2 s the agent prints:
```
[DomainWatcher] dns-cache:hackthissite.org: triggered scenario 'generic'
[agent] generic → HTTP 200
```
…and the Vercel dashboard flips to attack flows for the next 35 s, then auto-reverts.

> The agent uses three detection methods in parallel (Windows DNS cache poll, `psutil` connection poll, optional Scapy sniff). At least one of these works without admin on every recent Windows install.

---

## 6. Architecture at a glance

```
   ┌──────────────────────────────────────────┐
   │           Your laptop (browser)          │
   │                                          │
   │  ┌─────────────┐    ┌─────────────────┐  │
   │  │ Vercel page │    │ local_agent.py  │  │
   │  └──────┬──────┘    └────────┬────────┘  │
   └─────────┼────────────────────┼───────────┘
       SSE   │                    │  POST /trigger/{scenario}
             ▼                    ▼
   ┌──────────────────────────────────────────┐
   │     Azure Container App (backend)        │
   │   FastAPI + TrafficSimulator (always-on) │
   └──────────────────────────────────────────┘
```

- The user's browser hits **Vercel** for the UI.
- Vercel page opens an SSE connection to the **Azure backend** (`/monitor/live`).
- `local_agent.py` running on the laptop watches the OS for visits to the 5 watched sites.
- When detected, the agent POSTs `/trigger/<scenario>` to the Azure backend with the bearer token.
- The simulator switches to attack mode → next SSE push includes attack flows → UI lights up red.

---

## 7. Common pitfalls

| Symptom | Fix |
|---|---|
| `az acr build` says provider not registered | `az provider register --namespace Microsoft.ContainerRegistry --wait` |
| Container Apps stuck "Activating" | Bump memory to 2 Gi (`--memory 2.0Gi`); model needs ~1 GB at load |
| Vercel page loads but `/monitor/live` 404 | `VITE_API_BASE` env var was not set at build time — set it then **redeploy** |
| CORS error in browser console | Add the exact Vercel URL to `allow_origins` and redeploy backend |
| Mixed-content error | Use HTTPS Azure URL (Container Apps & App Service both give HTTPS for free) |
| Agent prints `HTTP 401` | `IDS_TRIGGER_TOKEN` env var on backend ≠ value in your local shell |
| Agent triggers but UI shows no attack | Confirm SSE is connected (green ● LIVE badge) and you waited 1–2 s for the next push |
| Visiting domain doesn't trigger | Browser DNS cache had a fresh entry — clear it (`chrome://net-internals/#dns` → Clear host cache, or `ipconfig /flushdns`) |
