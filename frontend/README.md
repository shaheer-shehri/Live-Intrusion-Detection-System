# IDS Frontend - Installation & Running Guide

## Prerequisites

- Node.js 18+ (`npm --version`)
- Backend API running on `http://localhost:8000`

## Installation

```bash
cd frontend
npm install
```

## Development Server

```bash
npm run dev
```

Navigate to **http://localhost:5173** in your browser.

The frontend will automatically:
- Connect to the backend API at `http://localhost:8000`
- Auto-refresh metrics every 5 seconds
- Display real-time predictions and drift monitoring

## Build for Production

```bash
npm run build
```

This creates an optimized build in `dist/` folder (~500KB gzipped).

## Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t ids-frontend .

# Run container
docker run -p 3000:3000 ids-frontend
```

Access at **http://localhost:3000**

## Features

✅ **Dashboard** - Real-time metrics, latency, throughput  
✅ **Predict** - Single network flow classification  
✅ **Batch Upload** - CSV file bulk prediction  
✅ **Drift Monitor** - Data drift detection with severity alerts  
✅ **Metrics** - Detailed API performance tracking  
✅ **Dark Theme** - Professional cybersecurity UI  

## Architecture

- **React 18** + TypeScript
- **Tailwind CSS** for styling
- **Chart.js** for visualizations
- **Axios** for API communication
- **Lucide React** for icons

## API Integration

The frontend connects to 5 backend endpoints:

1. `POST /predict` - Single flow prediction
2. `POST /predict-batch` - Batch predictions
3. `POST /assess-drift` - Drift assessment
4. `GET /metrics` - Performance metrics
5. `GET /health` - Health check

All requests/responses are automatically handled by `src/services/api.ts`

## Development Structure

```
frontend/
├── src/
│   ├── pages/
│   │   ├── Dashboard.tsx          # Main metrics dashboard
│   │   ├── PredictPage.tsx        # Single prediction form
│   │   ├── BatchUploadPage.tsx    # CSV upload
│   │   ├── DriftPage.tsx          # Drift monitoring
│   │   └── MetricsPage.tsx        # Performance metrics
│   ├── components/
│   │   ├── Navbar.tsx             # Top navigation
│   │   └── Sidebar.tsx            # Side menu
│   ├── services/
│   │   └── api.ts                 # Backend API client
│   ├── App.tsx                    # Main app component
│   ├── main.tsx                   # Entry point
│   └── index.css                  # Global + Tailwind styles
├── tailwind.config.js             # Tailwind configuration
├── postcss.config.js              # PostCSS configuration
├── vite.config.ts                 # Vite build configuration
├── Dockerfile                     # Container image
└── package.json                   # Dependencies
```

## Troubleshooting

### API Connection Failed

Ensure backend is running:

```bash
cd ..  # Go to project root
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Port Already in Use

Change frontend port:

```bash
npm run dev -- --port 5174
```

### CORS Errors

The backend needs CORS middleware enabled in `api/main.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Performance

- **Build size:** ~500KB gzipped
- **Initial load:** <2s
- **Page transitions:** Instant
- **Real-time updates:** 5s refresh rate
- **API latency:** <100ms average (shown in metrics)

## Future Enhancements

- WebSocket real-time predictions
- Export predictions to CSV/JSON
- Model comparison charts
- Advanced filtering & search
- User authentication
- Dark/Light theme toggle
- Mobile responsive design

---

**Built with ❤️ for cybersecurity professionals**
