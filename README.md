# AeroTwin GCS — React + FastAPI + ML

A working software-in-the-loop prototype for the SIH 2026 AeroTwin concept: **AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs**.

The interface follows the supplied dark military-green/orange reference screens. It contains a live synthetic telemetry stream, sensor matrix, physics residuals, cycle diagnostics, cross-channel verification, fault injection, ML-based anomaly/fault inference, component dependency graph and a fuel/wind-based base reachability demonstration.

> **Important safety and academic note:** All telemetry, engine readings, mission data, model training examples, RUL outputs and base locations are synthetic illustrative data. The ML models are trained on generated examples at backend startup. The app is a course-project demonstration, not a certified digital twin, validated prognostic system, flight-control system or operational aviation advisory. Do not use it to operate an aircraft. Replace synthetic data and validate models against approved engine datasets and engineering requirements before making real-world claims.

## Project structure

```text
aerotwin-gcs/
├── backend/
│   ├── main.py              # FastAPI, synthetic stream, ML models, APIs, DSA
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── package.json
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── styles.css
```

## Run locally

You need Python 3.10+ and Node.js 18+.

### 1. Start the backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend API docs: http://localhost:8000/docs

### 2. Start the React frontend (new terminal)

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (usually http://localhost:5173).

## Implemented Data Structures

| Data structure | Where it is used | Purpose |
|---|---|---|
| FIFO Queue | `TelemetryQueue` | Buffers incoming telemetry packets before processing |
| Circular Queue / Buffer | `CircularBuffer` | Keeps the latest 180 telemetry records without unbounded memory growth |
| Hash Map | `SensorRegistry` | Maps sensor/channel IDs to their latest readings for average O(1) lookup |
| Priority Queue (min-heap) | `AlertPriorityQueue` | Orders alerts so critical events are surfaced before warnings |
| Graph (adjacency list) | `ComponentGraph` | Represents dependencies between fuel, combustion, thermal, exhaust and sensor components |
| Stack | `DiagnosticStack` | Retains recent diagnostic and scenario actions in LIFO order |

## Implemented Machine Learning

- **Isolation Forest:** unsupervised anomaly detector over standardized seven-channel telemetry features.
- **Random Forest Classifier:** illustrative five-class fault classifier: normal, cooling degradation, fuel-delivery anomaly, sensor drift and lubrication warning.
- **StandardScaler:** feature normalization before model inference.
- **RUL demonstration:** a transparent illustrative health-to-RUL calculation, not a trained or validated RUL model.
- **LSTM/GRU extension point:** the rolling history endpoint provides time-series sequences for a future PyTorch sequence model. An LSTM is not falsely represented as implemented in this build.
- **PINN extension point:** physics residuals are computed from a simple 0-D expected-value baseline. This build does not claim to train a PINN.

## Demo scenarios

- Scenario A: Reset / nominal stream
- Scenario B: Cooling leak
- Scenario C: Injector clog
- Sensor drift
- Oil warning (available via API: `POST /api/scenario` with `{"scenario":"oil_warning"}`)
- Full spoof attack (available via API: `POST /api/scenario` with `{"scenario":"full_spoof"}`)

The emergency reroute page allows a base to be marked offline and a headwind gust to be toggled, then recomputes illustrative reachability and fuel margins.

## Main API endpoints

`GET /api/health`, `/api/telemetry`, `/api/overview`, `/api/history`, `/api/sensors`, `/api/cycles`, `/api/crossverify`, `/api/diagnostics`, `/api/reroute`

`POST /api/scenario`, `/api/base`, `/api/wind`, `/api/reset`
