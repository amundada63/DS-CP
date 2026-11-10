# AeroTwin API

Synthetic software-in-the-loop demonstration backend. It trains Isolation Forest and Random Forest models against generated illustrative telemetry at startup. It is not validated for real aircraft use.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs
