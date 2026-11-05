# EcoVision AI backend

Upload these files together with your trained `best.pt` to a GitHub repository.

Render build command: `pip install -r requirements.txt`
Render start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

Endpoints: `GET /health`, `POST /predict` (multipart image field named `file`).

Note: Render free instances sleep when idle and have limited CPU/RAM. Model inference may be slow or exceed free-tier memory limits.
