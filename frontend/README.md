# KnowledgePilot AI — React UI

This folder contains a React + Vite frontend for the existing Python RAG pipeline. It includes a responsive dashboard, PDF library, cited chat, evaluation metrics, and runtime settings.

## Run locally (Windows PowerShell)

Open two PowerShell terminals from the repository root.

### Terminal 1 — Python API

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

The API reads the same root `.env` as the existing Streamlit app. Keep `LLM_API_KEY` in that file; it is never sent to the browser.

### Terminal 2 — React UI

```powershell
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173.

## Optional API URL

Create `frontend/.env.local` only if the backend is not running at the default address:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Notes

- The existing Streamlit UI remains available with `streamlit run app.py`.
- The React UI and Streamlit UI use the same local Chroma data directory and selected embedding collection.
- Changing the embedding backend switches to that backend's collection. Documents indexed with a different embedding model are stored separately; re-index PDFs when changing chunk size or overlap.
- Settings changed from React are in-memory for the running API session. Set long-term defaults in the root `.env`.
- This API is designed for local development. Do not expose it to the public internet without authentication, upload limits, and additional security controls.
