# KnowledgePilot AI — React dashboard

The original Streamlit interface remains available. This folder adds a modern React UI backed by a FastAPI adapter.

## Run on Windows PowerShell

Terminal 1 — from the repository root:

```powershell
..venvScriptspython.exe -m pip install -r requirements-react.txt
..venvScriptspython.exe -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173.

Keep your existing `.env` in the repository root. Never put API keys in React code or commit `.env`. The API reads the existing Groq/OpenAI-compatible settings from the backend environment. UI changes to retrieval settings apply for the current API process; restart the API to reset them. The existing Streamlit app and evaluation workflow are unchanged.
