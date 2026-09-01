# Churn + ESG Full App

This is the complete application:
- `backend/` = FastAPI + ML API
- `frontend/` = Node.js + Express + EJS web app

## Start backend

From `backend/`:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the Telco CSV in `backend/` as:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

Then:

```bash
python train_model.py
python run_api.py
```

FastAPI:
- http://localhost:8000
- http://localhost:8000/docs

## Start frontend

Open a second terminal:

```bash
cd frontend
npm install
node server.js
```

Open:

http://localhost:3000

If the backend is hosted elsewhere:

```bash
API_BASE_URL=http://your-host:8000 node server.js
```

## Application flow

Browser form -> Express frontend -> FastAPI `/predict` -> saved ML model + K-Means -> result page.
