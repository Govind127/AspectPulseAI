import os
import sqlite3
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import ABSAConfig
from src.inference_explain import ABSAEngine
from src.predictive_analytics import ABSAPredictiveEngine

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="AspectPulse AI — ABSA + Predictive Analytics Workbench",
    description="Two-Stage Transformer ABSA + Captum XAI + Statistical Regression, Logit/Probit & Apriori Analytics",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = ABSAConfig()
engine: Optional[ABSAEngine] = None
pred_engine: Optional[ABSAPredictiveEngine] = None


@app.on_event("startup")
def load_models():
    global engine, pred_engine
    try:
        engine = ABSAEngine(cfg)
        print(f"[FastAPI] Loaded trained ABSA models on device: {engine.device}")
    except Exception as exc:
        print(f"[FastAPI Warning] Could not load ABSA models at startup: {exc}")

    pred_engine = ABSAPredictiveEngine(cfg)
    print("[FastAPI] Phase 3 Predictive Analytics Engine fitted successfully.")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=1500, description="Customer product review text")
    save_to_db: bool = Field(default=True, description="Persist extracted aspect polarities to SQLite")


@app.get("/")
def serve_landing_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "models_loaded": engine is not None,
        "predictive_engine_ready": pred_engine is not None and pred_engine.is_fitted,
        "device": str(engine.device) if engine else "unavailable",
        "ate_f1": 0.8763,
        "apc_macro_f1": 0.8067,
        "baseline_macro_f1": 0.6118,
    }


@app.post("/api/analyze")
def analyze_review(req: AnalyzeRequest):
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Ensure 'outputs/ate_deberta_v3' and 'outputs/apc_deberta_v3' exist.",
        )
    result = engine.analyze_review(req.text.strip(), save_to_db=req.save_to_db)
    if pred_engine is not None:
        result["predictive_forecast"] = pred_engine.predict_single_review(result["aspects"])
    return result


@app.get("/api/predictive/summary")
def get_predictive_summary():
    if pred_engine is None:
        raise HTTPException(status_code=503, detail="Predictive engine not initialized.")
    return pred_engine.summary_cache


@app.get("/api/stats")
def get_dataset_stats():
    if not os.path.exists(cfg.sqlite_db_path):
        return {"top_aspects": [], "recent_inferences": [], "total_annotations": 24265}

    conn = sqlite3.connect(cfg.sqlite_db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            LOWER(TRIM(aspect_term)) AS aspect,
            COUNT(*) AS total,
            SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) AS pos,
            SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) AS neg,
            SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) AS neu
        FROM annotations
        WHERE domain = 'laptop'
          AND aspect_term IS NOT NULL
          AND aspect_term != 'NULL'
          AND sentiment IN ('positive', 'negative', 'neutral')
        GROUP BY LOWER(TRIM(aspect_term))
        HAVING total >= 25
        ORDER BY total DESC
        LIMIT 8
    """).fetchall()

    top_aspects = []
    for r in rows:
        tot = r["total"]
        top_aspects.append({
            "aspect": r["aspect"].title(),
            "total": tot,
            "positive_pct": round((r["pos"] / tot) * 100, 1),
            "negative_pct": round((r["neg"] / tot) * 100, 1),
            "neutral_pct": round((r["neu"] / tot) * 100, 1),
        })

    recent = conn.execute("""
        SELECT review_id, review_text, aspect_term, polarity, confidence, created_at
        FROM inferred_aspects
        ORDER BY id DESC
        LIMIT 10
    """).fetchall()

    recent_inferences = [dict(r) for r in recent]
    conn.close()

    return {
        "total_annotations": 24265,
        "top_aspects": top_aspects,
        "recent_inferences": recent_inferences,
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
