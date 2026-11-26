# AspectPulse AI — Granular Aspect-Based Sentiment Analysis & Predictive Analytics Engine

A full-stack, three-phase NLP and econometric analytics pipeline that transforms unstructured product reviews into granular aspect polarities, token-level Captum attribution explanations, and statistical quality/risk forecasts—served via FastAPI and SQLite.

---

## System Architecture

```text
Unstructured Review
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Aspect Term Extraction (ATE)                    │
│ Fine-tuned RoBERTa BIO Sequence Tagger (B-ASP/I-ASP/O)   │
└───────────────────────────┬──────────────────────────────┘
                            │ Extracted Aspect Spans
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 2: Aspect Polarity & Captum XAI (APC)              │
│ Sentence-Aspect Pair Classifier + Integrated Gradients   │
└───────────────────────────┬──────────────────────────────┘
                            │ Weighted Aspect Vector (X1..X7)
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 3: Predictive Analytics & Risk Engine              │
│ • Multiple & Quadratic OLS Regression (Star Rating)      │
│ • Dummy Logit & Probit Models (Return Risk Probability)  │
│ • Apriori Association Rule Mining (Defect Co-occurrence) │
│ • Simple Exponential Smoothing (Sentiment Forecasting)   │
└──────────────────────────────────────────────────────────┘
