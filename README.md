AspectPulse AITwo-Stage Transformer Aspect-Based Sentiment Analysis (ABSA), Explainable AI (Captum), and Econometric Predictive Analytics EngineAspectPulse AI is an end-to-end NLP and statistical modeling workbench that transforms unstructured, mixed-sentiment customer reviews into granular aspect polarities, token-level feature attributions, and forward-looking product quality predictions.Instead of collapsing a nuanced review ("The OLED display is stunning and boot speed is instant, but the battery drains in three hours") into a misleading aggregate score, AspectPulse AI extracts every product attribute, isolates its contextual sentiment, explains why the model predicted that polarity, and feeds those aspect signals into econometric regression and probability models to forecast star ratings and return risk.System ArchitectureRaw Customer Review
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ PHASE 1: Aspect Term Extraction (ATE)                             │
│ Fine-tuned RoBERTa-base BIO Token Classifier (O, B-ASP, I-ASP)    │
│ • Inverse-frequency class weighting [0.5, 2.0, 2.0]               │
└───────────────────────────────────┬───────────────────────────────┘
                                    │ Extracted Aspect Spans
                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│ PHASE 2: Aspect Polarity Classification (APC) & Explainability    │
│ Fine-tuned RoBERTa-base Sequence-Pair Classifier                  │
│ • Input: <s> Review Text </s></s> Aspect Term </s>                │
│ • Captum Layer Integrated Gradients (XAI Token Heatmaps)          │
└───────────────────────────────────┬───────────────────────────────┘
                                    │ Structured Aspect Vector (X₁…X₇)
                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│ PHASE 3: Econometric Predictive Analytics Engine                  │
│ • Multiple Linear & Quadratic OLS Regression (Star Rating Ŷ)      │
│ • Full Model Diagnostics (t-tests, F-test, VIF, Breusch-Pagan, DW)│
│ • Dummy-Variable Logit & Probit Return-Risk Probability Models    │
│ • Apriori Defect Association Rules & Exponential Smoothing        │
└───────────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
             FastAPI REST Backend + Interactive Tailwind UI
                   (Persisted in SQLite: absa_store.db)
Key FeaturesUnified Multi-Domain Data Pipeline: Normalizes heterogeneous SemEval-2014 XML and ACOS TSV datasets (24,265 records) into an indexed SQLite database (data/absa_store.db) with nullable span offsets.Class-Weighted RoBERTa Fine-Tuning: Custom PyTorch WeightedTokenTrainer and WeightedSequenceTrainer subclasses prevent majority-class collapse on skewed BIO tags and neutral polarities.Captum Integrated Gradients XAI: Hooks directly into the RoBERTa embedding layer against a PAD-token reference baseline to highlight the exact modifier tokens driving each sentiment decision.Econometric Predictive Analytics (src/predictive_analytics.py):OLS & Quadratic Curvature Regression: Estimates aspect impact weights ($\beta_i$) on 1–5 star ratings with diminishing-returns modeling ($\sum X_i^2$).Statistical Diagnostics: Computes individual $t$-test $p$-values, overall $F$-statistic, $R^2$ / Adjusted $R^2$, Variance Inflation Factor (VIF), Breusch-Pagan heteroscedasticity LM test, Durbin-Watson autocorrelation, and Shapiro-Wilk residual normality.Logit & Probit Binary Probability Models: Predicts customer return/churn probability $P(Y=1 \mid X) \in [0,1]$ from negative aspect dummy variables alongside Odds Ratios.Apriori Association Rule Mining & Time-Series Forecasting: Discovers co-occurring hardware defect rules (Support, Confidence, Lift) and forecasts net sentiment trends via Moving Average and Exponential Smoothing.Interactive FastAPI Workbench: Real-time inference dashboard with live XAI attribution heatmaps, SQLite telemetry, and statistical diagnostic tables.Empirical Benchmark ResultsEvaluated on held-out test splits across classical and Transformer modeling paradigms:Pipeline StageModel ArchitectureKey MetricTest ScoreGain vs. BaselineClassical BaselineTF-IDF (1–2 n-grams) + LinearSVCMacro-F161.18%—Phase 1 (ATE)roberta-base BIO Token TaggerSpan-Level F187.63%+26.45%Phase 2 (APC)roberta-base Sequence-Pair ClassifierMacro-F1 (Accuracy)80.67% (88.07%)+19.49%Project Structureaspectpulse-ai/
├── data/
│   ├── absa_store.db              # SQLite relational store (ground-truth + live audit logs)
│   └── unified_absa_dataset.csv   # Normalized 24,265-row dataset
├── outputs/
│   ├── ate_model/                 # Fine-tuned Phase 1 RoBERTa ATE weights & tokenizer
│   └── apc_model/                 # Fine-tuned Phase 2 RoBERTa APC weights & tokenizer
├── src/
│   ├── config.py                  # Central paths, hyperparameters, and BIO/polarity mappings
│   ├── extract_data.py            # SemEval XML & ACOS TSV parser and normalizer
│   ├── data_pipeline.py           # SQLite seeding, offset-aligned BIO tagging, pair datasets
│   ├── baselines.py               # TF-IDF + LinearSVC benchmark script
│   ├── train_ate.py               # Phase 1 class-weighted sequence tagger training
│   ├── train_apc.py               # Phase 2 class-weighted polarity classifier training
│   ├── inference_explain.py       # Dual-stage inference engine + Captum Integrated Gradients
│   └── predictive_analytics.py    # Phase 3 OLS, Diagnostics, Logit/Probit & Apriori engine
├── static/
│   └── index.html                 # Interactive single-page web workbench
├── app.py                         # FastAPI server & REST endpoints
├── main.py                        # Modular CLI pipeline orchestrator
└── requirements.txt               # Python dependency manifest
Quickstart & Installation1. Clone & Set Up Virtual Environmentgit clone https://github.com/<your-username>/aspectpulse-ai.git
cd aspectpulse-ai

python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
2. Launch the Web Workbench (Using Pre-Trained Checkpoints)uvicorn app:app --reload
Open http://127.0.0.1:8000 in your browser to run live reviews, inspect Captum token attribution heatmaps, and explore the Phase 3 econometric diagnostics.3. CLI Pipeline Orchestration (main.py)Run individual pipeline stages from the terminal as needed:python main.py --step extract    # Parse raw XML/TSV archives into unified CSV
python main.py --step db         # Seed SQLite database & build tokenized datasets
python main.py --step baseline   # Evaluate TF-IDF + LinearSVC baseline
python main.py --step ate        # Fine-tune Phase 1 RoBERTa ATE tagger
python main.py --step apc        # Fine-tune Phase 2 RoBERTa APC classifier
python main.py --step demo --text "The OLED display is bright, but the battery drains fast."
REST API EndpointsMethodEndpointDescriptionGET/Serves the interactive AspectPulse AI web dashboard (static/index.html)GET/api/healthReturns model readiness and active compute device (cuda / cpu)POST/api/analyzeRuns Phase 1 ATE, Phase 2 APC + Captum XAI, and Phase 3 Star/Return Risk predictionGET/api/statsReturns aggregate SQLite database and sentiment distribution telemetryGET/api/predictive/summaryReturns OLS coefficients, VIF, Breusch-Pagan, Durbin-Watson, Logit/Probit Odds Ratios, and Apriori rulesLicenseDeveloped by Govind Girish Raut (M.Sc. Data Science, Department of Technology, Savitribai Phule Pune University). Released for academic and research use.
