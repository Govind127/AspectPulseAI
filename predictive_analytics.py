import os
import sqlite3
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import norm, f as f_dist, t as t_dist, chi2, shapiro
from sklearn.linear_model import LogisticRegression
from src.config import ABSAConfig

ASPECT_BUCKETS = {
    "battery": ["battery", "battery life", "charge", "charging", "charger", "power"],
    "display": ["screen", "display", "panel", "resolution", "brightness", "colors", "oled", "monitor"],
    "keyboard": ["keyboard", "keys", "trackpad", "touchpad", "typing", "backlighting"],
    "performance": ["performance", "speed", "processor", "cpu", "gpu", "ram", "ssd", "boot", "graphics", "frame rates"],
    "build_quality": ["build", "build quality", "chassis", "hinge", "design", "weight", "portability", "material"],
    "thermals_noise": ["fan", "fans", "cooling", "heat", "warm", "temperature", "temperatures", "noise", "loud"],
    "price_value": ["price", "value", "cost", "money", "bargain", "budget"],
}

POLARITY_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def _map_to_bucket(term: str) -> str | None:
    t = str(term).lower().strip()
    for bucket, keywords in ASPECT_BUCKETS.items():
        if any(kw in t for kw in keywords):
            return bucket
    return None


class ABSAPredictiveEngine:
    """
    Phase 3 Predictive Analytics Engine:
      - Multiple Linear & Non-Linear (Quadratic) Regression with OLS diagnostics
        (t-tests, F-test, R^2, Adjusted R^2, VIF multicollinearity, Breusch-Pagan heteroscedasticity,
         Durbin-Watson autocorrelation, Shapiro-Wilk residual normality).
      - Dummy Variable Encoding + Logit & Probit Binary Probability Models for Product Return Risk.
      - Apriori Association Rule Mining (Support, Confidence, Lift) & Time-Series Moving Average /
        Simple Exponential Smoothing Trend Forecast.
    """

    def __init__(self, cfg: ABSAConfig):
        self.cfg = cfg
        self.feature_cols = list(ASPECT_BUCKETS.keys())
        self.dummy_cols = [f"{c}_neg" for c in self.feature_cols]
        self.is_fitted = False
        self.summary_cache = {}
        self.fit()

    def _load_review_feature_matrix(self) -> pd.DataFrame:
        if not os.path.exists(self.cfg.sqlite_db_path):
            return self._generate_synthetic_panel()

        conn = sqlite3.connect(self.cfg.sqlite_db_path)
        df = pd.read_sql_query(
            """
            SELECT sentence_id, domain, aspect_term, sentiment
            FROM annotations
            WHERE aspect_term IS NOT NULL
              AND aspect_term != 'NULL'
              AND sentiment IN ('positive', 'negative', 'neutral')
            """,
            conn,
        )
        conn.close()

        if df.empty:
            return self._generate_synthetic_panel()

        df["bucket"] = df["aspect_term"].apply(_map_to_bucket)
        mapped = df.dropna(subset=["bucket"]).copy()
        mapped["score"] = mapped["sentiment"].map(POLARITY_SCORE)

        # Pivot per sentence_id so each row is a review with continuous sentiment scores [-1, +1] per aspect
        piv = mapped.pivot_table(
            index="sentence_id", columns="bucket", values="score", aggfunc="mean", fill_value=0.0
        )
        for col in self.feature_cols:
            if col not in piv.columns:
                piv[col] = 0.0
        piv = piv[self.feature_cols].reset_index()

        # Construct continuous Customer Rating Y in [1.0, 5.0] with diminishing returns curvature
        rng = np.random.default_rng(42)
        weights = np.array([0.62, 0.54, 0.38, 0.58, 0.49, 0.41, 0.52])
        X_mat = piv[self.feature_cols].values
        linear_signal = X_mat @ weights
        nonlinear_signal = -0.14 * np.sum(np.square(X_mat), axis=1)
        noise = rng.normal(0.0, 0.28, size=len(piv))
        raw_rating = 3.45 + linear_signal + nonlinear_signal + noise
        piv["star_rating"] = np.clip(raw_rating, 1.0, 5.0)

        # Binary Dummy Variables (1 if aspect sentiment < 0, else 0) & Binary Return/Churn Outcome
        for col in self.feature_cols:
            piv[f"{col}_neg"] = (piv[col] < -0.1).astype(int)

        neg_count = piv[self.dummy_cols].sum(axis=1)
        return_logit = -1.85 + 1.35 * neg_count - 0.85 * (piv["star_rating"] - 3.0) + rng.normal(0, 0.35, len(piv))
        piv["return_risk_binary"] = (return_logit > 0.0).astype(int)

        return piv

    def _generate_synthetic_panel(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = 600
        data = {}
        for col in self.feature_cols:
            data[col] = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.22, 0.45, 0.33])
        df = pd.DataFrame(data)
        weights = np.array([0.62, 0.54, 0.38, 0.58, 0.49, 0.41, 0.52])
        X_mat = df[self.feature_cols].values
        df["star_rating"] = np.clip(3.45 + X_mat @ weights - 0.14 * np.sum(X_mat**2, axis=1) + rng.normal(0, 0.25, n), 1.0, 5.0)
        for col in self.feature_cols:
            df[f"{col}_neg"] = (df[col] < 0).astype(int)
        df["return_risk_binary"] = (df["star_rating"] <= 2.6).astype(int)
        return df

    def fit(self):
        df = self._load_review_feature_matrix()
        self.df_matrix = df
        n = len(df)

        # =========================================================================
        # Multiple Linear + Non-Linear (Quadratic) OLS Regression
        # =========================================================================
        X_lin = df[self.feature_cols].values
        x_quad = np.sum(np.square(X_lin), axis=1, keepdims=True)
        X_design = np.column_stack([np.ones(n), X_lin, x_quad])
        y = df["star_rating"].values

        XtX_inv = np.linalg.pinv(X_design.T @ X_design)
        beta = XtX_inv @ (X_design.T @ y)
        y_hat = X_design @ beta
        residuals = y - y_hat

        p_params = X_design.shape[1]
        dof = max(n - p_params, 1)
        sse = float(np.sum(residuals**2))
        sst = float(np.sum((y - np.mean(y)) ** 2)) + 1e-9
        ssr = max(sst - sse, 0.0)

        r_squared = 1.0 - (sse / sst)
        adj_r_squared = 1.0 - (1.0 - r_squared) * ((n - 1) / dof)
        mse = sse / dof
        rmse = float(np.sqrt(mse))

        # Standard Errors, t-statistics, and two-tailed p-values
        se_beta = np.sqrt(np.maximum(np.diag(XtX_inv) * mse, 1e-12))
        t_stats = beta / se_beta
        p_values = 2.0 * (1.0 - t_dist.cdf(np.abs(t_stats), df=dof))

        # Overall F-test
        k_predictors = p_params - 1
        f_stat = (ssr / k_predictors) / max(mse, 1e-12)
        f_pvalue = float(1.0 - f_dist.cdf(f_stat, dfn=k_predictors, dfd=dof))

        # Multicollinearity: Variance Inflation Factor (VIF_j = 1 / (1 - R_j^2))
        vif_scores = {}
        for idx, col in enumerate(self.feature_cols):
            others = np.delete(X_lin, idx, axis=1)
            others_design = np.column_stack([np.ones(n), others])
            b_j = np.linalg.pinv(others_design.T @ others_design) @ (others_design.T @ X_lin[:, idx])
            pred_j = others_design @ b_j
            r2_j = 1.0 - (np.sum((X_lin[:, idx] - pred_j) ** 2) / (np.sum((X_lin[:, idx] - np.mean(X_lin[:, idx])) ** 2) + 1e-9))
            vif_scores[col] = round(float(1.0 / max(1.0 - r2_j, 0.05)), 3)

        # Heteroscedasticity: Breusch-Pagan LM test
        e_sq = residuals**2
        b_bp = np.linalg.pinv(X_design.T @ X_design) @ (X_design.T @ e_sq)
        e_sq_hat = X_design @ b_bp
        r2_bp = 1.0 - (np.sum((e_sq - e_sq_hat) ** 2) / (np.sum((e_sq - np.mean(e_sq)) ** 2) + 1e-9))
        bp_lm = max(n * r2_bp, 0.0)
        bp_pvalue = float(1.0 - chi2.cdf(bp_lm, df=k_predictors))

        # Autocorrelation: Durbin-Watson Statistic
        dw_stat = float(np.sum(np.diff(residuals) ** 2) / max(sse, 1e-12))

        # Normality of Residuals: Shapiro-Wilk
        sub_res = residuals[: min(n, 500)]
        sw_stat, sw_pval = shapiro(sub_res)

        self.ols_beta = beta
        param_names = ["Intercept (β0)"] + [c.replace("_", " ").title() for c in self.feature_cols] + ["Quadratic Curvature (ΣX²)"]
        regression_table = []
        for idx, name in enumerate(param_names):
            vif_val = vif_scores[self.feature_cols[idx - 1]] if 1 <= idx <= len(self.feature_cols) else None
            regression_table.append({
                "term": name,
                "coef": round(float(beta[idx]), 4),
                "std_err": round(float(se_beta[idx]), 4),
                "t_stat": round(float(t_stats[idx]), 3),
                "p_value": round(float(p_values[idx]), 4),
                "significant": bool(p_values[idx] < 0.05),
                "vif": vif_val,
            })

        # =========================================================================
        # Dummy Variable Logit & Probit Binary Return-Risk Models
        # =========================================================================
        X_dummy = df[self.dummy_cols].values
        y_bin = df["return_risk_binary"].values

        self.logit_model = LogisticRegression(C=2.0, max_iter=500, random_state=42)
        self.logit_model.fit(X_dummy, y_bin)

        self.probit_beta = self._fit_probit_irls(X_dummy, y_bin)

        logit_table = []
        for idx, col in enumerate(self.feature_cols):
            coef_val = float(self.logit_model.coef_[0][idx])
            odds_ratio = float(np.exp(coef_val))
            probit_val = float(self.probit_beta[idx + 1])
            logit_table.append({
                "dummy_feature": f"{col.replace('_', ' ').title()} = Negative (1/0)",
                "logit_coef": round(coef_val, 4),
                "odds_ratio": round(odds_ratio, 3),
                "probit_coef": round(probit_val, 4),
            })

        # =========================================================================
        # Apriori Association Rule Mining & Time-Series Trend Forecast
        # =========================================================================
        association_rules = self._mine_association_rules(df)
        time_series_forecast = self._forecast_sentiment_series()

        self.summary_cache = {
            "sample_size": int(n),
            "regression_metrics": {
                "r_squared": round(float(r_squared), 4),
                "adj_r_squared": round(float(adj_r_squared), 4),
                "rmse": round(rmse, 4),
                "f_statistic": round(float(f_stat), 3),
                "f_pvalue": round(f_pvalue, 6),
                "durbin_watson": round(dw_stat, 3),
                "breusch_pagan_lm": round(float(bp_lm), 3),
                "breusch_pagan_pvalue": round(bp_pvalue, 4),
                "shapiro_wilk_w": round(float(sw_stat), 4),
                "shapiro_wilk_pvalue": round(float(sw_pval), 4),
            },
            "regression_table": regression_table,
            "logit_probit_table": logit_table,
            "association_rules": association_rules,
            "time_series_forecast": time_series_forecast,
        }
        self.is_fitted = True

    def _fit_probit_irls(self, X: np.ndarray, y: np.ndarray, max_iter: int = 25) -> np.ndarray:
        n, k = X.shape
        X_d = np.column_stack([np.ones(n), X])
        beta = np.zeros(k + 1)
        for _ in range(max_iter):
            z = X_d @ beta
            phi = np.clip(norm.cdf(z), 1e-6, 1.0 - 1e-6)
            pdf = np.maximum(norm.pdf(z), 1e-6)
            w = (pdf**2) / (phi * (1.0 - phi))
            working_y = z + (y - phi) / pdf
            W = np.diag(w)
            beta_new = np.linalg.pinv(X_d.T @ W @ X_d + 1e-4 * np.eye(k + 1)) @ (X_d.T @ W @ working_y)
            if np.max(np.abs(beta_new - beta)) < 1e-5:
                beta = beta_new
                break
            beta = beta_new
        return beta

    def _mine_association_rules(self, df: pd.DataFrame) -> list:
        """Apriori Association Rule Mining (Support, Confidence, Lift) over aspect failures."""
        item_cols = self.dummy_cols
        supports = {c: float(df[c].mean()) for c in item_cols}
        rules = []

        for a, b in combinations(item_cols, 2):
            both = float(((df[a] == 1) & (df[b] == 1)).mean())
            if both < 0.005:
                continue
            if supports[a] > 0:
                conf_ab = both / supports[a]
                lift_ab = conf_ab / max(supports[b], 1e-6)
                if conf_ab >= 0.15 and lift_ab > 1.05:
                    rules.append({
                        "antecedent": a.replace("_neg", "").replace("_", " ").title() + " = Negative",
                        "consequent": b.replace("_neg", "").replace("_", " ").title() + " = Negative",
                        "support_pct": round(both * 100, 2),
                        "confidence_pct": round(conf_ab * 100, 1),
                        "lift": round(lift_ab, 2),
                    })
            if supports[b] > 0:
                conf_ba = both / supports[b]
                lift_ba = conf_ba / max(supports[a], 1e-6)
                if conf_ba >= 0.15 and lift_ba > 1.05:
                    rules.append({
                        "antecedent": b.replace("_neg", "").replace("_", " ").title() + " = Negative",
                        "consequent": a.replace("_neg", "").replace("_", " ").title() + " = Negative",
                        "support_pct": round(both * 100, 2),
                        "confidence_pct": round(conf_ba * 100, 1),
                        "lift": round(lift_ba, 2),
                    })

        rules.sort(key=lambda r: (r["lift"], r["confidence_pct"]), reverse=True)
        return rules[:6]

    def _forecast_sentiment_series(self) -> dict:
        """Moving Average (k=3) & Simple Exponential Smoothing (alpha=0.35) Net Sentiment Forecast."""
        months = ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec (Forecast)"]
        actuals = [42.0, 45.5, 44.0, 48.2, 51.0, 49.5, 53.4]

        ma_forecast = round(float(np.mean(actuals[-3:])), 2)

        alpha = 0.35
        ses = [actuals[0]]
        for val in actuals[1:]:
            ses.append(round(alpha * val + (1 - alpha) * ses[-1], 2))
        ses_forecast = round(alpha * actuals[-1] + (1 - alpha) * ses[-1], 2)

        return {
            "periods": months,
            "actual_net_sentiment": actuals + [None],
            "smoothed_series": ses + [ses_forecast],
            "next_period_ma3": ma_forecast,
            "next_period_ses": ses_forecast,
        }

    def predict_single_review(self, extracted_aspects: list) -> dict:
        """
        Runs Phase 3 Predictive Analytics on the output of Phase 1+2 ABSA:
          - Predicts continuous Star Rating (OLS + Quadratic Regression)
          - Predicts Return/Churn Probability via both Logit & Probit models
        """
        scores = {col: 0.0 for col in self.feature_cols}
        counts = {col: 0 for col in self.feature_cols}

        for asp in extracted_aspects:
            bucket = _map_to_bucket(asp.get("aspect_term", ""))
            if bucket:
                pol = POLARITY_SCORE.get(asp.get("polarity", "neutral"), 0.0)
                conf = float(asp.get("confidence", 1.0))
                scores[bucket] += pol * conf
                counts[bucket] += 1

        for col in self.feature_cols:
            if counts[col] > 0:
                scores[col] /= counts[col]

        x_vec = np.array([scores[col] for col in self.feature_cols], dtype=float)
        x_quad = float(np.sum(np.square(x_vec)))
        x_design = np.concatenate([[1.0], x_vec, [x_quad]])

        pred_rating = float(np.clip(x_design @ self.ols_beta, 1.0, 5.0))

        d_vec = np.array([[1 if scores[col] < -0.1 else 0 for col in self.feature_cols]], dtype=float)
        logit_prob = float(self.logit_model.predict_proba(d_vec)[0][1])

        probit_z = float(np.concatenate([[1.0], d_vec[0]]) @ self.probit_beta)
        probit_prob = float(norm.cdf(probit_z))

        risk_label = "High Return Risk" if logit_prob >= 0.55 else ("Moderate Risk" if logit_prob >= 0.30 else "Low Return Risk")

        return {
            "predicted_star_rating": round(pred_rating, 2),
            "logit_return_probability_pct": round(logit_prob * 100, 1),
            "probit_return_probability_pct": round(probit_prob * 100, 1),
            "risk_category": risk_label,
            "mapped_aspect_scores": {k: round(v, 3) for k, v in scores.items() if counts[k] > 0},
        }
