import os
import json
import sqlite3
import uuid
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)
from captum.attr import LayerIntegratedGradients
from src.config import ABSAConfig, ATE_ID2LABEL, APC_ID2LABEL


class ABSAEngine:
    def __init__(self, cfg: ABSAConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ate_dir = os.path.join(cfg.output_dir, "ate_deberta_v3")
        apc_dir = os.path.join(cfg.output_dir, "apc_deberta_v3")

        if not os.path.isdir(ate_dir) or not os.path.isdir(apc_dir):
            raise FileNotFoundError(
                f"Trained models not found in '{cfg.output_dir}'. "
                "Please run 'python main.py --step ate' and 'python main.py --step apc' first."
            )

        self.ate_tokenizer = AutoTokenizer.from_pretrained(ate_dir, use_fast=True, local_files_only=True)
        self.ate_model = AutoModelForTokenClassification.from_pretrained(ate_dir, local_files_only=True).to(self.device).eval()

        self.apc_tokenizer = AutoTokenizer.from_pretrained(apc_dir, use_fast=True, local_files_only=True)
        self.apc_model = AutoModelForSequenceClassification.from_pretrained(apc_dir, local_files_only=True).to(self.device).eval()

        embedding_layer = self.apc_model.get_input_embeddings()
        self.lig = LayerIntegratedGradients(self._forward_apc, embedding_layer)

    def _forward_apc(self, input_ids, attention_mask):
        outputs = self.apc_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    @torch.no_grad()
    def extract_aspects(self, text: str) -> list:
        encoded = self.ate_tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

        logits = self.ate_model(**inputs).logits[0]
        pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        aspects = []
        current_start, current_end = None, None

        for tag_id, (start_char, end_char) in zip(pred_ids, offsets):
            if start_char == 0 and end_char == 0:
                continue
            tag = ATE_ID2LABEL[tag_id]

            if tag == "B-ASP":
                if current_start is not None:
                    aspects.append({
                        "aspect_term": text[current_start:current_end].strip(),
                        "from": current_start,
                        "to": current_end,
                    })
                current_start, current_end = start_char, end_char
            elif tag == "I-ASP" and current_start is not None:
                current_end = end_char
            else:
                if current_start is not None:
                    aspects.append({
                        "aspect_term": text[current_start:current_end].strip(),
                        "from": current_start,
                        "to": current_end,
                    })
                    current_start, current_end = None, None

        if current_start is not None:
            aspects.append({
                "aspect_term": text[current_start:current_end].strip(),
                "from": current_start,
                "to": current_end,
            })

        return [a for a in aspects if a["aspect_term"]]

    def classify_and_explain(self, text: str, aspect_term: str, n_steps: int = 25) -> dict:
        encoded = self.apc_tokenizer(
            text,
            aspect_term,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        with torch.no_grad():
            logits = self.apc_model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_class = int(np.argmax(probs))

        pad_id = self.apc_tokenizer.pad_token_id or 0
        baseline_ids = torch.full_like(input_ids, pad_id)
        for special_id in self.apc_tokenizer.all_special_ids:
            baseline_ids[input_ids == special_id] = special_id

        attributions = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=pred_class,
            n_steps=n_steps,
        )

        token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        max_abs = np.max(np.abs(token_scores)) + 1e-9
        norm_scores = (token_scores / max_abs).tolist()

        tokens = self.apc_tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        sep_token = self.apc_tokenizer.sep_token
        token_attributions = []
        # Only keep attributions belonging to the review sentence (before first SEP)
        for tok, score in zip(tokens, norm_scores):
            if tok == sep_token and len(token_attributions) > 0:
                break
            if tok in self.apc_tokenizer.all_special_tokens:
                continue
            clean_tok = tok.replace("\u0120", " ").replace("\u2581", " ").strip()
            if clean_tok:
                token_attributions.append({"token": clean_tok, "score": round(score, 4)})

        return {
            "aspect_term": aspect_term,
            "polarity": APC_ID2LABEL[pred_class],
            "confidence": round(float(probs[pred_class]), 4),
            "probabilities": {
                APC_ID2LABEL[i]: round(float(probs[i]), 4) for i in range(len(probs))
            },
            "token_attributions": token_attributions,
        }

    def analyze_review(self, text: str, save_to_db: bool = False) -> dict:
        extracted = self.extract_aspects(text)
        results = []
        for item in extracted:
            explained = self.classify_and_explain(text, item["aspect_term"])
            explained["from"] = item["from"]
            explained["to"] = item["to"]
            results.append(explained)

        if save_to_db and results and os.path.exists(self.cfg.sqlite_db_path):
            self._persist_results(text, results)

        return {"review_text": text, "aspects": results}

    def _persist_results(self, text: str, results: list):
        conn = sqlite3.connect(self.cfg.sqlite_db_path)
        review_id = str(uuid.uuid4())[:8]
        for r in results:
            conn.execute(
                "INSERT INTO inferred_aspects (review_id, review_text, aspect_term, polarity, confidence) VALUES (?, ?, ?, ?, ?)",
                (review_id, text, r["aspect_term"], r["polarity"], r["confidence"]),
            )
        conn.commit()
        conn.close()


if __name__ == "__main__":
    engine = ABSAEngine(ABSAConfig())
    sample = "The battery life is exceptional, but the build quality feels cheap and fragile."
    print(json.dumps(engine.analyze_review(sample), indent=2))
