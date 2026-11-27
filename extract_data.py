import os
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from src.config import ABSAConfig

SENTIMENT_MAP = {"0": "negative", "1": "neutral", "2": "positive"}


def extract_zip_to_unified_csv(zip_path: str, output_csv: str) -> pd.DataFrame:
    """Extracts raw ACOS (.tsv) and SemEval-2014 Task 4 (.xml) files from Project ABSA.zip into a unified CSV."""
    records = []

    acos_files = [
        ("ACOS", "laptop", "dev", "Project ABSA/Dataset/ACOS-main/data/Laptop-ACOS/laptop_quad_dev.tsv"),
        ("ACOS", "laptop", "test", "Project ABSA/Dataset/ACOS-main/data/Laptop-ACOS/laptop_quad_test.tsv"),
        ("ACOS", "laptop", "train", "Project ABSA/Dataset/ACOS-main/data/Laptop-ACOS/laptop_quad_train.tsv"),
        ("ACOS", "restaurant", "dev", "Project ABSA/Dataset/ACOS-main/data/Restaurant-ACOS/rest16_quad_dev.tsv"),
        ("ACOS", "restaurant", "test", "Project ABSA/Dataset/ACOS-main/data/Restaurant-ACOS/rest16_quad_test.tsv"),
        ("ACOS", "restaurant", "train", "Project ABSA/Dataset/ACOS-main/data/Restaurant-ACOS/rest16_quad_train.tsv"),
    ]

    semeval_files = [
        ("SemEval2014Task4", "laptop", "trial", "Project ABSA/Dataset/SemEval2014Task4/laptops-trial.xml"),
        ("SemEval2014Task4", "laptop", "train", "Project ABSA/Dataset/SemEval2014Task4/SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train_v2.xml"),
        ("SemEval2014Task4", "laptop", "test", "Project ABSA/Dataset/SemEval2014Task4/ABSA_Gold_TestData/Laptops_Test_Gold.xml"),
        ("SemEval2014Task4", "restaurant", "trial", "Project ABSA/Dataset/SemEval2014Task4/restaurants-trial.xml"),
        ("SemEval2014Task4", "restaurant", "train", "Project ABSA/Dataset/SemEval2014Task4/SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml"),
        ("SemEval2014Task4", "restaurant", "test", "Project ABSA/Dataset/SemEval2014Task4/ABSA_Gold_TestData/Restaurants_Test_Gold.xml"),
    ]

    with zipfile.ZipFile(zip_path, "r") as z:
        for dataset, domain, split, rel_path in acos_files:
            content = z.read(rel_path).decode("utf-8")
            for line_idx, line in enumerate(content.splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split("\t")
                text = parts[0]
                tokens = text.split(" ")
                quads = parts[1:]

                if not quads:
                    records.append({
                        "dataset": dataset, "domain": domain, "split": split,
                        "source_file": os.path.basename(rel_path),
                        "sentence_id": f"{domain}_{split}_{line_idx}", "text": text,
                        "aspect_term": None, "aspect_start": None, "aspect_end": None,
                        "aspect_category": None, "opinion_term": None,
                        "opinion_start": None, "opinion_end": None, "sentiment": None,
                    })
                else:
                    for q in quads:
                        a_span, cat, sent_code, o_span = q.split(" ")
                        a_s, a_e = map(int, a_span.split(","))
                        o_s, o_e = map(int, o_span.split(","))
                        records.append({
                            "dataset": dataset, "domain": domain, "split": split,
                            "source_file": os.path.basename(rel_path),
                            "sentence_id": f"{domain}_{split}_{line_idx}", "text": text,
                            "aspect_term": " ".join(tokens[a_s:a_e]) if a_s != -1 else "NULL",
                            "aspect_start": a_s, "aspect_end": a_e,
                            "aspect_category": cat,
                            "opinion_term": " ".join(tokens[o_s:o_e]) if o_s != -1 else "NULL",
                            "opinion_start": o_s, "opinion_end": o_e,
                            "sentiment": SENTIMENT_MAP.get(sent_code, sent_code),
                        })

        for dataset, domain, split, rel_path in semeval_files:
            root = ET.fromstring(z.read(rel_path))
            for sent in root.findall("sentence"):
                sid = sent.get("id")
                text = sent.findtext("text") or ""
                terms = sent.findall("./aspectTerms/aspectTerm")
                cats = sent.findall("./aspectCategories/aspectCategory")

                if not terms and not cats:
                    records.append({
                        "dataset": dataset, "domain": domain, "split": split,
                        "source_file": os.path.basename(rel_path),
                        "sentence_id": sid, "text": text,
                        "aspect_term": None, "aspect_start": None, "aspect_end": None,
                        "aspect_category": None, "opinion_term": None,
                        "opinion_start": None, "opinion_end": None, "sentiment": None,
                    })
                for t in terms:
                    records.append({
                        "dataset": dataset, "domain": domain, "split": split,
                        "source_file": os.path.basename(rel_path),
                        "sentence_id": sid, "text": text,
                        "aspect_term": t.get("term"),
                        "aspect_start": int(t.get("from")),
                        "aspect_end": int(t.get("to")),
                        "aspect_category": None, "opinion_term": None,
                        "opinion_start": None, "opinion_end": None,
                        "sentiment": t.get("polarity"),
                    })
                for c in cats:
                    records.append({
                        "dataset": dataset, "domain": domain, "split": split,
                        "source_file": os.path.basename(rel_path),
                        "sentence_id": sid, "text": text,
                        "aspect_term": None, "aspect_start": None, "aspect_end": None,
                        "aspect_category": c.get("category"),
                        "opinion_term": None, "opinion_start": None, "opinion_end": None,
                        "sentiment": c.get("polarity"),
                    })

    df = pd.DataFrame(records)
    for col in ["aspect_start", "aspect_end", "opinion_start", "opinion_end"]:
        df[col] = df[col].astype("Int64")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[Extract] Exported {len(df)} unified records to '{output_csv}'.")
    return df


if __name__ == "__main__":
    cfg = ABSAConfig()
    extract_zip_to_unified_csv(cfg.zip_path, cfg.csv_path)
