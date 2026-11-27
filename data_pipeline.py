import sqlite3
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from src.config import ABSAConfig, ATE_LABEL2ID, APC_LABEL2ID


def init_sqlite_store(csv_path: str, db_path: str) -> pd.DataFrame:
    """Loads unified CSV into SQLite for structured querying and persistence."""
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)

    df.to_sql("annotations", conn, if_exists="replace", index=False)

    sentences_df = (
        df[["dataset", "domain", "split", "sentence_id", "text"]]
        .drop_duplicates(subset=["dataset", "domain", "split", "sentence_id"])
        .reset_index(drop=True)
    )
    sentences_df.to_sql("raw_sentences", conn, if_exists="replace", index=False)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS inferred_aspects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id TEXT,
            review_text TEXT,
            aspect_term TEXT,
            polarity TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print(f"[SQLite] Synced {len(df)} annotation rows and {len(sentences_df)} unique sentences to '{db_path}'.")
    return df


def build_ate_datasets(cfg: ABSAConfig) -> DatasetDict:
    """Prepares sentence-level BIO-tagged datasets for Phase 1 (Aspect Term Extraction)."""
    df = pd.read_csv(cfg.csv_path)
    if cfg.ate_dataset_filter:
        df = df[df["dataset"] == cfg.ate_dataset_filter].copy()

    valid_spans = df[df["aspect_term"].notna() & (df["aspect_start"] >= 0)].copy()

    grouped_spans = {}
    for _, row in valid_spans.iterrows():
        key = (row["dataset"], row["domain"], row["split"], str(row["sentence_id"]), str(row["text"]))
        grouped_spans.setdefault(key, []).append((int(row["aspect_start"]), int(row["aspect_end"])))

    all_sentences = df[["dataset", "domain", "split", "sentence_id", "text"]].drop_duplicates()
    sentence_records = []
    for _, row in all_sentences.iterrows():
        key = (row["dataset"], row["domain"], row["split"], str(row["sentence_id"]), str(row["text"]))
        sentence_records.append({
            "split": "dev" if row["split"] in ["dev", "trial"] else row["split"],
            "text": str(row["text"]),
            "spans": grouped_spans.get(key, [])
        })

    sent_df = pd.DataFrame(sentence_records)
    tokenizer = AutoTokenizer.from_pretrained(cfg.ate_model_name, use_fast=True)

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_length,
            return_offsets_mapping=True,
        )
        all_labels = []
        for idx, offsets in enumerate(tokenized["offset_mapping"]):
            spans = examples["spans"][idx]
            labels = []
            for start_char, end_char in offsets:
                if start_char == 0 and end_char == 0:
                    labels.append(-100)
                    continue

                tag = "O"
                for asp_start, asp_end in spans:
                    if start_char >= asp_start and end_char <= asp_end:
                        tag = "B-ASP" if start_char == asp_start else "I-ASP"
                        break
                    elif start_char < asp_end and end_char > asp_start:
                        tag = "B-ASP" if start_char <= asp_start else "I-ASP"
                        break
                labels.append(ATE_LABEL2ID[tag])
            all_labels.append(labels)

        tokenized["labels"] = all_labels
        tokenized.pop("offset_mapping")
        return tokenized

    ds_dict = DatasetDict()
    for split_name in ["train", "dev", "test"]:
        sub = sent_df[sent_df["split"] == split_name].reset_index(drop=True)
        if len(sub) > 0:
            raw_ds = Dataset.from_pandas(sub[["text", "spans"]])
            ds_dict[split_name] = raw_ds.map(tokenize_and_align, batched=True, remove_columns=["text", "spans"])

    return ds_dict


def build_apc_datasets(cfg: ABSAConfig) -> DatasetDict:
    """Prepares cross-sentence pair inputs: [CLS] Review_Text [SEP] Aspect_Term [SEP] for Phase 2."""
    df = pd.read_csv(cfg.csv_path)
    if cfg.apc_dataset_filter:
        df = df[df["dataset"] == cfg.apc_dataset_filter].copy()

    apc_df = df[
        df["aspect_term"].notna()
        & (df["aspect_term"] != "NULL")
        & (df["sentiment"].isin(APC_LABEL2ID.keys()))
    ].copy()

    apc_df["split_norm"] = apc_df["split"].replace({"trial": "dev"})
    apc_df["label"] = apc_df["sentiment"].map(APC_LABEL2ID)

    tokenizer = AutoTokenizer.from_pretrained(cfg.apc_model_name, use_fast=True)

    def tokenize_pairs(examples):
        return tokenizer(
            examples["text"],
            examples["aspect_term"],
            truncation=True,
            max_length=cfg.max_length,
        )

    ds_dict = DatasetDict()
    for split_name in ["train", "dev", "test"]:
        sub = apc_df[apc_df["split_norm"] == split_name][["text", "aspect_term", "label"]].reset_index(drop=True)
        if len(sub) > 0:
            raw_ds = Dataset.from_pandas(sub)
            ds_dict[split_name] = raw_ds.map(tokenize_pairs, batched=True, remove_columns=["text", "aspect_term"])

    return ds_dict
