import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from src.config import ABSAConfig, APC_LABEL2ID, APC_LABELS


def run_tfidf_svm_baseline(cfg: ABSAConfig):
    df = pd.read_csv(cfg.csv_path)
    if cfg.apc_dataset_filter:
        df = df[df["dataset"] == cfg.apc_dataset_filter].copy()

    apc_df = df[
        df["aspect_term"].notna()
        & (df["aspect_term"] != "NULL")
        & (df["sentiment"].isin(APC_LABEL2ID.keys()))
    ].copy()

    apc_df["combined_input"] = apc_df["aspect_term"].astype(str) + " [SEP] " + apc_df["text"].astype(str)
    apc_df["label"] = apc_df["sentiment"].map(APC_LABEL2ID)

    train_df = apc_df[apc_df["split"] == "train"]
    test_df = apc_df[apc_df["split"] == "test"]

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)),
        ("svm", LinearSVC(class_weight="balanced", random_state=cfg.seed))
    ])

    clf.fit(train_df["combined_input"], train_df["label"])
    preds = clf.predict(test_df["combined_input"])

    macro_f1 = f1_score(test_df["label"], preds, average="macro")
    print("=" * 60)
    print(f"BASELINE: TF-IDF (1-2 ngrams) + LinearSVC | Test Macro-F1: {macro_f1:.4f}")
    print("=" * 60)
    print(classification_report(test_df["label"], preds, target_names=APC_LABELS, digits=4))
    return macro_f1


if __name__ == "__main__":
    run_tfidf_svm_baseline(ABSAConfig())
