import argparse
import json
from src.config import ABSAConfig
from src.extract_data import extract_zip_to_unified_csv
from src.data_pipeline import init_sqlite_store
from src.baselines import run_tfidf_svm_baseline
from src.train_ate import train_ate
from src.train_apc import train_apc
from src.inference_explain import ABSAEngine


def main():
    parser = argparse.ArgumentParser(description="ABSA DeBERTa-v3 VS Code Pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "extract", "db", "baseline", "ate", "apc", "demo"],
        default="all",
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The battery life is exceptional, but the build quality feels cheap and fragile.",
        help="Custom review text for --step demo",
    )
    args = parser.parse_args()
    cfg = ABSAConfig()

    if args.step == "extract":
        extract_zip_to_unified_csv(cfg.zip_path, cfg.csv_path)

    if args.step in ("all", "db"):
        init_sqlite_store(cfg.csv_path, cfg.sqlite_db_path)

    if args.step in ("all", "baseline"):
        run_tfidf_svm_baseline(cfg)

    if args.step in ("all", "ate"):
        train_ate(cfg)

    if args.step in ("all", "apc"):
        train_apc(cfg)

    if args.step in ("all", "demo"):
        engine = ABSAEngine(cfg)
        result = engine.analyze_review(args.text)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
