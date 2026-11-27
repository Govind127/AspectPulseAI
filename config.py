import os
from pathlib import Path
from dataclasses import dataclass
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class ABSAConfig:
    # Paths resolved relative to Project_ABSA root
    zip_path: str = str(BASE_DIR / "data" / "Project ABSA.zip")
    csv_path: str = str(BASE_DIR / "data" / "unified_absa_dataset.csv")
    sqlite_db_path: str = str(BASE_DIR / "data" / "absa_store.db")
    output_dir: str = str(BASE_DIR / "outputs")

    # Stable, fast bidirectional encoder (immune to DeBERTa-v3 SDPA NaN overflow)
    # You can also test 'bert-base-uncased' here for your baseline comparison table
    ate_model_name: str = "roberta-base"
    apc_model_name: str = "roberta-base"

    # Proven stable hyperparameters
    max_length: int = 128
    ate_batch_size: int = 16
    apc_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    ate_epochs: int = 4
    apc_epochs: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    use_fp16: bool = torch.cuda.is_available()

    # Dataset filters
    ate_dataset_filter: str = "SemEval2014Task4"
    apc_dataset_filter: str | None = None

ATE_LABELS = ["O", "B-ASP", "I-ASP"]
ATE_LABEL2ID = {label: idx for idx, label in enumerate(ATE_LABELS)}
ATE_ID2LABEL = {idx: label for idx, label in enumerate(ATE_LABELS)}

APC_LABELS = ["negative", "neutral", "positive"]
APC_LABEL2ID = {label: idx for idx, label in enumerate(APC_LABELS)}
APC_ID2LABEL = {idx: label for idx, label in enumerate(APC_LABELS)}

os.makedirs(BASE_DIR / "outputs", exist_ok=True)