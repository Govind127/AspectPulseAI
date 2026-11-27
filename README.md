# Project ABSA — VS Code Local Workspace

## 1. Setup Environment (VS Code Terminal)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Install CUDA-matched PyTorch (example for CUDA 12.6):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```
*(On Linux/macOS, activate with `source .venv/bin/activate` and update `.vscode/settings.json` interpreter path to `${workspaceFolder}/.venv/bin/python`.)*

## 2. Run Pipeline Stages
`data/unified_absa_dataset.csv` is already included in `data/`. Run stages via CLI or VS Code Run & Debug (`F5`):
```bash
python main.py --step db        # Load CSV into SQLite (data/absa_store.db)
python main.py --step baseline  # Run TF-IDF + LinearSVC benchmark
python main.py --step ate       # Train Phase 1 DeBERTa-v3 BIO Tagger
python main.py --step apc       # Train Phase 2 DeBERTa-v3 Aspect Polarity Classifier
python main.py --step demo --text "Fast charging speed and crisp display, though the trackpad rattles."
```
