# Command reference

```bash
pip install -e ".[dev]"                 # CPU torch from PyPI; for GPU install a CUDA wheel first:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# data (Kaggle: accept the dataset terms, put kaggle.json in ~/.kaggle, then download manually or via kaggle CLI)
kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset -p data/raw --unzip
xact-tb audit-data --root data/raw --out manifests/private          # manifest.csv, exclusions.csv, audit_summary.json

# proposed model (config YAML keys: pre, train, swin, run_name; see configs/experiments)
xact-tb train-proposed --manifest manifests/private/manifest.csv --data-root data/raw \
    --out runs/proposed_s42 --config configs/experiments/proposed.yaml --device cuda --no-test   # dev run
xact-tb train-proposed ... (without --no-test for the single locked test evaluation)

# baselines (identical split / preprocessing / budget)
xact-tb train-baseline --baseline swin|coatnet3|maxvit --manifest ... --data-root ... --out runs/swin_s42

# inference (own or hash-verified bundles only)
xact-tb predict --bundle runs/proposed_s42/bundle --input path/to/image_or_dir

xact-tb audit-metrics        # arithmetic audit of the paper's Table 3
xact-tb benchmark [--flops] [--device cuda]
xact-tb smoke                # SYNTHETIC end-to-end software check, meaningless metrics

pytest --cov ; ruff check src tests ; ruff format --check src tests
```
Resume: re-run the same `train-*` command; `last.pt` (model, optimizer, scheduler, history) is picked up automatically.
