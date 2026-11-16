# Data statement
* Dataset: Kaggle "Tuberculosis (TB) Chest X-ray Database" (Rahman et al.; Qatar Univ., Univ. Dhaka, Hamad Medical Corp.; sources include NLM Montgomery/Shenzhen, Belarus, NIAID TB portal, RSNA). Respect its licence/terms and cite the original authors.
* Download date, archive version and file checksums are recorded in `manifests/private/audit_summary.json` / `manifest.csv` by `xact-tb audit-data` (not yet run: no credentials in this environment).
* Labels: Normal = 0, TB = 1 from folder names. No patient IDs; source tags only when file names carry them.
* Corrupt files and label-conflicting duplicate groups are excluded and listed in `exclusions.csv`.
* Images and per-image records are never committed to this repository.
