# Security
Report vulnerabilities privately to the maintainer via GitHub security advisories.
* Bundles/checkpoints use torch/joblib (pickle-capable): load only bundles you built or whose SHA-256 you verified.
* Torch weights load with `weights_only=True`; image paths are confined to the data root; no implicit downloads at inference.
* Secrets belong in environment variables or a credential manager; a token ever pasted into chat/logs must be revoked.
