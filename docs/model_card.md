# Model card — XACT-TB reconstruction

* **Status**: no trained weights released; none evaluated on real data.
* **Intended use**: research on binary TB-vs-Normal chest-X-ray classification and explainability. Not for clinical use.
* **Architecture**: CNN (256-D) ‖ Swin (768-D) → StandardScaler → SVM (frozen features).
* **Output**: class label and SVM decision margin (not a probability; no calibration performed).
* **Known limits**: binary task only (other pathologies unseen); image-level splits (patient leakage possible); single public data source; explanations are qualitative and branch-specific (Grad-CAM explains the CNN auxiliary head, not the SVM).
* **Ethics/safety**: possible dataset/source bias; do not deploy without prospective, patient-level, multi-site validation and regulatory review.
