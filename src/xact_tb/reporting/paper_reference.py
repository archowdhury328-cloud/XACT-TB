"""Values transcribed from the paper (Tables 1, 3, 4, 5). REFERENCE ONLY: never targets, never
mixed into measured-results columns. Source: Sikder et al., Discover AI 6:1030 (2026)."""

TEST_PER_CLASS = 525  # Table 1: 525 TB + 525 Normal test images (1,050 total)

# Table 3 (percent). AUC columns reported as 95.0/97.0/99.0/100.
PAPER_TABLE3 = {
    "Swin Transformer": {
        "roc_auc": 95.0,
        "accuracy": 92.54,
        "sensitivity": 91.23,
        "specificity": 93.67,
        "precision": 90.26,
        "f1": 90.67,
    },
    "CoAtNet-3": {
        "roc_auc": 97.0,
        "accuracy": 97.89,
        "sensitivity": 97.30,
        "specificity": 97.40,
        "precision": 96.77,
        "f1": 97.14,
    },
    "MaxViT": {
        "roc_auc": 99.0,
        "accuracy": 98.78,
        "sensitivity": 98.12,
        "specificity": 98.19,
        "precision": 97.77,
        "f1": 97.79,
    },
    "XACT-TB (proposed)": {
        "roc_auc": 100.0,
        "accuracy": 99.44,
        "sensitivity": 99.41,
        "specificity": 99.53,
        "precision": 99.37,
        "f1": 99.38,
    },
}

# Table 4: AUC (95% CI), p-value vs proposed
PAPER_TABLE4 = {
    "Swin Transformer": (0.950, 0.875, 1.000, 0.208),
    "CoAtNet-3": (0.970, 0.912, 1.000, 0.341),
    "MaxViT": (0.990, 0.956, 1.000, 0.621),
    "XACT-TB (proposed)": (1.000, 0.988, 1.000, None),
}

# Table 5 (published; RTX 3060 FP32): params M, GFLOPs, size MB, latency ms/img
PAPER_TABLE5 = {
    "Swin Transformer": (50, 8.7, 200, 0.867),
    "CoAtNet-3": (168, 34.7, 672, 0.189),
    "MaxViT": (31, 5.6, 124, 0.453),
    "XACT-TB (proposed)": (75, 7.8, 245, 0.358),
}
