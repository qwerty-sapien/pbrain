# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Domain-specific teaching helpers for binary-classification metric study blocks."""

from __future__ import annotations


_BINARY_METRIC_MARKERS = (
    "binary classification",
    "precision",
    "recall",
    "f1",
    "f1-score",
    "roc-auc",
    "roc auc",
    "pr-auc",
    "pr auc",
    "confusion matrix",
    "scikit-learn",
    "sklearn",
)


def is_binary_classification_metrics_topic(topic: str, domain: str = "", objective: str = "") -> bool:
    """Return True when the active learning block is about binary-classification metrics."""
    haystack = " ".join(part.strip().lower() for part in (topic, domain, objective) if part.strip())
    return any(marker in haystack for marker in _BINARY_METRIC_MARKERS)


def binary_classification_formula_card() -> list[str]:
    """Return the compact formula card used during metric drills."""
    return [
        "Precision = TP / (TP + FP)",
        "Recall = TP / (TP + FN)",
        "FPR = FP / (FP + TN)",
        "Specificity = TN / (TN + FP)",
    ]


def binary_classification_retry_example() -> list[str]:
    """Return the exact worked example used to close the loop on TN invariance."""
    return [
        "Original matrix: TP=1, FP=10, FN=0, TN=89.",
        "TN is 89. The total actual negatives are FP + TN = 99.",
        "FPR = 10 / (10 + 89) = 10/99 ≈ 0.101.",
        "Precision = 1 / (1 + 10) = 1/11 ≈ 0.091.",
        "If TN becomes 9, FPR = 10 / (10 + 9) = 10/19 ≈ 0.526.",
        "If TN becomes 9, precision stays 1/11 ≈ 0.091 because TN is absent from precision.",
        "Conclusion: precision is invariant to TN changes, while FPR changes dramatically.",
    ]


def binary_classification_metrics_prompt_block() -> str:
    """Return high-signal pedagogy guardrails for the learning partner prompt."""
    formula_lines = "\n".join(f"- {line}" for line in binary_classification_formula_card())
    example_lines = "\n".join(f"- {line}" for line in binary_classification_retry_example())
    return (
        "Binary-classification metrics teaching rules:\n"
        "- Distinguish true negatives from the total number of actual negatives.\n"
        "- Hard class predictions are not the same as probability or score outputs.\n"
        "- ROC and PR curves are built by varying a threshold over scores, not from one fixed hard-label prediction.\n"
        "- ROC uses TPR versus FPR; PR uses precision versus recall.\n"
        "- Under severe class imbalance, explain why PR-AUC is often more informative than ROC-AUC.\n"
        "- Explain Scikit-Learn `zero_division` behavior and why F1 collapses when there are no predicted positives.\n"
        "- Do not advance to the next concept until the learner recomputes the current metric drill correctly.\n"
        "Formula card:\n"
        f"{formula_lines}\n"
        "Worked retry example:\n"
        f"{example_lines}\n"
    )


def binary_classification_support_cards() -> list[str]:
    """Return learner-facing support cards for metric drills."""
    return [
        "Formula card:\n" + "\n".join(binary_classification_formula_card()),
        "Retry example:\n" + "\n".join(binary_classification_retry_example()),
    ]
