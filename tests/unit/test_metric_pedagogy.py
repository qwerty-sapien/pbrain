# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pb.core.metrics_pedagogy import (
    binary_classification_formula_card,
    binary_classification_metrics_prompt_block,
    binary_classification_retry_example,
    is_binary_classification_metrics_topic,
)


def test_binary_metric_detector_matches_scikit_learn_metric_scope() -> None:
    assert is_binary_classification_metrics_topic(
        "Verify Binary Classification Metrics",
        domain="evaluation of binary classification models using Scikit-Learn metrics",
        objective="Explain precision, recall, F1-score, and ROC-AUC under class imbalance.",
    )


def test_binary_metric_formula_card_has_expected_formulas() -> None:
    formulas = binary_classification_formula_card()

    assert "Precision = TP / (TP + FP)" in formulas
    assert "Recall = TP / (TP + FN)" in formulas
    assert "FPR = FP / (FP + TN)" in formulas
    assert "Specificity = TN / (TN + FP)" in formulas


def test_binary_metric_retry_example_closes_loop_on_tn_invariance() -> None:
    lines = binary_classification_retry_example()

    assert "TN is 89. The total actual negatives are FP + TN = 99." in lines
    assert "FPR = 10 / (10 + 89) = 10/99 ≈ 0.101." in lines
    assert "Precision = 1 / (1 + 10) = 1/11 ≈ 0.091." in lines
    assert "If TN becomes 9, FPR = 10 / (10 + 9) = 10/19 ≈ 0.526." in lines
    assert "If TN becomes 9, precision stays 1/11 ≈ 0.091 because TN is absent from precision." in lines
    assert "Conclusion: precision is invariant to TN changes, while FPR changes dramatically." in lines


def test_binary_metric_prompt_block_mentions_roc_pr_and_zero_division() -> None:
    prompt = binary_classification_metrics_prompt_block()

    assert "ROC and PR curves are built by varying a threshold over scores" in prompt
    assert "PR-AUC is often more informative than ROC-AUC" in prompt
    assert "Scikit-Learn `zero_division` behavior" in prompt
