"""
Regression tests for the evaluation harness itself — the actual scoring
logic that every ablation table and headline number in this project depends
on. These are the same checks that were run ad-hoc against known values
during development; codified here so a future change to any metric function
can't silently break correctness without a test failing.

Run: pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_eval.eval.retrieval_metrics import recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k
from rag_eval.eval.generation_metrics import exact_match, answer_contained, f1_overlap, has_citation, is_abstention
from rag_eval.eval.calibration import cohens_kappa
from rag_eval.corpus_b.numeric_metrics import extract_numbers, is_correct


def test_recall_at_k_full_hit():
    assert recall_at_k(["a", "b", "c", "d", "e"], ["b", "e"]) == 1.0


def test_recall_at_k_zero_hit():
    assert recall_at_k(["x", "y", "z"], ["b", "e"]) == 0.0


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c", "d", "e"], ["b", "e"]) == 0.4


def test_reciprocal_rank_first_hit_at_rank_2():
    assert reciprocal_rank(["a", "b", "c", "d", "e"], ["b", "e"]) == 0.5


def test_reciprocal_rank_zero_when_not_found():
    assert reciprocal_rank(["x", "y", "z"], ["b", "e"]) == 0.0


def test_ndcg_perfect_order_is_1():
    assert ndcg_at_k(["b", "e", "x", "y", "z"], ["b", "e"]) == 1.0


def test_ndcg_imperfect_order_below_1():
    score = ndcg_at_k(["a", "b", "c", "d", "e"], ["b", "e"])
    assert 0 < score < 1.0


def test_exact_match_true():
    assert exact_match("Prussian", "Prussian") == 1.0


def test_exact_match_false():
    assert exact_match("Prussian", "German") == 0.0


def test_answer_contained_finds_short_gold_in_long_sentence():
    assert answer_contained("U2", "U2 released it first as the lead single [1].") == 1.0


def test_answer_contained_false_when_absent():
    assert answer_contained("Prussian", "Oliver Reed played Otto von Bismarck [1].") == 0.0


def test_f1_overlap_partial():
    score = f1_overlap("Kurt Julian Weill", "Kurt Weill")
    assert 0 < score < 1.0


def test_has_citation_bracket_style():
    assert has_citation("The answer is X [1].") is True


def test_has_citation_fullwidth_style():
    assert has_citation("The answer is X\u30101\u3011.") is True


def test_has_citation_false_when_absent():
    assert has_citation("The answer is X.") is False


def test_is_abstention_true():
    assert is_abstention("I don't know based on the given information.") is True


def test_is_abstention_false():
    assert is_abstention("The answer is Paris.") is False


def test_cohens_kappa_perfect_agreement():
    assert cohens_kappa(["yes", "no", "yes"], ["yes", "no", "yes"]) == 1.0


def test_cohens_kappa_partial_agreement_in_reasonable_range():
    human = ["yes", "yes", "yes", "no", "no", "no", "yes", "yes", "no", "no"]
    judge = ["yes", "yes", "no", "no", "no", "yes", "yes", "yes", "no", "no"]
    kappa = cohens_kappa(human, judge)
    assert 0.3 < kappa < 0.9


def test_extract_numbers_plain_commas():
    assert extract_numbers("Total assets were 216,369,000 dollars.") == [216369000.0]


def test_extract_numbers_million_suffix():
    numbers = extract_numbers("Total assets were approximately $216.4 million.")
    assert numbers == [216400000.0]


def test_extract_numbers_accounting_negative():
    assert extract_numbers("Net loss was (50,000) for the year.") == [-50000.0]


def test_is_correct_within_tolerance():
    assert is_correct(216369000, "Total assets were approximately $216.4 million.") is True


def test_is_correct_wrong_value():
    assert is_correct(216369000, "Total assets were $100 million.") is False


def test_is_correct_exact_negative_match():
    assert is_correct(-50000, "The net loss was (50,000).") is True
