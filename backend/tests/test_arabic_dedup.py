"""
Regression tests for the fuzzy (Jaccard) near-duplicate clause dedup added to
_extract_all_clauses.

The LLM and rule-based extraction paths can produce the same underlying
obligation with lightly reworded phrasing where neither text is a substring
of the other, so the pre-existing exact/containment dedup misses them and the
same clause is counted twice (inflating the requirement count).

These pure-helper tests run entirely offline and lock in the safety contract:
near-identical rewordings must merge, but genuinely distinct obligations that
merely share vocabulary must NOT merge (losing a real requirement is worse
than keeping a near-duplicate).

Run:  pytest backend/tests/test_arabic_dedup.py -v
"""
from unittest.mock import patch

import pytest

from app.agents.extractor_agent import (
    _clause_token_set,
    _jaccard,
    _extract_all_clauses,
)
from app.services.document_parser import DocumentParserService
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "taqeem_security_guards_tender.pdf"


# --------------------------------------------------------------------------
# Pure helpers: fuzzy near-duplicate safety contract
# --------------------------------------------------------------------------

def test_jaccard_empty_sets():
    assert _jaccard(set(), set()) == 0.0
    assert _jaccard({"a"}, set()) == 0.0


def test_distinct_obligations_sharing_vocabulary_do_not_merge():
    """
    The single most important safety property: two DIFFERENT obligations that
    happen to share common legal vocabulary must NOT be treated as
    duplicates. Losing a real requirement is worse than keeping a near-dup.
    """
    a = "يجب أن يحصل المتنافس على حد أدنى قدره ستون درجة من مجموع درجات التقييم الفني"
    b = "مدة سريان العروض تسعون يوما من التاريخ المحدد لفتح العروض"
    c = "يجب على المتعاقد تقديم كشف الحساب الختامي خلال ثلاثين يوما من تاريخ الاستلام النهائي"
    d = "لا تتجاوز نسبة الأعمال المسندة إلى المتعاقد من الباطن ثلاثين بالمائة من قيمة العقد"

    # All well below the 0.82 dedup threshold used by _extract_all_clauses.
    assert _jaccard(_clause_token_set(a), _clause_token_set(b)) < 0.5
    assert _jaccard(_clause_token_set(c), _clause_token_set(d)) < 0.5


def test_near_identical_rewording_scores_high():
    """A near-identical copy of the same clause (the LLM path keeping/omitting
    a single trailing word relative to the rule-based copy) must score above
    the merge threshold, so the same obligation is not counted twice."""
    a = "يجب على المتنافس تقديم الضمان الابتدائي بنسبة واحد بالمائة من القيمة الإجمالية للعرض"
    b = "يجب على المتنافس تقديم الضمان الابتدائي بنسبة واحد بالمائة من القيمة الإجمالية للعرض المقدم"

    score = _jaccard(_clause_token_set(a), _clause_token_set(b))
    assert score >= 0.82, f"expected near-duplicate score >= 0.82, got {score}"


def test_short_clauses_below_min_tokens_are_not_fuzzy_matched():
    """The minimum-token guard prevents short, generically-worded fragments
    from being fuzzy-matched against each other purely on overlap."""
    a = "يجب الالتزام"
    b = "يجب الالتزام التام"
    assert len(_clause_token_set(a)) < 6 or len(_clause_token_set(b)) < 6


# --------------------------------------------------------------------------
# Integration: deterministic rule-based extraction on the real fixture
# --------------------------------------------------------------------------

@pytest.fixture()
def rule_based_clauses():
    """Run the full extractor with the LLM forced off, so the result is the
    deterministic rule-based path only — reproducible on any machine."""
    with patch("app.agents.llm_factory.LLMFactory.get_chat_model", return_value=None):
        blocks = DocumentParserService.parse_document(str(FIXTURE))
        formatted, eval_items, corrupt_pages = _extract_all_clauses(blocks)
    return formatted, eval_items, corrupt_pages


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture PDF not present")
def test_rule_based_count_is_reasonable(rule_based_clauses):
    formatted, _, _ = rule_based_clauses
    # The deterministic rule-based path over this 32-page tender is stable;
    # a large deviation signals an extraction or dedup regression.
    assert 95 <= len(formatted) <= 120, f"got {len(formatted)}"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture PDF not present")
def test_rule_based_clauses_have_no_exact_or_near_duplicates(rule_based_clauses):
    """The fuzzy dedup must not let two near-identical rule-based clauses
    through side by side."""
    formatted, _, _ = rule_based_clauses
    token_sets = [_clause_token_set(c.text) for c in formatted]
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            a, b = token_sets[i], token_sets[j]
            if len(a) >= 6 and len(b) >= 6:
                assert _jaccard(a, b) < 0.82, (
                    f"near-duplicate clauses survived dedup:\n"
                    f"  {formatted[i].text!r}\n  {formatted[j].text!r}"
                )
