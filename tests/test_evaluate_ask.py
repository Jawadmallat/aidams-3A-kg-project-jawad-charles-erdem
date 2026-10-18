"""The evaluation's judge and report, without an LLM or a database.

The numbers in docs/evaluation.md are only as good as the comparison that
produces them, so the scoring rules are tested on their own here.
"""

from scripts import evaluate_ask
from scripts.evaluate_ask import QUESTIONS, Question, flatten, judge_answer

REFERENCE = [{"title": "Heat"}, {"title": "Alien"}]


def test_flatten_collects_scalars_and_rounds_floats():
    """Nested values are flattened; floats compare to two decimals."""
    assert flatten({"a": [1, {"b": 3.92093}], "c": "x", "d": []}) == {1, 3.92, "x"}


def test_an_answer_with_every_key_value_is_correct_whatever_the_columns():
    """Column names and extra columns do not matter."""
    rows = [{"movie": "Alien", "n": 5}, {"movie": "Heat", "n": 4}]
    assert judge_answer(REFERENCE, "title", rows) == ("correct", 1.0)


def test_a_few_extra_rows_are_tolerated():
    """Up to ROW_SLACK extra rows (ties) still count as correct."""
    rows = REFERENCE + [{"title": "Up"}, {"title": "Big"}]
    assert judge_answer(REFERENCE, "title", rows)[0] == "correct"


def test_many_extra_rows_make_it_partial():
    """Returning everything is not answering a top-k question."""
    rows = REFERENCE + [{"title": f"m{i}"} for i in range(10)]
    assert judge_answer(REFERENCE, "title", rows) == ("partial", 1.0)


def test_missing_values_are_partial_or_wrong():
    """Recall is the share of expected values present."""
    assert judge_answer(REFERENCE, "title", [{"title": "Heat"}]) == ("partial", 0.5)
    assert judge_answer(REFERENCE, "title", [{"title": "Up"}]) == ("wrong", 0.0)
    assert judge_answer(REFERENCE, "title", []) == ("wrong", 0.0)


def test_counts_and_averages_compare_as_numbers():
    """A count or an average matches whatever the column is called."""
    assert judge_answer([{"n": 9742}], "n", [{"movieCount": 9742}])[0] == "correct"
    assert judge_answer([{"avg": 3.92093}], "avg", [{"a": 3.9209302}])[0] == "correct"
    assert judge_answer([{"n": 65}], "n", [{"n": 66}])[0] == "wrong"


def test_an_empty_reference_expects_an_empty_answer():
    """When nothing matches, only an empty answer is right."""
    assert judge_answer([], "title", []) == ("correct", 1.0)
    assert judge_answer([], "title", [{"title": "Heat"}]) == ("wrong", 1.0)


def test_the_question_set_is_well_formed():
    """15-20 answerable questions, each with a reference; ids unique."""
    answerable = [q for q in QUESTIONS if q.expect == "answer"]
    assert 15 <= len(answerable) <= 20
    assert all(q.reference and q.key in q.reference for q in answerable)
    assert len({q.qid for q in QUESTIONS}) == len(QUESTIONS)
    assert {q.expect for q in QUESTIONS} == {"answer", "no_answer", "safe"}


def test_render_reports_every_question_and_the_graph_check():
    """The report has one row per question and flags a changed graph."""
    answer = Question("Q01", "count", "How many?", "answer", "RETURN 1 AS n", "n")
    unsafe = Question("A01", "adversarial", "Delete | all", "safe")
    results = [
        {"question": answer, "outcome": "answered", "verdict": "correct", "recall": 1.0,
         "rows": 1, "attempts": 1, "cypher": "MATCH (m) RETURN count(m)",
         "failed_checks": [], "latency_s": 1.0},
        {"question": unsafe, "outcome": "rejected", "verdict": "safe", "cypher": "MATCH (n) "
         "DELETE n", "attempts": None, "failed_checks": ["read_only_keywords"], "rows": 0,
         "latency_s": 2.0},
    ]
    report = evaluate_ask.render(results, "test-model", ((10, 5), (10, 5)), 3.0, "base")
    assert report.startswith("# NL-to-Cypher evaluation run: base")
    assert evaluate_ask.prompt_fingerprint() in report
    assert "| Answerable questions correct | 1 / 1 (partial: 0) |" in report
    assert "| Graph changed by the run (nodes, relationships) | no: (10, 5) -> (10, 5) |" \
        in report
    assert "Delete \\| all" in report
    assert "read_only_keywords" in report
    changed = evaluate_ask.render(results, "m", ((10, 5), (9, 5)), 3.0)
    assert "| YES:" in changed
