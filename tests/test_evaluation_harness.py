"""Unit tests for Phase 3 evaluation harness: scorers, integrity, stats, bootstrap, and stub eval."""

from kvpipe.config import ArmConfig, SuiteConfig
from kvpipe.data.loaders import load_suite_data
from kvpipe.engines.stub import StubEngine
from kvpipe.run.integrity import check_integrity, write_review_file
from kvpipe.run.sampler import evaluate_suite_on_arm
from kvpipe.run.scoring import extract_boxed_answer, score_math, score_mcq, score_retrieval
from kvpipe.stats import evaluate_gate_verdict, paired_bootstrap_ci, power_planning


def test_scorers():
    # Math
    assert extract_boxed_answer("Step by step... \\boxed{42}") == "42"
    assert score_math("Therefore, \\boxed{100}", "100") is True
    assert score_math("Answer: \\boxed{\\frac{1}{2}}", "0.5") is True
    assert score_math("The answer is 55", "55") is True
    assert score_math("No answer here", "42") is False

    # MCQ
    assert score_mcq("The correct answer is (B).", "B") is True
    assert score_mcq("Option C is clearly correct.", "C") is True
    assert score_mcq("I pick (A)", "D") is False

    # Retrieval
    assert score_retrieval("The secret password is ABC_123.", "ABC_123") is True
    assert score_retrieval("No secret here", "ABC_123") is False


def test_integrity_checks_corruptions():
    # Clean output
    clean = check_integrity("<think>Clean thought</think> The result is \\boxed{42}.", finish_reason="stop", task_type="math")
    assert clean["passed"] is True
    assert clean["is_empty"] is False
    assert clean["corrupt_chars"] is False

    # Empty output
    empty = check_integrity("", finish_reason="stop", task_type="math")
    assert empty["passed"] is False
    assert "empty_output" in empty["failure_reasons"]

    # Corrupt replacement character
    corrupt_rep = check_integrity("The answer is \ufffd corrupted.", finish_reason="stop", task_type="math")
    assert corrupt_rep["passed"] is False
    assert "corrupt_characters" in corrupt_rep["failure_reasons"]

    # Runs of question marks
    corrupt_q = check_integrity("The answer is ???????? wrong.", finish_reason="stop", task_type="math")
    assert corrupt_q["passed"] is False
    assert "corrupt_characters" in corrupt_q["failure_reasons"]

    # Severe repetition
    repetitive = check_integrity(("repeat this phrase again " * 50) + " \\boxed{42}", finish_reason="stop", task_type="math")
    assert repetitive["passed"] is False
    assert any("repetition" in r for r in repetitive["failure_reasons"])

    # Missing boxed answer
    no_box = check_integrity("Thought without any boxed answer.", finish_reason="stop", task_type="math")
    assert no_box["passed"] is False
    assert "no_boxed_answer" in no_box["failure_reasons"]


def test_paired_bootstrap_and_verdicts():
    # Pass scenario: candidate and reference have matching high performance
    cand_records = [{"item_id": f"p{i}", "correct": True} for i in range(20)]
    ref_records = [{"item_id": f"p{i}", "correct": True} for i in range(20)]

    ci = paired_bootstrap_ci(cand_records, ref_records, num_resamples=1000)
    verdict = evaluate_gate_verdict(ci["ci_lower"], ci["ci_upper"], margin=2.0)
    assert verdict == "PASS"

    # Fail scenario: large degradation
    cand_fail = [{"item_id": f"p{i}", "correct": False} for i in range(20)]
    ref_pass = [{"item_id": f"p{i}", "correct": True} for i in range(20)]
    ci_fail = paired_bootstrap_ci(cand_fail, ref_pass, num_resamples=1000)
    verdict_fail = evaluate_gate_verdict(ci_fail["ci_lower"], ci_fail["ci_upper"], margin=2.0)
    assert verdict_fail == "FAIL"

    # Inconclusive scenario: CI spans -margin (-2.0)
    assert evaluate_gate_verdict(ci_lower=-3.5, ci_upper=1.0, margin=2.0) == "INCONCLUSIVE"


def test_power_planning():
    plan = power_planning(delta_std=4.0, target_margin=2.0, max_budget_samples=64)
    assert plan["recommended_samples_per_problem"] >= 8
    assert "budget_sufficient" in plan


def test_stub_evaluation_and_review_file(tmp_path):
    stub = StubEngine()
    arm_a0 = ArmConfig(id="A0", engine="stub", model="Qwen/Qwen3-8B")
    suite_aime = SuiteConfig(
        id="aime25",
        name="AIME 2025",
        dataset_id="test",
        sample_count=10,
        samples_per_problem=2,
    )

    handle = stub.launch(arm_a0, port=8985)
    try:
        items = load_suite_data(suite_aime)
        results = evaluate_suite_on_arm(
            arm=arm_a0,
            suite=suite_aime,
            base_url=handle.base_url,
            items=items[:10],
            runs_dir=tmp_path / "runs",
        )
        assert len(results) == 20
        # Write review file
        rev_path = write_review_file(
            arm_id="A0",
            samples=results,
            output_dir=tmp_path / "review",
            num_samples=5,
        )
        assert rev_path.exists()
        content = rev_path.read_text(encoding="utf-8")
        assert "# Review Generations for Arm: A0" in content
    finally:
        handle.stop()
