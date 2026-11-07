"""Tests for CLI commands (profile-kv, probe-positions, audit) and Kaggle Qwen3.5 script."""

import sys
from pathlib import Path

import torch
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvpipe.cli import main
from scripts.kaggle_qwen35_128k_turboquant import (
    GROUP_SIZE,
    ArmSpec,
    TurboQuantHybridCache,
    check_integrity,
    fwht_last_dim,
    quant_dequant_tensor,
)


def test_cli_profile_kv(tmp_path):
    runner = CliRunner()
    res = runner.invoke(main, ["profile-kv", "--model", "Qwen/Qwen3.5-0.8B", "--output-dir", str(tmp_path)])
    assert res.exit_code == 0
    assert "K/V Norm Profile completed" in res.output
    assert (tmp_path / "Qwen_Qwen3.5-0.8B.json").exists()


def test_cli_probe_positions():
    runner = CliRunner()
    res = runner.invoke(main, ["probe-positions", "--arm", "a0", "--port", "8996"])
    assert res.exit_code == 0
    assert "Position Probes for Arm A0" in res.output


def test_cli_audit_markdown_and_json():
    runner = CliRunner()
    res_json = runner.invoke(main, ["audit", "--report", "artifacts/report.json"])
    assert res_json.exit_code == 0
    assert "AUDIT PASSED" in res_json.output

    res_md = runner.invoke(main, ["audit", "--report", "artifacts/report.md"])
    assert res_md.exit_code == 0
    assert "AUDIT PASSED" in res_md.output


def test_fwht_orthogonality():
    # Fast Walsh-Hadamard Transform must be an involution: fwht(fwht(x)) == x
    x = torch.randn(2, 4, 16, 256)
    rotated = fwht_last_dim(x)
    recovered = fwht_last_dim(rotated)
    assert torch.allclose(x, recovered, atol=1e-5)


def test_quant_dequant_turboquant():
    x = torch.randn(2, 4, 128, 256)
    # 8-bit quantization
    deq8, bytes8 = quant_dequant_tensor(x, bits=8, use_hadamard=True, group_size=GROUP_SIZE)
    assert deq8.shape == x.shape
    assert bytes8 < x.numel() * 2  # Less than FP16
    assert torch.allclose(x, deq8, atol=0.2)

    # 4-bit quantization
    deq4, bytes4 = quant_dequant_tensor(x, bits=4, use_hadamard=True, group_size=GROUP_SIZE)
    assert bytes4 < bytes8


def test_kaggle_integrity_checker():
    # Normal text
    ok, reasons = check_integrity("This is a clean logical response with clear steps.")
    assert ok is True
    assert len(reasons) == 0

    # Corrupted text
    corrupt1, r1 = check_integrity("Something bad happened \ufffd here")
    assert corrupt1 is False
    assert "utf8_replacement_char" in r1

    corrupt2, r2 = check_integrity("Why did this occur ???? Unknown")
    assert corrupt2 is False
    assert "question_mark_run" in r2

    # Loop repetition
    loop_text = "repeat this pattern again " * 10
    corrupt3, r3 = check_integrity(loop_text)
    assert corrupt3 is False
    assert "4gram_repetition" in r3


def test_turboquant_hybrid_cache_spec():
    arm = ArmSpec(id="A7_plus", name="TurboQuant+", k_bits=8, v_bits=4, hadamard=True, boundary_protect=True)
    cache = TurboQuantHybridCache(config=None, arm=arm)
    assert cache.arm.id == "A7_plus"

    # Layer 3 is boundary layer -> 8-bit
    k_bits_b, v_bits_b = cache._layer_bits(3)
    assert k_bits_b == 8
    assert v_bits_b == 8

    # Layer 7 is regular layer -> 8-bit K, 4-bit V
    k_bits_r, v_bits_r = cache._layer_bits(7)
    assert k_bits_r == 8
    assert v_bits_r == 4
