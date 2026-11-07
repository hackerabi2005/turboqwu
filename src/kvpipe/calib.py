"""TriAttention calibration stats resolver and wrapper (PLAN.md P5.1)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SHIPPED_STATS_DIR = Path("third_party/triattention/triattention/calibration/for_aime25_experiment")

MODEL_TO_SHIPPED_STATS = {
    "qwen/qwen3-8b": SHIPPED_STATS_DIR / "qwen3_8b.pt",
    "qwen3-8b": SHIPPED_STATS_DIR / "qwen3_8b.pt",
    "deepseek-ai/deepseek-r1-distill-llama-8b": SHIPPED_STATS_DIR / "ds_llama8b.pt",
    "deepseek-r1-distill-llama-8b": SHIPPED_STATS_DIR / "ds_llama8b.pt",
    "deepseek-ai/deepseek-r1-distill-qwen-7b": SHIPPED_STATS_DIR / "ds_qwen7b.pt",
    "deepseek-r1-distill-qwen-7b": SHIPPED_STATS_DIR / "ds_qwen7b.pt",
}


def resolve_calibration_stats(
    model: str,
    output_dir: str | Path = "artifacts/calib",
    calibration_text: str | Path | None = None,
    allow_custom_calibrate: bool = False,
) -> Path:
    """Resolve TriAttention calibration statistics path.

    Uses official shipped stats for verified models, or runs calibration script.
    """
    model_norm = model.strip().lower()

    # 1. Shipped stats check
    if model_norm in MODEL_TO_SHIPPED_STATS:
        shipped = MODEL_TO_SHIPPED_STATS[model_norm]
        if shipped.exists():
            return shipped.resolve()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sanitized_model = model.replace("/", "_")
    target_stats = out_dir / f"{sanitized_model}_stats.pt"

    if target_stats.exists():
        return target_stats.resolve()

    if not allow_custom_calibrate:
        # If no shipped stats and custom calibration not explicitly permitted, look for default
        default_pt = SHIPPED_STATS_DIR / "qwen3_8b.pt"
        if default_pt.exists():
            return default_pt.resolve()
        raise FileNotFoundError(f"No precomputed calibration stats found for model '{model}'")

    # 2. Run official calibration script
    calib_script = Path("third_party/triattention/scripts/calibrate.py")
    if not calib_script.exists():
        raise FileNotFoundError(f"TriAttention calibration script not found at {calib_script}")

    text_input = calibration_text or "third_party/triattention/README.md"

    cmd = [
        sys.executable,
        str(calib_script),
        "--model",
        model,
        "--input",
        str(text_input),
        "--output",
        str(target_stats),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(f"Calibration failed: {res.stderr}")

    return target_stats.resolve()
