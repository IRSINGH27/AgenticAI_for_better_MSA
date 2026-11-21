import subprocess
import tempfile
import shutil
import os
from typing import Optional

from config import CONDA_ENV


def run_trimal(alignment_text: str, params: Optional[dict] = None) -> str:
    if params is None:
        params = {}

    strategy = params.get("strategy", "-gappyout")
    # Gentle modes only — strict/strictplus/nogaps wipe gap columns (lose indels).
    valid_strategies = {"-gappyout", "-automated1"}
    if strategy not in valid_strategies:
        strategy = "-gappyout"

    extra_args_str = params.get("extra_args", "")
    extra_args = extra_args_str.split() if extra_args_str else []

    try:
        check = subprocess.run(
            ["conda", "run", "-n", CONDA_ENV, "trimal", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if check.returncode != 0:
            raise RuntimeError("Trimal not found in conda env")
    except FileNotFoundError:
        raise RuntimeError("Conda not found. Install conda or add trimal to PATH")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.fasta")
        output_path = os.path.join(tmpdir, "output.fasta")

        with open(input_path, "w") as f:
            f.write(alignment_text)

        cmd = ["conda", "run", "-n", CONDA_ENV, "trimal", "-in", input_path, "-out", output_path, strategy] + extra_args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Trimal failed: {e.stderr.strip()}") from e

        with open(output_path, "r") as f:
            return f.read()


TRIMAL_STRATEGIES = {
    "-gappyout": "Automated selection based on gap distribution (gentle, keeps indels — default)",
    "-automated1": "Heuristic auto selection based on similarity statistics (gentle)",
}

TRIMAL_COMMON_FLAGS = {
    "-gt": "Gap threshold (0.0-1.0, default 0.0)",
    "-st": "Sequence identity threshold",
    "-cons": "Minimum percentage for consensus",
    "-seqoverlap": "Minimum percentage of good positions a sequence must keep (0-100)",
    "-htmlout": "Output HTML report",
    "-fasta": "Force FASTA output"
}