import subprocess
import tempfile
import shutil
import os
from typing import Optional

from config import CONDA_ENV


def run_msa(fasta_text: str, params: Optional[dict] = None) -> str:
    if params is None:
        params = {}

    strategy = params.get("strategy", "--auto")
    valid_strategies = {"--auto", "--localpair", "--globalpair", "--genafpair", "--retree", "--fast"}
    if strategy not in valid_strategies:
        strategy = "--auto"

    extra_args_str = params.get("extra_args", "")
    extra_args = extra_args_str.split() if extra_args_str else []

    if "--maxiterate" in extra_args:
        idx = extra_args.index("--maxiterate")
        if idx + 1 < len(extra_args) and not extra_args[idx + 1].startswith("-"):
            try:
                extra_args[idx + 1] = str(max(1000, int(extra_args[idx + 1])))
            except ValueError:
                extra_args[idx + 1] = "1000"
        else:
            extra_args = extra_args[:idx + 1] + ["1000"]

    try:
        check = subprocess.run(
            ["conda", "run", "-n", CONDA_ENV, "mafft", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if check.returncode != 0:
            raise RuntimeError("MAFFT not found in conda env")
    except FileNotFoundError:
        raise RuntimeError("Conda not found. Install conda or add mafft to PATH")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.fasta")
        output_path = os.path.join(tmpdir, "output.fasta")

        with open(input_path, "w") as f:
            f.write(fasta_text)

        cmd = ["conda", "run", "-n", CONDA_ENV, "mafft", strategy] + extra_args + [input_path]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"MAFFT failed: {e.stderr.strip()}") from e

        return result.stdout