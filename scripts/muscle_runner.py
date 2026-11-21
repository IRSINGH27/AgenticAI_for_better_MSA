import subprocess
import tempfile
import shutil
import os
from typing import Optional

from config import CONDA_ENV


def run_muscle(fasta_text: str, params: Optional[dict] = None) -> str:
    if params is None:
        params = {}

    algorithm = params.get("algorithm") or params.get("strategy", "align")
    algorithm = str(algorithm).lstrip("-")
    valid_algorithms = {"align", "progressive", "super5"}
    if algorithm not in valid_algorithms:
        algorithm = "align"

    cmd = ["conda", "run", "-n", CONDA_ENV, "muscle", "-" + algorithm]

    diversified = params.get("diversified", False)
    stratified = params.get("stratified", False)
    use_maxcc = params.get("maxcc", False)

    try:
        maxiters = int(params.get("maxiters", 1000))
    except (ValueError, TypeError):
        maxiters = 1000
    maxiters = max(1000, maxiters)
    use_maxiters = algorithm in ("align", "super5")

    if "threads" in params:
        try:
            cmd.extend(["-threads", str(int(params["threads"]))])
        except (ValueError, TypeError):
            pass

    extra_args_str = params.get("extra_args", "")
    if extra_args_str:
        allowed_extra = {
            "-perm", "-perturb", "-replicates", "-threads", "-quiet",
            "-diversified", "-stratified", "-maxcc",
            "-minconf", "-maxgapfract", "-nocpuinfo",
        }
        extra_tokens = extra_args_str.split()
        filtered = []
        skip_next = False
        for i, tok in enumerate(extra_tokens):
            if skip_next:
                skip_next = False
                continue
            base = tok.split("=")[0]
            if base in allowed_extra:
                filtered.append(tok)
                if base in ("-perm", "-perturb", "-replicates", "-threads", "-minconf", "-maxgapfract") and "=" not in tok:
                    if i + 1 < len(extra_tokens) and not extra_tokens[i + 1].startswith("-"):
                        filtered.append(extra_tokens[i + 1])
                        skip_next = True
        cmd.extend(filtered)

    try:
        check = subprocess.run(
            ["conda", "run", "-n", CONDA_ENV, "muscle", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if check.returncode != 0:
            raise RuntimeError("MUSCLE not found in conda env")
    except FileNotFoundError:
        raise RuntimeError("Conda not found. Install conda or add muscle to PATH")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.fasta")

        with open(input_path, "w") as f:
            f.write(fasta_text)

        perm_args = []
        if "perm" in params:
            perm = params["perm"]
            valid_perms = {"none", "abc", "acb", "bca"}
            if perm in valid_perms:
                perm_args.extend(["-perm", perm])

        if "perturb" in params:
            try:
                perm_args.extend(["-perturb", str(int(params["perturb"]))])
            except (ValueError, TypeError):
                pass

        if "replicates" in params:
            try:
                perm_args.extend(["-replicates", str(int(params["replicates"]))])
            except (ValueError, TypeError):
                pass

        maxiters_arg = ["-maxiters", str(maxiters)] if use_maxiters else []

        if diversified and not stratified:
            output_path = os.path.join(tmpdir, "diversified.afa")
            cmd.extend([input_path, "-diversified"] + perm_args + maxiters_arg + ["-output", output_path])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"MUSCLE failed: {e.stderr.strip()}") from e

            with open(output_path, "r") as f:
                return f.read()

        elif stratified:
            ensemble_path = os.path.join(tmpdir, "stratified.efa")
            cmd.extend([input_path, "-stratified"] + perm_args + maxiters_arg + ["-output", ensemble_path])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"MUSCLE failed: {e.stderr.strip()}") from e

            if use_maxcc:
                maxcc_path = os.path.join(tmpdir, "maxcc.afa")
                maxcc_cmd = ["conda", "run", "-n", CONDA_ENV, "muscle", "-maxcc", ensemble_path, "-output", maxcc_path]
                try:
                    subprocess.run(maxcc_cmd, capture_output=True, text=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"MUSCLE maxcc failed: {e.stderr.strip()}") from e

                with open(maxcc_path, "r") as f:
                    return f.read()
            else:
                with open(ensemble_path, "r") as f:
                    return f.read()
        else:
            output_path = os.path.join(tmpdir, "output.afa")
            cmd.extend([input_path] + perm_args + maxiters_arg + ["-output", output_path])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"MUSCLE failed: {e.stderr.strip()}") from e

            with open(output_path, "r") as f:
                return f.read()


MUSCLE_ALGORITHMS = {
    "align": "Standard progressive alignment (default, fast)",
    "progressive": "Classic progressive algorithm",
    "super5": "Super5 algorithm (high accuracy, slower, for large datasets)"
}

MUSCLE_PERM_OPTIONS = {
    "none": "No permutation - use input order as-is (fastest)",
    "abc": "Permutation abc - good for general use",
    "acb": "Permutation acb - alternative ordering",
    "bca": "Permutation bca - alternative ordering"
}

MUSCLE_COMMON_FLAGS = {
    "-diversified": "Use diversified algorithm - single best alignment from ensemble approach (recommended for diverse sequences)",
    "-stratified": "Generate stratified ensemble (EFA format) with multiple replicates for column confidence scoring",
    "-maxcc": "Extract the replicate with highest column confidence from a stratified ensemble",
    "-perm <type>": "Guide tree permutation: none, abc, acb, bca (default: none)",
    "-perturb <seed>": "Perturbation seed for ensemble diversity (0, 1, 2, ...)",
    "-replicates <n>": "Number of replicates (default: 4 for diversified, 100 for stratified)",
    "-maxiters <n>": "Pipeline sets this automatically (min 1000); muscle 5 may report 'not used'",
    "-threads <n>": "Number of threads (default: 1)",
    "-quiet": "Suppress progress output"
}