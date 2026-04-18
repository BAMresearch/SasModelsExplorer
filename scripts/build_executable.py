#!/usr/bin/env python3

"""Build a PyInstaller executable with bundled sasmodels assets."""

import importlib.util
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if importlib.util.find_spec("PyInstaller") is None:
        print("PyInstaller is not installed for this interpreter.\nInstall it with: python -m pip install pyinstaller")
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--windowed",
        "-n",
        "SasModelsExplorer",
        "--collect-all",
        "sasmodels",
        "--collect-submodules",
        "sasmodels.models",
        "--hidden-import",
        "scipy.special._cdflib",
        str(repo_root / "ModelExplorer" / "__main__.py"),
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
