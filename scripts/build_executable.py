#!/usr/bin/env python3

"""Build a PyInstaller executable with bundled sasmodels assets."""

import subprocess
import sys


def main() -> int:
    cmd = [
        "pyinstaller",
        "--windowed",
        "-n",
        "SasModelsExplorer",
        "--collect-all",
        "sasmodels",
        "--collect-submodules",
        "sasmodels.models",
        "--hidden-import",
        "scipy.special._cdflib",
        "ModelExplorer/__main__.py",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
