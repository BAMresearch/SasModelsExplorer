"""Runtime compatibility helpers for sasmodels in frozen applications."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any

_SASMODELS_RUNTIME_CONFIGURED = False


def _normalize_path(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _apply_tinycc_tccbox_compat(kerneldll: ModuleType, tccbox: ModuleType) -> bool:
    """Patch sasmodels tinycc include path when tccbox uses ``libtcc`` layout."""

    if getattr(kerneldll, "COMPILER", None) != "tinycc":
        return False

    cc = getattr(kerneldll, "CC", None)
    if not isinstance(cc, list):
        return False

    tcc_dist = Path(str(tccbox.tcc_dist_dir()))
    legacy_include = tcc_dist / "include"
    include_dir = Path(str(tccbox.tcc_include_dir()))

    if legacy_include.is_dir() or not include_dir.is_dir():
        return False

    updated_cc: list[Any] = list(cc)
    updated = False
    for index, arg in enumerate(updated_cc):
        if not isinstance(arg, str) or not arg.startswith("-I"):
            continue
        include_path = arg[2:]
        if _normalize_path(include_path) == _normalize_path(legacy_include):
            updated_cc[index] = f"-I{include_dir}"
            updated = True

    if not updated:
        return False

    vars(kerneldll)["CC"] = updated_cc
    logging.info("Adjusted sasmodels tinycc include path to %s", include_dir)
    return True


def configure_sasmodels_runtime() -> None:
    """Configure runtime compatibility workarounds for sasmodels."""

    global _SASMODELS_RUNTIME_CONFIGURED
    if _SASMODELS_RUNTIME_CONFIGURED:
        return
    _SASMODELS_RUNTIME_CONFIGURED = True

    if os.name != "nt":
        return

    try:
        import sasmodels.kerneldll as kerneldll
    except Exception as exc:  # pragma: no cover - import failure is non-fatal fallback
        logging.debug("Skipping sasmodels runtime patch (kerneldll import failed): %s", exc)
        return

    try:
        import tccbox
    except Exception as exc:  # pragma: no cover - import failure is non-fatal fallback
        logging.debug("Skipping sasmodels runtime patch (tccbox import failed): %s", exc)
        return

    _apply_tinycc_tccbox_compat(kerneldll, tccbox)
