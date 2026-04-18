from __future__ import annotations

from types import SimpleNamespace

from ModelExplorer.utils.sasmodels_runtime import _apply_tinycc_tccbox_compat


def test_apply_tinycc_tccbox_compat_rewrites_missing_include_path(tmp_path):
    tcc_dist = tmp_path / "tcc_dist"
    tcc_dist.mkdir()
    include_dir = tcc_dist / "libtcc"
    include_dir.mkdir()

    kerneldll = SimpleNamespace(
        COMPILER="tinycc",
        CC=["tcc.exe", "-nostdinc", f"-I{tcc_dist / 'include'}", "-shared"],
    )
    tccbox = SimpleNamespace(
        tcc_dist_dir=lambda: str(tcc_dist),
        tcc_include_dir=lambda: str(include_dir),
    )

    updated = _apply_tinycc_tccbox_compat(kerneldll, tccbox)

    assert updated is True
    assert f"-I{include_dir}" in kerneldll.CC
