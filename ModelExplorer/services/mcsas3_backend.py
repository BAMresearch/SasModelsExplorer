"""Typed adapter around canonical McSAS3 data-loading APIs."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, cast

if TYPE_CHECKING:
    import pandas as pd
    from mcsas3.data_model import DataBundle, ProcessingData


class ProcessingBackend(Protocol):
    """Protocol describing the backend surface required by ``data_loader``."""

    def prepare_processing_from_file(
        self,
        data_path: Path,
        *,
        result_index: int,
        workflow_config: Mapping[str, object],
    ) -> "ProcessingData":
        """Prepare canonical processing data from a source file."""

    def selected_bundle_from_processing(self, processing: "ProcessingData", *, stage_name: str) -> "DataBundle":
        """Return a stage-selected canonical bundle from processing data."""

    def frame_from_bundle(self, bundle: "DataBundle") -> "pd.DataFrame":
        """Convert a canonical bundle to a plotting dataframe."""


@dataclass(frozen=True, slots=True)
class CanonicalMcSAS3Backend:
    """Runtime adapter that delegates to canonical McSAS3 functions."""

    _prepare_processing_from_file: Callable[..., object]
    _selected_bundle_from_processing: Callable[..., object]
    _frame_from_bundle: Callable[..., object]

    def prepare_processing_from_file(
        self,
        data_path: Path,
        *,
        result_index: int,
        workflow_config: Mapping[str, object],
    ) -> object:
        """Prepare canonical processing data from ``data_path`` and workflow config."""

        return self._prepare_processing_from_file(
            data_path,
            result_index=result_index,
            **dict(workflow_config),
        )

    def selected_bundle_from_processing(self, processing: object, *, stage_name: str) -> object:
        """Select a canonical bundle for a given stage name."""

        return self._selected_bundle_from_processing(processing, stage_name=stage_name)

    def frame_from_bundle(self, bundle: object) -> object:
        """Create a dataframe projection from a canonical bundle."""

        return self._frame_from_bundle(bundle)


def load_mcsas3_backend() -> ProcessingBackend:
    """Return a backend adapter over canonical McSAS3 APIs."""

    prepare_processing_from_file: Callable[..., object]
    selected_bundle_from_processing: Callable[..., object]

    try:
        prepare_processing_from_file = _load_callable("mcsas3", "prepare_1d_processing_data_from_file")
        selected_bundle_from_processing = _load_callable("mcsas3", "selected_bundle_from_processing")
    except Exception as top_level_exc:  # pragma: no cover - exercised by runtime dependency checks
        # Keep a fallback path for branch snapshots where symbols are only exposed in submodules.
        try:
            prepare_processing_from_file = _load_callable("mcsas3.workflows", "prepare_1d_processing_data_from_file")
            selected_bundle_from_processing = _load_callable("mcsas3.data_adapters", "selected_bundle_from_processing")
        except Exception as module_exc:
            raise ImportError(
                "Failed to import McSAS3 canonical workflow. "
                "This project currently expects McSAS3 branch 'in_depth_upgrades'. "
                "Install/update with:\n"
                "  pip install --upgrade "
                '"mcsas3 @ git+https://github.com/BAMresearch/McSAS3.git@in_depth_upgrades"\n'
                f"Top-level import error: {top_level_exc}\n"
                f"Submodule import error: {module_exc}"
            ) from module_exc

    try:
        frame_from_bundle = _load_callable("mcsas3.data_adapters", "frame_from_bundle")
    except Exception as exc:
        raise ImportError(f"Failed to import McSAS3 data adapters: {exc}") from exc

    return CanonicalMcSAS3Backend(
        _prepare_processing_from_file=prepare_processing_from_file,
        _selected_bundle_from_processing=selected_bundle_from_processing,
        _frame_from_bundle=frame_from_bundle,
    )


def _load_callable(module_name: str, symbol_name: str) -> Callable[..., object]:
    """Load a callable symbol from module path and raise on missing symbol."""

    module = importlib.import_module(module_name)
    symbol = getattr(module, symbol_name, None)
    if not callable(symbol):
        raise TypeError(f"{module_name}.{symbol_name} is not callable")
    return cast(Callable[..., object], symbol)
