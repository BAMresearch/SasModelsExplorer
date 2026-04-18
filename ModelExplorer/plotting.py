"""Plotting helpers for model and overlay visualization."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from numpy.typing import NDArray

from .types import OverlayData
from .utils.units import MODEL_INTENSITY_SCALE


class PlotManager:
    """Small wrapper around a Matplotlib canvas for consistent plotting."""

    def __init__(self, figsize: tuple[float, float] = (6, 4)) -> None:
        """Create the figure, axes, and Qt canvas."""

        matplotlib.use("QtAgg", force=True)
        self.figure, self.ax = plt.subplots(figsize=figsize)
        self.canvas = FigureCanvas(self.figure)
        self.scale = MODEL_INTENSITY_SCALE

    def plot(
        self,
        q_values: NDArray[np.float64],
        intensity_values: NDArray[np.float64],
        q_unit: str,
        data: OverlayData | None = None,
        chi_square_text: str | None = None,
    ) -> None:
        """Render the current intensity curve on log-log axes."""

        self.ax.clear()
        self.ax.set_axisbelow(True)
        self.ax.minorticks_on()

        model_label = "Model" if not chi_square_text else f"Model ({chi_square_text})"
        self.ax.plot(q_values, intensity_values * self.scale, "-", label=model_label)

        if data is not None:
            if data.ISigma is not None:
                self.ax.errorbar(
                    data.Q,
                    data.I,
                    yerr=data.ISigma,
                    fmt="o",
                    markersize=3,
                    label=data.label,
                )
            else:
                self.ax.plot(data.Q, data.I, "o", markersize=3, label=data.label)

        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel(f"Q ({q_unit})")
        self.ax.set_ylabel("I (1/(m sr))")
        self.ax.grid(which="major", color="0.85", linewidth=0.6)
        self.ax.grid(which="minor", color="0.9", linewidth=0.4)
        if data is not None or chi_square_text:
            self.ax.legend()
        self.canvas.draw()
