# ModelExplorer/plotting.py

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from .types import OverlayData
from .utils.units import MODEL_INTENSITY_SCALE


class PlotManager:
    """Small wrapper around a Matplotlib canvas for consistent plotting."""

    def __init__(self, figsize=(6, 4)) -> None:
        """Create the figure, axes, and Qt canvas."""
        matplotlib.use("QtAgg", force=True)
        self.figure, self.ax = plt.subplots(figsize=figsize)
        self.canvas = FigureCanvas(self.figure)
        self.scale = MODEL_INTENSITY_SCALE

    def plot(
        self,
        Q,
        I,  # noqa: E741
        Q_unit: str,
        data: Optional[OverlayData] = None,
        chi_square_text: Optional[str] = None,
    ) -> None:
        """Render the current intensity curve on log-log axes."""
        self.ax.clear()
        self.ax.set_axisbelow(True)
        self.ax.minorticks_on()
        model_label = "Model"
        if chi_square_text:
            model_label = f"Model ({chi_square_text})"
        self.ax.plot(Q, I * self.scale, "-", label=model_label)
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
        self.ax.set_xlabel(f"Q ({Q_unit})")
        self.ax.set_ylabel("I (1/(m sr))")
        self.ax.grid(which="major", color="0.85", linewidth=0.6)
        self.ax.grid(which="minor", color="0.9", linewidth=0.4)
        if data is not None or chi_square_text:
            self.ax.legend()
        self.canvas.draw()
