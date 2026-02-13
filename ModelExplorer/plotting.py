# ModelExplorer/plotting.py

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class PlotManager:
    """Small wrapper around a Matplotlib canvas for consistent plotting."""

    def __init__(self, figsize=(6, 4)) -> None:
        """Create the figure, axes, and Qt canvas."""
        self.figure, self.ax = plt.subplots(figsize=figsize)
        self.canvas = FigureCanvas(self.figure)
        self.scale = 100.0

    def plot(
        self,
        Q,
        I,  # noqa: E741
        Q_unit: str,
        data: Optional[Dict[str, Any]] = None,
        chi_square_text: Optional[str] = None,
    ) -> None:
        """Render the current intensity curve on log-log axes."""
        self.ax.clear()
        model_label = "Model"
        if chi_square_text:
            model_label = f"Model ({chi_square_text})"
        self.ax.plot(Q, I * self.scale, "-", label=model_label)
        if data is not None:
            data_Q = data.get("Q")
            data_I = data.get("I")
            data_sigma = data.get("ISigma")
            label = data.get("label", "Data")
            if data_Q is not None and data_I is not None:
                if data_sigma is not None:
                    self.ax.errorbar(
                        data_Q,
                        data_I,
                        yerr=data_sigma,
                        fmt="o",
                        markersize=3,
                        label=label,
                    )
                else:
                    self.ax.plot(data_Q, data_I, "o", markersize=3, label=label)
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel(f"Q ({Q_unit})")
        self.ax.set_ylabel("I (1/(m sr))")
        if data is not None or chi_square_text:
            self.ax.legend()
        self.canvas.draw()
