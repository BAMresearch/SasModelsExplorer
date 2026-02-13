# ModelExplorer/plotting.py

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class PlotManager:
    """Small wrapper around a Matplotlib canvas for consistent plotting."""

    def __init__(self, figsize=(6, 4)) -> None:
        """Create the figure, axes, and Qt canvas."""
        self.figure, self.ax = plt.subplots(figsize=figsize)
        self.canvas = FigureCanvas(self.figure)

    def plot(self, q, intensity, qunit: str) -> None:
        """Render the current intensity curve on log-log axes."""
        self.ax.clear()
        self.ax.plot(q, intensity * 100.0, "-")
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel(f"q ({qunit})")
        self.ax.set_ylabel("I (1/(m sr))")
        self.canvas.draw()
