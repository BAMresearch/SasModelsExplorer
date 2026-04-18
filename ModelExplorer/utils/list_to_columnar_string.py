"""Formatting helper for display-friendly columnar string lists."""

from __future__ import annotations

import math


def list_to_columnar_string(
    values: list[str],
    ncols: int = 2,
    minimum_column_width: int = 0,
    padding: str = "  ",
    ordering: str = "columns",
) -> str:
    """Format string values into fixed-width columnar text."""

    if len(values) == 0:
        return ""
    if ncols <= 0:
        raise ValueError("ncols must be a positive integer.")

    max_width = max(minimum_column_width, max(len(value) for value in values))
    cut = int(math.ceil(len(values) / ncols))
    padded = list(values)
    while len(padded) % ncols != 0:
        padded.append("")

    columnar_lines: list[str] = []
    for row_index in range(cut):
        if ordering == "columns":
            row_values = [padded[row_index + col_index * cut].ljust(max_width) for col_index in range(ncols)]
        else:
            row_values = [padded[row_index * ncols + col_index].ljust(max_width) for col_index in range(ncols)]
        columnar_lines.append(padding.join(row_values))
    return "\n".join(columnar_lines) + "\n"
