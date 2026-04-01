"""CLI entrypoint for launching the SasModels Explorer GUI."""

from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from ModelExplorer.modelexplorer import SasModelApp
from ModelExplorer.utils.configure_logging import configure_logging


def setup_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for model selection and logging options."""

    parser = argparse.ArgumentParser(description="SasModels Explorer")
    parser.add_argument("model", nargs="?", default="sphere@hardsphere", help="Model name to display")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity to INFO level.",
    )
    parser.add_argument(
        "-vv",
        "--very_verbose",
        action="store_true",
        help="Increase output verbosity to DEBUG level.",
    )
    parser.add_argument(
        "-l",
        "--logging",
        action="store_true",
        help="Write log output to a timestamped file.",
    )
    return parser.parse_args(args)


def main() -> None:
    """Launch the Qt application."""

    argv = sys.argv
    app = QApplication(argv)
    args = setup_args(argv[1:])
    configure_logging(
        args.verbose,
        args.very_verbose,
        log_to_file=args.logging,
        log_file_prepend="SasModelsExplorer_",
    )
    window = SasModelApp(args.model)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
