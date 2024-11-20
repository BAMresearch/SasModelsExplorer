# ModelExplorer/__main__.py

from .modelexplorer import SasModelApp
from .utils.configure_logging import configure_logging
from PyQt5.QtWidgets import QApplication
import sys

def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO)

def setup_args(args=None):
    import argparse

    parser = argparse.ArgumentParser(description='SasModels interactive app')
    parser.add_argument('model', nargs='?', default='sphere@hardsphere',
                        help='Model name to display')
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
        help="Write log out to a timestamped file.",
    )
    args = parser.parse_args(args)

    return args

def main():
    argv = sys.argv
    app = QApplication(argv)
    args = setup_args(argv[1:])
    configure_logging(args.verbose, args.very_verbose, log_to_file=args.logging, log_file_prepend="HDF5Translator_")

    window = SasModelApp(args.model)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
