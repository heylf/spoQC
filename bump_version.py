# In[]
import re
import argparse
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:

    # parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    # mandatory
    parser.add_argument(
        "-v", "--version",
        dest="version",
        type=str, 
        help="New version number.",
        required=True
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args_ns = parser.parse_args()
    args = vars(args_ns)
    version = args['version']

    # setup.py
    text = Path("setup.py").read_text()
    text = re.sub(
        r"version='(.*?)'",
        f"version='{version}'",
        text,
    )
    outfile = Path("setup.py")
    outfile.write_text(text)

    # pyproject.toml
    text = Path("pyproject.toml").read_text()
    text = re.sub(
        r"version = \"(.*?)\"",
        f"version = \"{version}\"",
        text,
    )
    outfile = Path("pyproject.toml")
    outfile.write_text(text)

    # docs/source/conf.py
    text = Path("docs/source/conf.py").read_text()
    text = re.sub(
        r"release = '(.*?)'",
        f"release = '{version}'",
        text,
    )
    outfile = Path("docs/source/conf.py")
    outfile.write_text(text)

    # __init__.py
    text = Path("spoqc/__init__.py").read_text()
    text = re.sub(
        r"__version__ = \"(.*?)\"",
        f"__version__ = \"{version}\"",
        text,
    )
    outfile = Path("spoqc/__init__.py")
    outfile.write_text(text)

# %%
