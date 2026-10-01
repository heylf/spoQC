from __future__ import annotations

from .cli_args import build_parser
from .core import threads

# --dev_test and -s unittest run on a fixed budget, so their outputs compare across hosts
DEV_TEST_THREADS = 8


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    threads.configure(
        DEV_TEST_THREADS if args.dev_test or args.step == "unittest" else args.threads
    )
    # Imported here, not at module level: spawned figure workers import this module as their
    # __main__, and spoqc.cli takes ~20 s and ~700 MB to import. It also must come after
    # threads.configure, which has to run before the heavy libraries load.
    from .cli import main as run

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
