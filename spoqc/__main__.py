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
    from .cli import main as run  # heavy imports only after the thread budget is set

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
