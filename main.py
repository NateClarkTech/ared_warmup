"""Run warm-start A/RED from a JSON config.

    python main.py tiny.json
    python main.py

With no file name, the configs are printed as a numbered list and one number
is read from stdin.
"""

from __future__ import annotations

import sys

from ared.algorithm import format_result, run
from ared.config import ConfigError


def main(argv=None, stdin=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) > 1:
        print("Usage: python main.py [config.json]", file=sys.stderr)
        return 2
    config_name = argv[0] if argv else None
    try:
        result = run(config_name, stdin=stdin)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 1
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    sys.stdout.write(format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
