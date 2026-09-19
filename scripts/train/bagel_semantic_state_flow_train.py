#!/usr/bin/env python3
"""Removed semantic-token local-flow trainer."""

from __future__ import annotations

import sys


def main() -> None:
    raise RuntimeError(
        "semantic-token local-flow training was removed; "
        "use the Read-Route-Write memory body loop"
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
