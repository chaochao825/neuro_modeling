"""Fail closed until the prospectively specified Exp43 implementation is frozen."""

from __future__ import annotations


LOCK_MESSAGE = (
    "Exp43 has a prospective development protocol but no frozen implementation; "
    "no outcome may be generated from this placeholder"
)


def main() -> None:
    raise SystemExit(LOCK_MESSAGE)


if __name__ == "__main__":
    main()
