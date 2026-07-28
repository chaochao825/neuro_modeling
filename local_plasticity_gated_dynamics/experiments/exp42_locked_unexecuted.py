"""Fail-closed marker for the historical, unexecuted Exp42 plan."""

from __future__ import annotations


LOCK_MESSAGE = (
    "Exp42 is permanently locked as an unexecuted historical plan because "
    "Exp41 failed its entry gate; see "
    "docs/exp42_actuator_factorization_audit_plan_20260727.md"
)


def main() -> None:
    raise SystemExit(LOCK_MESSAGE)


if __name__ == "__main__":
    main()
