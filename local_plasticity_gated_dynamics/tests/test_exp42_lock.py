from __future__ import annotations

import pytest

from experiments.exp42_locked_unexecuted import LOCK_MESSAGE, main


def test_exp42_entry_point_fails_closed_with_historical_reason() -> None:
    with pytest.raises(SystemExit, match="Exp41 failed its entry gate") as error:
        main()
    assert str(error.value) == LOCK_MESSAGE
