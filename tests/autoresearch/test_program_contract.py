from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_program_preserves_bootstrap_crash_and_state_commit_contracts() -> None:
    path = Path("autoresearch/program.md")
    text = path.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 74
    assert "No campaign begins until Phase 0 is complete" in text
    assert "human materializes that winner" in text
    assert "every per-trial state commit includes changed `ideas.md`" in text
    assert "On a second failure, log `crash`" in text
    assert "revert repair and proposal commits in reverse order before the next trial" in text
