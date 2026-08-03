"""Settings load from the environment and fail fast on bad values."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_system.core.config import Settings


def test_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # isolate from a developer's real .env
    settings = Settings()
    assert settings.debug is False
    assert settings.log_level == "INFO"
    assert settings.data_dir == Path("data")
    assert settings.log_file is None


def test_reads_prefixed_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TS_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TS_DEBUG", "true")
    settings = Settings()
    assert settings.log_level == "DEBUG"
    assert settings.debug is True


def test_invalid_log_level_fails_at_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TS_LOG_LEVEL", "VERBOSE")
    with pytest.raises(ValidationError):
        Settings()


def test_module_exposes_no_settings_singleton() -> None:
    import trading_system.core.config as config_module

    assert not hasattr(config_module, "settings")
