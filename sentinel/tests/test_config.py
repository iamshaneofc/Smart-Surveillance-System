def test_settings_defaults():
    from packages.config import Settings

    settings = Settings(_env_file=None)
    assert settings.app_name == "SENTINEL"
    assert settings.auth_mode == "disabled"
    assert settings.evidence.pre_seconds == 10
    assert settings.evidence.retention_days_by_severity["critical"] == 90


def test_settings_env_override(monkeypatch):
    from packages.config import Settings

    monkeypatch.setenv("SENTINEL_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("SENTINEL_EVIDENCE__PRE_SECONDS", "25")
    settings = Settings(_env_file=None)
    assert settings.log_level == "DEBUG"
    assert settings.evidence.pre_seconds == 25
