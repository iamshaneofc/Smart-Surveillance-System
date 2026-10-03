import pytest

from packages.config import Settings


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        env="test",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        bus_url="memory://",
        auth_mode="disabled",
        rate_limit_per_minute=10000,
        log_level="WARNING",
    )


@pytest.fixture()
def app(settings):
    from packages.db import base as db_base

    db_base.dispose()
    db_base.configure(settings.database_url)
    db_base.create_all()

    from apps.api.main import create_app

    application = create_app(settings)
    yield application
    application.state.bus.close()
    db_base.dispose()


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def api_key_settings(tmp_path) -> Settings:
    from packages.config import ApiKeySpec

    return Settings(
        env="test",
        database_url=f"sqlite:///{tmp_path / 'auth.db'}",
        bus_url="memory://",
        auth_mode="api_key",
        auth_api_keys=[
            ApiKeySpec(key="viewer-key", user="viewer1", roles=["viewer"]),
            ApiKeySpec(key="operator-key", user="operator1", roles=["operator"]),
            ApiKeySpec(key="admin-key", user="admin1", roles=["admin"]),
        ],
        rate_limit_per_minute=10000,
        log_level="WARNING",
    )


@pytest.fixture()
def api_key_client(api_key_settings):
    from fastapi.testclient import TestClient

    from packages.db import base as db_base

    db_base.dispose()
    db_base.configure(api_key_settings.database_url)
    db_base.create_all()

    from apps.api.main import create_app

    application = create_app(api_key_settings)
    with TestClient(application) as test_client:
        yield test_client
    application.state.bus.close()
    db_base.dispose()


@pytest.fixture()
def db_session(settings):
    from packages.db import base as db_base

    already = db_base.is_configured()
    if not already:
        db_base.configure(settings.database_url)
        db_base.create_all()
    session = next(db_base.get_session())
    yield session
    session.close()
