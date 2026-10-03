from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from packages.common.logging import get_logger

log = get_logger(__name__)


class Base(DeclarativeBase):
    pass


_engine = None
_sessionmaker: sessionmaker | None = None


def configure(url: str) -> None:
    global _engine, _sessionmaker
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url.endswith(":memory:"):
            kwargs["poolclass"] = StaticPool
    _engine = create_engine(url, **kwargs)
    _sessionmaker = sessionmaker(bind=_engine, expire_on_commit=False, autoflush=False)
    log.info("database configured", extra={"url": url.split("@")[-1]})


def is_configured() -> bool:
    return _sessionmaker is not None


def _ensure_configured() -> sessionmaker:
    global _sessionmaker
    if _sessionmaker is None:
        from packages.config import get_settings

        configure(get_settings().database_url)
    assert _sessionmaker is not None
    return _sessionmaker


def get_session() -> Iterator[Session]:
    factory = _ensure_configured()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    factory = _ensure_configured()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all() -> None:
    from packages.db import models  # noqa: F401

    _ensure_configured()
    assert _engine is not None
    Base.metadata.create_all(_engine)


def dispose() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _sessionmaker = None
