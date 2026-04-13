from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

try:
    from demeter import settings as _settings
    from demeter.models import Base
except ImportError:
    import settings as _settings
    from models import Base


def _build_engine():
    Path(_settings.SOLAR_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{_settings.SOLAR_DB_PATH}")

    @event.listens_for(engine, "connect")
    def set_wal_mode(conn, _):
        conn.execute("PRAGMA journal_mode=WAL")

    return engine


_engine = _build_engine()
_SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(_engine)


@contextmanager
def get_session() -> Session:
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
