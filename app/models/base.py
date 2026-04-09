"""Database base setup and utilities."""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


def init_db(database_url: str):
    """Initialize database connection."""
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
    )
    Base.metadata.create_all(bind=engine)
    return engine


def get_session_factory(engine):
    """Get session factory bound to engine."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
