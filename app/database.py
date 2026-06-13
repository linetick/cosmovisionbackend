import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/cosmovision",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=100,
    max_overflow=400,
    pool_timeout=60,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        try:
            db.close()
        except Exception:
            pass
