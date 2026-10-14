"""Refresh-token rotation on a UUID-typed model, under SQLite (what project
test suites run on): ids arrive as strings, the column wants UUIDs."""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import UUID, Boolean, Column, DateTime, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import StaticPool

from sunset.services.auth import AuthService

Base = declarative_base()


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    token_hash = Column(String, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, default=False)
    replaced_by = Column(UUID(as_uuid=True), nullable=True)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


@pytest.mark.asyncio
async def test_rotation_accepts_string_user_ids(session):
    auth = AuthService(jwt_secret="x" * 32, refresh_token_model=RefreshToken)
    user_id = uuid.uuid4()

    raw, row = await auth.create_refresh_token(str(user_id), session)
    assert row.user_id == user_id

    # Reload from the database, as a later request would (naive datetimes under SQLite).
    session.expire_all()
    access, new_raw = await auth.rotate_refresh_token(raw, session)
    assert new_raw != raw
    assert auth.get_user_id_from_token(access) == str(user_id)

    with pytest.raises(ValueError, match="reuse"):
        await auth.rotate_refresh_token(raw, session)
