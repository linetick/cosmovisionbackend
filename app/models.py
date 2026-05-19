from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, String, Text, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id             = Column(Integer, primary_key=True)
    email          = Column(String(255), unique=True, nullable=False)
    password_hash  = Column(String(255), nullable=False)
    created_at     = Column(DateTime, server_default=func.now())
    tokens_used    = Column(Integer, server_default="0", nullable=False)
    tokens_reset_at = Column(DateTime, nullable=True)
    refresh_token            = Column(String(512), nullable=True, index=True)
    refresh_token_expires_at = Column(DateTime, nullable=True)

    subscriptions       = relationship("Subscription", back_populates="user")
    sessions            = relationship("Session", back_populates="user")
    owned_groups        = relationship("Group", back_populates="owner")
    group_memberships   = relationship("GroupMember", back_populates="user")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    started_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime, nullable=False)
    is_active  = Column(Boolean, server_default="true", nullable=False)

    user = relationship("User", back_populates="subscriptions")


class Group(Base):
    __tablename__ = "groups"

    id         = Column(Integer, primary_key=True)
    name       = Column(String(255), nullable=False)
    owner_id   = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    owner         = relationship("User", back_populates="owned_groups")
    members       = relationship("GroupMember", back_populates="group")
    subscriptions = relationship("GroupSubscription", back_populates="group")


class GroupMember(Base):
    __tablename__ = "group_members"

    id       = Column(Integer, primary_key=True)
    group_id = Column(Integer, ForeignKey("groups.id", ondelete="CASCADE"), nullable=False)
    user_id  = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role     = Column(String(50), server_default="member", nullable=False)  # owner / member
    added_at = Column(DateTime, server_default=func.now())

    group = relationship("Group", back_populates="members")
    user  = relationship("User", back_populates="group_memberships")


class GroupSubscription(Base):
    __tablename__ = "group_subscriptions"

    id          = Column(Integer, primary_key=True)
    group_id    = Column(Integer, ForeignKey("groups.id", ondelete="CASCADE"), nullable=False)
    started_at  = Column(DateTime, server_default=func.now())
    expires_at  = Column(DateTime, nullable=False)
    is_active   = Column(Boolean, server_default="true", nullable=False)
    max_members = Column(Integer, server_default="30", nullable=False)

    group = relationship("Group", back_populates="subscriptions")


class Spacecraft(Base):
    __tablename__ = "spacecraft"

    id             = Column(String(100), primary_key=True)
    name           = Column(String(255), nullable=False)
    description    = Column(Text, nullable=True)
    spacecraft_name = Column(String(255), nullable=True)
    format         = Column(String(20), nullable=False)
    file_name      = Column(String(255), nullable=False)
    relative_path  = Column(String(500), nullable=False)
    scene          = Column(JSONB, nullable=True)

    sessions = relationship("Session", back_populates="spacecraft")


class Session(Base):
    __tablename__ = "sessions"

    id            = Column(Integer, primary_key=True)
    user_id       = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    spacecraft_id = Column(String(100), ForeignKey("spacecraft.id", ondelete="SET NULL"), nullable=True)
    title         = Column(String(255), nullable=True)
    created_at    = Column(DateTime, server_default=func.now())
    updated_at    = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user       = relationship("User", back_populates="sessions")
    spacecraft = relationship("Spacecraft", back_populates="sessions")
    messages   = relationship("Message", back_populates="session")


class Message(Base):
    __tablename__ = "messages"

    id           = Column(Integer, primary_key=True)
    session_id   = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role         = Column(String(20), nullable=False)   # user / assistant
    content      = Column(Text, nullable=False)
    intent       = Column(String(50), nullable=True)    # info / action / hybrid / off_topic
    tokens_count = Column(Integer, nullable=True)
    created_at   = Column(DateTime, server_default=func.now())

    session = relationship("Session", back_populates="messages")
