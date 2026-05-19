"""add auth tokens to users

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("refresh_token", sa.String(512), nullable=True))
    op.add_column("users", sa.Column("refresh_token_expires_at", sa.DateTime(), nullable=True))
    op.create_index("ix_users_refresh_token", "users", ["refresh_token"])


def downgrade() -> None:
    op.drop_index("ix_users_refresh_token", "users")
    op.drop_column("users", "refresh_token_expires_at")
    op.drop_column("users", "refresh_token")
