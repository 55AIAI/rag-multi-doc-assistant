"""Initial private metadata schema.

Revision ID: 001
Revises:
"""
from alembic import op
from app.core import Base
from app import models  # noqa: F401

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    Base.metadata.create_all(op.get_bind())


def downgrade():
    Base.metadata.drop_all(op.get_bind())
