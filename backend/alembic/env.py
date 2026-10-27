import os
from alembic import context
from sqlalchemy import engine_from_config, pool
from app.core import Base
from app import models  # noqa: F401

config = context.config
config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
target_metadata = Base.metadata


def run_migrations_offline():
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata,
                      literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    engine = engine_from_config(config.get_section(config.config_ini_section),
                                prefix="sqlalchemy.", poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
