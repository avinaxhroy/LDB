# app/db/session.py
import time
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

# Create engine with existing settings (keeping your existing configuration)
engine = create_engine(settings.DATABASE_URL)

# Create session factory - your existing setup
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models - your existing setup
Base = declarative_base()


# Database dependency for routes - your existing function
def get_db():
    """
    Dependency to get DB session for API endpoints

    Yields:
        Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ----- New Monitoring Code Starts Here -----

# Add query timing for monitoring
@event.listens_for(engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log query start time before execution"""
    conn.info.setdefault('query_start_time', []).append(time.time())


@event.listens_for(engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log query execution time after execution"""
    total_time = time.time() - conn.info['query_start_time'].pop()

    # Log slow queries (taking more than 100ms)
    if total_time > 0.1:
        logger.warning(
            f"Slow query detected: {total_time:.4f} seconds",
            extra={
                "query_time_ms": round(total_time * 1000, 2),
                "query": statement[:200] + '...' if len(statement) > 200 else statement
            }
        )
    else:
        logger.debug(
            f"Query executed: {total_time:.4f} seconds",
            extra={
                "query_time_ms": round(total_time * 1000, 2),
            }
        )


# Add some basic connection pool monitoring
@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Track connection checkout for monitoring"""
    logger.debug("Database connection checked out from pool")
