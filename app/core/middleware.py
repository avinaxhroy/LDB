# app/core/middleware.py
from fastapi import Request
from time import time
import logging

logger = logging.getLogger(__name__)


async def performance_middleware(request: Request, call_next):
    """Middleware to track request performance"""
    start_time = time()

    # Process the request
    response = await call_next(request)

    # Calculate processing time
    process_time = time() - start_time

    # Log performance data
    logger.info(
        f"Request to {request.url.path} processed in {process_time:.4f} seconds"
    )

    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)

    return response
