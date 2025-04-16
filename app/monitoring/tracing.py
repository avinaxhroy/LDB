# app/monitoring/tracing.py
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def setup_tracing(app: Any, engine: Optional[Any] = None):
    """Configure OpenTelemetry tracing with advanced options"""
    try:
        # Import OpenTelemetry components
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        # Framework-specific instrumentors
        try:
            from opentelemetry.instrumentation.flask import FlaskInstrumentor
            flask_available = True
        except ImportError:
            flask_available = False

        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            fastapi_available = True
        except ImportError:
            fastapi_available = False

        # Database instrumentation
        try:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
            sqlalchemy_available = True
        except ImportError:
            sqlalchemy_available = False

        # Redis instrumentation
        try:
            from opentelemetry.instrumentation.redis import RedisInstrumentor
            redis_available = True
        except ImportError:
            redis_available = False

        # HTTP client instrumentation
        try:
            from opentelemetry.instrumentation.requests import RequestsInstrumentor
            requests_available = True
        except ImportError:
            requests_available = False

        # Set up resource with detailed service info
        service_name = getattr(app, 'title', 'desi-hiphop-recommendation')
        resource = Resource.create({
            SERVICE_NAME: service_name,
            "service.version": getattr(app, 'version', '1.0.0'),
            "deployment.environment": "production"
        })

        # Create a tracer provider
        tracer_provider = TracerProvider(resource=resource)

        # Add console exporter for development visibility
        console_processor = BatchSpanProcessor(ConsoleSpanExporter())
        tracer_provider.add_span_processor(console_processor)

        # Add OTLP exporter for production systems that support it
        try:
            otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
            otlp_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(otlp_processor)
            logger.info("OTLP exporter configured successfully")
        except Exception as e:
            logger.warning(f"OTLP exporter not configured: {str(e)}")

        # Set the global tracer provider
        trace.set_tracer_provider(tracer_provider)

        # Instrument web framework based on type
        framework_module = app.__class__.__module__.split('.')[0]

        if framework_module == 'flask' and flask_available:
            logger.info("Instrumenting Flask application")
            FlaskInstrumentor().instrument_app(app)
        elif framework_module == 'fastapi' and fastapi_available:
            logger.info("Instrumenting FastAPI application")
            FastAPIInstrumentor.instrument_app(app)
        else:
            logger.warning(f"No instrumentation available for framework: {framework_module}")

        # Instrument SQLAlchemy if engine is provided
        if engine and sqlalchemy_available:
            logger.info("Instrumenting SQLAlchemy")
            SQLAlchemyInstrumentor().instrument(engine=engine)

        # Instrument Redis client if available
        if redis_available:
            logger.info("Instrumenting Redis client")
            RedisInstrumentor().instrument()

        # Instrument HTTP client if available
        if requests_available:
            logger.info("Instrumenting Requests client")
            RequestsInstrumentor().instrument()

        logger.info("Tracing setup complete")
        return tracer_provider

    except ImportError as e:
        logger.warning(f"OpenTelemetry setup failed due to missing dependencies: {str(e)}")
        logger.warning("Install opentelemetry-api, opentelemetry-sdk and instrumentation packages")
        return None
    except Exception as e:
        logger.error(f"Failed to set up tracing: {str(e)}")
        return None
