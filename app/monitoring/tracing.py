# app/monitoring/tracing.py
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer = None


def setup_tracing(app=None, db_engine=None):
    """
    Set up distributed tracing
    
    Args:
        app: The FastAPI or Flask application
        db_engine: SQLAlchemy engine for database tracing
        
    Returns:
        Tracer object
    """
    global _tracer
    
    try:
        # Try to use OpenTelemetry if available
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        
        # Get service name from environment or use default
        service_name = os.getenv("MONITORING_SERVICE_NAME", "desi-hiphop-recommendation")
        
        # Create resource
        resource = Resource.create({"service.name": service_name})
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Set up exporter if OTLP endpoint is configured
        otlp_endpoint = os.getenv("OTLP_ENDPOINT")
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
            logger.info(f"OpenTelemetry tracing configured with OTLP exporter to {otlp_endpoint}")
        
        # Create tracer
        _tracer = trace.get_tracer(__name__)
        
        # Instrument FastAPI if available
        if app and "fastapi" in str(type(app)):
            try:
                from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
                # Fix: Create an instance of the instrumentor before calling instrument_app
                instrumentor = FastAPIInstrumentor()
                instrumentor.instrument_app(app)
                logger.info("FastAPI instrumented for tracing")
            except ImportError:
                logger.warning("OpenTelemetry FastAPI instrumentation not available")
            except Exception as e:
                logger.error(f"Error instrumenting FastAPI: {str(e)}")
        
        # Instrument Flask if available
        elif app and "flask" in str(type(app)):
            try:
                from opentelemetry.instrumentation.flask import FlaskInstrumentor
                # Fix: Create an instance of the instrumentor before calling instrument_app
                instrumentor = FlaskInstrumentor()
                instrumentor.instrument_app(app)
                logger.info("Flask instrumented for tracing")
            except ImportError:
                logger.warning("OpenTelemetry Flask instrumentation not available")
            except Exception as e:
                logger.error(f"Error instrumenting Flask: {str(e)}")
        
        # Instrument SQLAlchemy if available
        if db_engine:
            try:
                from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
                # Fix: Create an instance of the instrumentor before calling instrument
                instrumentor = SQLAlchemyInstrumentor()
                instrumentor.instrument(engine=db_engine)
                logger.info("SQLAlchemy instrumented for tracing")
            except ImportError:
                logger.warning("OpenTelemetry SQLAlchemy instrumentation not available")
            except Exception as e:
                logger.error(f"Error instrumenting SQLAlchemy: {str(e)}")
        
        logger.info("Distributed tracing initialized")
        return _tracer
    
    except ImportError:
        logger.warning("OpenTelemetry not installed. Tracing disabled. Install with: pip install opentelemetry-api opentelemetry-sdk")
        
        # Return dummy tracer that does nothing
        class DummyTracer:
            def start_as_current_span(self, name, *args, **kwargs):
                class DummySpan:
                    def __enter__(self): return self
                    def __exit__(self, *args, **kwargs): pass
                    def add_event(self, *args, **kwargs): pass
                    def set_attribute(self, *args, **kwargs): pass
                    def set_attributes(self, *args, **kwargs): pass
                return DummySpan()
            
            def start_span(self, *args, **kwargs):
                return self.start_as_current_span(*args, **kwargs)
        
        _tracer = DummyTracer()
        return _tracer
    except Exception as e:
        logger.error(f"Error setting up tracing: {str(e)}")
        return None


def get_tracer():
    """Get the global tracer instance"""
    if _tracer is None:
        return setup_tracing()
    return _tracer


def trace_function(func):
    """Decorator to trace a function"""
    def wrapper(*args, **kwargs):
        tracer = get_tracer()
        with tracer.start_as_current_span(func.__name__):
            return func(*args, **kwargs)
    return wrapper
