# app/monitoring/tracing.py
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer = None


def setup_tracing(app=None, db_engine=None):
    """
    Setup distributed tracing using OpenTelemetry for the application.
    
    This function attempts to set up OpenTelemetry tracing for the application
    and database connections if the required packages are available.
    If packages are not available, it returns a dummy tracer that provides
    the same interface without actually doing anything.
    
    Args:
        app: The Flask or FastAPI application to instrument
        db_engine: The SQLAlchemy database engine to instrument
        
    Returns:
        tracer: OpenTelemetry tracer or dummy tracer
    """
    try:
        # Try to import OpenTelemetry packages
        import opentelemetry
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        import os
        
        # Setup resource with service info
        resource = Resource.create({
            "service.name": os.getenv("MONITORING_SERVICE_NAME", "desi-hiphop-recommendation"),
            "service.version": os.getenv("APP_VERSION", "1.0.0"),
            "deployment.environment": os.getenv("ENVIRONMENT", "production")
        })
        
        # Create tracer provider with the resource
        tracer_provider = TracerProvider(resource=resource)
        
        # Configure OTLP exporter if endpoint is provided
        otlp_endpoint = os.getenv("OTLP_ENDPOINT")
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        
        # Register the tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Get a tracer
        tracer = trace.get_tracer(__name__)
        
        # Instrument Flask if app is Flask
        if app and hasattr(app, 'wsgi_app'):
            try:
                from opentelemetry.instrumentation.flask import FlaskInstrumentor
                FlaskInstrumentor().instrument_app(app)
                logger.info("Flask app instrumented with OpenTelemetry")
            except ImportError:
                logger.warning("Flask instrumentation not available")
        
        # Instrument FastAPI if app is FastAPI
        elif app and hasattr(app, 'router'):
            try:
                from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
                FastAPIInstrumentor.instrument_app(app)
                logger.info("FastAPI app instrumented with OpenTelemetry")
            except ImportError:
                logger.warning("FastAPI instrumentation not available")
        
        # Instrument SQLAlchemy if engine is provided
        if db_engine:
            try:
                from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
                SQLAlchemyInstrumentor().instrument(engine=db_engine)
                logger.info("SQLAlchemy engine instrumented with OpenTelemetry")
            except ImportError:
                logger.warning("SQLAlchemy instrumentation not available")
        
        logger.info("OpenTelemetry tracing setup successfully")
        return tracer
    
    except ImportError:
        logger.warning("OpenTelemetry packages not found, using dummy tracer")
        # Create a dummy tracer to avoid errors when tracing is not available
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
        
        return DummyTracer()

    except Exception as e:
        logger.error(f"Error setting up tracing: {str(e)}")
        # Return dummy tracer on error to avoid application failures
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
        
        return DummyTracer()


def get_tracer():
    """Get the global tracer instance"""
    if _tracer is None:
        return setup_tracing()
    return _tracer


def trace_function(tracer=None):
    """
    Decorator to trace a function with OpenTelemetry.
    
    Args:
        tracer: OpenTelemetry tracer (optional)
        
    Returns:
        Decorator function
    """
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get tracer directly if not provided
            nonlocal tracer
            if tracer is None:
                try:
                    from opentelemetry import trace
                    tracer = trace.get_tracer(__name__)
                except ImportError:
                    # Just return the original function if tracing not available
                    return func(*args, **kwargs)
            
            # Create the span
            with tracer.start_as_current_span(func.__name__):
                return func(*args, **kwargs)
                
        return wrapper
    
    return decorator
