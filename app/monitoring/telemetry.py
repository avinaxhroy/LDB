# app/monitoring/telemetry.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource


def setup_telemetry(app, engine=None):
    """Configure OpenTelemetry instrumentation"""
    resource = Resource(attributes={
        SERVICE_NAME: "desi-hiphop-recommendation"
    })

    # Set up tracer provider
    tracer_provider = TracerProvider(resource=resource)

    # Use console exporter for simplified setup
    # In production, replace with proper backend exporter
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    tracer_provider.add_span_processor(processor)

    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

    # Ensure proper initialization of SQLAlchemyInstrumentor
    if engine:
        SQLAlchemyInstrumentor().instrument(engine=engine)

    return tracer_provider
