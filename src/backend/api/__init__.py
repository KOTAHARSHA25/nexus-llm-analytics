"""Nexus LLM Analytics — REST API Package
=========================================

FastAPI router modules for all public HTTP and WebSocket endpoints.

Submodules
----------
``analyze``      Primary query processing (sync + SSE streaming).
``upload``       Secure multi-format file ingestion.
``report``       PDF / enhanced report generation and download.
``visualize``    Chart generation and code-based visualization.
``viz_enhance``  LIDA-inspired visualization enhancement.
``models``       Ollama model inventory and selection.
``health``       Readiness / liveness probes and system status.
``history``      Code-execution history retrieval.
``feedback``     User feedback collection.
"""
