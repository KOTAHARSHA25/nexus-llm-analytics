"""Input / Output Components — Nexus LLM Analytics
==================================================

Public surface for code generation, result interpretation,
chain-of-thought parsing, and enterprise PDF report generation.

Modules
-------
code_generator
    LLM-driven Python code generation with sandbox execution.
cot_parser
    Chain-of-Thought extraction and critic feedback parsing.
pdf_generator
    Enterprise-grade ReportLab PDF report builder.
result_interpreter
    Domain-agnostic, human-readable result formatting.

v2.0 Enterprise Additions
-------------------------
* Lazy ``__getattr__`` imports — submodules loaded on first access.
* Expanded ``__all__`` for explicit public API.
"""
from __future__ import annotations

__all__ = [
    "code_generator",
    "cot_parser",
    "pdf_generator",
    "result_interpreter",
]


def __getattr__(name: str):
    """Lazy-import submodules on first attribute access."""
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
