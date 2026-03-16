"""Backend Utilities Package — Nexus LLM Analytics
===================================================

Consolidated data-processing and optimisation utilities.

Submodules
----------
data_utils
    DataFrame operations, path resolution, code cleaning.
data_optimizer
    LLM-optimised data preparation (flattening, sampling).

v2.0 Enterprise Additions
-------------------------
* Enterprise module docstring with submodule catalogue.
"""

from __future__ import annotations

from .data_utils import (
    clean_column_name,
    clean_column_names,
    read_dataframe,
    get_column_properties,
    create_data_summary,
    validate_dataframe,
    clean_code_snippet,
    infer_data_types,
    DataPathResolver,
    preprocess_visualization_code
)

from .data_optimizer import (
    DataOptimizer
)

__all__ = [
    # Data utilities
    'clean_column_name',
    'clean_column_names', 
    'read_dataframe',
    'get_column_properties',
    'create_data_summary',
    'validate_dataframe',
    'clean_code_snippet',
    'infer_data_types',
    'DataPathResolver',
    'preprocess_visualization_code',
    # Data optimizer
    'DataOptimizer'
]
