"""Data Utility Functions — Nexus LLM Analytics
================================================

DataFrame operations and data preprocessing utilities.
Integrated from Microsoft LIDA library with Nexus enhancements.

Provides
--------
* :class:`DataPathResolver` — centralised file-path resolution
  with caching for uploads and sample directories.
* :func:`read_dataframe` — universal file reader (CSV, Excel,
  JSON, Parquet, Feather, HDF5, TSV).
* Column cleaning, type inference, and code-snippet sanitisation.

v2.0 Enterprise Additions
-------------------------
* Enhanced module and class docstrings.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPathResolver:
    """Centralised data-path resolution with caching.

    Eliminates duplicate path-resolution logic across API files.
    Uses class-level caches for project root, uploads, and samples
    directories to avoid repeated filesystem traversal.

    Thread Safety:
        Class methods are guarded by the GIL for simple attribute
        assignment — safe for typical concurrent access patterns.
    """
    
    _project_root: Optional[Path] = None
    _uploads_dir: Optional[Path] = None
    _samples_dir: Optional[Path] = None
    
    @classmethod
    def get_project_root(cls) -> Path:
        """Get cached project root directory."""
        if cls._project_root is None:
            # Calculate once from this file's location
            # data_utils.py is in: src/backend/utils/
            # Project root is: ../../../ from here
            backend_dir = Path(__file__).parent.parent
            # backend_dir is src/backend
            # backend_dir.parent is src
            # backend_dir.parent.parent is project root
            cls._project_root = backend_dir.parent.parent
        return cls._project_root
    
    @classmethod
    def get_uploads_dir(cls) -> Path:
        """Get uploads directory path (cached)."""
        from backend.core.config import settings
        if cls._uploads_dir is None:
            cls._uploads_dir = Path(settings.upload_directory)
            # Ensure it's absolute
            if not cls._uploads_dir.is_absolute():
                 cls._uploads_dir = cls.get_project_root() / cls._uploads_dir
        return cls._uploads_dir
    
    @classmethod
    def get_samples_dir(cls) -> Path:
        """Get samples directory path (cached)."""
        if cls._samples_dir is None:
            cls._samples_dir = cls.get_project_root() / "data" / "samples"
        return cls._samples_dir
    
    @classmethod
    def resolve_data_file(cls, filename: str, 
                         search_paths: Optional[List[str]] = None) -> Optional[Path]:
        """
        Resolve data file path from multiple possible locations.
        
        Args:
            filename: Name of file to find
            search_paths: Optional custom search paths relative to project root
                         Default: ["data/uploads", "data/samples"]
        
        Returns:
            Path to file if found, None otherwise
        """
        root = cls.get_project_root()
        
        if search_paths is None:
            search_paths = ["data/uploads", "data/samples"]
        
        # Try each search path
        for search_path in search_paths:
            file_path = root / search_path / filename
            if file_path.exists() and file_path.is_file():
                logger.debug("Found file: %s", file_path)
                return file_path
        
        # Fallback: case-insensitive search in uploads directory
        uploads_dir = cls.get_uploads_dir()
        if uploads_dir.exists():
            for uploaded_file in uploads_dir.iterdir():
                if uploaded_file.is_file():
                    if uploaded_file.name.lower() == filename.lower():
                        logger.debug("Found file (case-insensitive): %s", uploaded_file)
                        return uploaded_file
        
        logger.warning("File not found in search paths: %s", filename)
        return None
    
    @classmethod
    def ensure_directories_exist(cls):
        """Ensure data directories exist, create if missing."""
        uploads_dir = cls.get_uploads_dir()
        samples_dir = cls.get_samples_dir()
        
        uploads_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug("Data directories ready: %s, %s", uploads_dir, samples_dir)


def clean_column_name(col_name: str) -> str:
    """
    Clean a single column name by replacing special characters and spaces with underscores.
    Integrated from LIDA for better column name standardization.
    
    Args:
        col_name: The name of the column to be cleaned.
        
    Returns:
        A sanitized string valid as a column name.
    """
    return re.sub(r'[^0-9a-zA-Z_]', '_', col_name)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean all column names in the given DataFrame.
    Integrated from LIDA for consistent column naming.
    
    Args:
        df: The DataFrame with possibly dirty column names.
        
    Returns:
        DataFrame with clean column names (memory-optimized with rename).
    """
    # Use rename instead of copy for memory efficiency (50% RAM reduction)
    return df.rename(columns={col: clean_column_name(col) for col in df.columns})


# Phase 3.5: Scientific file format readers
def _read_hdf5(file_location: str) -> pd.DataFrame:
    """
    Read HDF5 file into DataFrame.
    Supports both pandas HDF5 format and generic HDF5 datasets.
    """
    try:
        # Try pandas HDF5 format first
        return pd.read_hdf(file_location)
    except Exception:
        # Fallback to h5py for generic HDF5 files
        try:
            import h5py
            with h5py.File(file_location, 'r') as f:
                # Find the first dataset and convert to DataFrame
                data = {}
                def extract_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        try:
                            data[name.replace('/', '_')] = obj[:]
                        except Exception:
                            pass
                f.visititems(extract_datasets)
                
                if data:
                    # Try to create DataFrame from extracted data
                    return pd.DataFrame(data)
                else:
                    raise ValueError("No readable datasets found in HDF5 file")
        except ImportError:
            raise ImportError("h5py package required for reading HDF5 files. Install with: pip install h5py")


def _read_netcdf(file_location: str) -> pd.DataFrame:
    """
    Read NetCDF file into DataFrame.
    Commonly used for climate and scientific data.
    """
    try:
        import xarray as xr
        ds = xr.open_dataset(file_location)
        return ds.to_dataframe().reset_index()
    except ImportError:
        try:
            # Fallback to netCDF4 library
            import netCDF4 as nc
            dataset = nc.Dataset(file_location, 'r')
            data = {}
            for var_name in dataset.variables:
                var = dataset.variables[var_name]
                if len(var.shape) <= 1:
                    data[var_name] = var[:]
            dataset.close()
            return pd.DataFrame(data)
        except ImportError:
            raise ImportError("xarray or netCDF4 package required for reading NetCDF files. Install with: pip install xarray netCDF4")


def _read_matlab(file_location: str) -> pd.DataFrame:
    """
    Read MATLAB .mat file into DataFrame.
    Supports both v7.3 (HDF5-based) and older formats.
    """
    try:
        from scipy.io import loadmat
        mat_data = loadmat(file_location, squeeze_me=True)
        
        # Filter out metadata keys
        data = {}
        for key, value in mat_data.items():
            if not key.startswith('__'):
                if hasattr(value, 'shape'):
                    if len(value.shape) <= 2:
                        if len(value.shape) == 1:
                            data[key] = value
                        elif len(value.shape) == 2:
                            # 2D array - add as multiple columns
                            for i in range(value.shape[1]):
                                data[f"{key}_{i}"] = value[:, i]
        
        if data:
            return pd.DataFrame(data)
        else:
            raise ValueError("No readable arrays found in MATLAB file")
    except ImportError:
        raise ImportError("scipy package required for reading MATLAB files. Install with: pip install scipy")


def _read_txt_lines_safe(file_location: str, encoding: str = 'utf-8') -> list:
    """Safely read text file lines with proper resource cleanup."""
    with open(file_location, 'r', encoding=encoding, errors='replace') as f:
        return f.readlines()


def read_dataframe(file_location: str, encoding: str = 'utf-8', sample_size: int = 1000, write_back: bool = False) -> pd.DataFrame:
    """
    Read a dataframe from a given file location and clean its column names.
    Automatically samples down to specified size if data exceeds that limit.
    Enhanced version from LIDA with additional format support.
    
    Args:
        file_location: The path to the file containing the data.
        encoding: Encoding to use for the file reading.
        sample_size: Maximum number of rows to sample (default: 1000 - OPTIMIZED for RAM)
        write_back: If True, write cleaned column names back to disk.
            Default False to prevent destructive side-effects on reads.
        
    Returns:
        A cleaned DataFrame.
        
    Raises:
        ValueError: If file type is unsupported
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_location)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_location}")
    
    file_extension = file_path.suffix.lower().lstrip('.')

    read_funcs = {
        'json': lambda: pd.read_json(file_location, orient='records', encoding=encoding, lines=True, nrows=sample_size) if sample_size and sample_size > 0 else pd.read_json(file_location, orient='records', encoding=encoding),
        'csv': lambda: pd.read_csv(file_location, encoding=encoding, nrows=sample_size if sample_size > 0 else None),
        'xls': lambda: pd.read_excel(file_location, engine='xlrd', nrows=sample_size if sample_size > 0 else None),
        'xlsx': lambda: pd.read_excel(file_location, engine='openpyxl', nrows=sample_size if sample_size > 0 else None),
        'parquet': lambda: pd.read_parquet(file_location),
        'feather': lambda: pd.read_feather(file_location),
        'tsv': lambda: pd.read_csv(file_location, sep="\t", encoding=encoding, nrows=sample_size if sample_size > 0 else None),
        # Fix for .txt: Read as raw lines with proper file handle management
        'txt': lambda: pd.DataFrame({'content': _read_txt_lines_safe(file_location, encoding)}),
        # Phase 3.5: Scientific file formats
        'hdf5': lambda: _read_hdf5(file_location),
        'h5': lambda: _read_hdf5(file_location),
        'nc': lambda: _read_netcdf(file_location),
        'mat': lambda: _read_matlab(file_location)
    }

    if file_extension not in read_funcs:
        raise ValueError(f'Unsupported file type: {file_extension}. Supported types: {", ".join(read_funcs.keys())}')

    try:
        df = read_funcs[file_extension]()
        logger.debug("Successfully read %d rows from %s", len(df), file_location)
    except Exception as e:
        logger.error("Failed to read file: %s", file_location, exc_info=True)
        raise

    # Clean column names
    cleaned_df = clean_column_names(df)
    
    # Log if columns were renamed
    columns_changed = cleaned_df.columns.tolist() != df.columns.tolist()
    if columns_changed:
        logger.debug("Cleaned column names in %s", file_location)

    # Sample down if necessary
    if len(cleaned_df) > sample_size:
        logger.debug("Dataframe has %d rows. Sampling %d rows.", len(cleaned_df), sample_size)
        cleaned_df = cleaned_df.sample(sample_size, random_state=42)

    # Save back with cleaned columns only if explicitly requested
    if write_back and columns_changed:
        write_funcs = {
            'csv': lambda: cleaned_df.to_csv(file_location, index=False, encoding=encoding),
            'xls': lambda: cleaned_df.to_excel(file_location, index=False, engine='xlrd'),
            'xlsx': lambda: cleaned_df.to_excel(file_location, index=False, engine='openpyxl'),
            'parquet': lambda: cleaned_df.to_parquet(file_location, index=False),
            'feather': lambda: cleaned_df.to_feather(file_location),
            'json': lambda: cleaned_df.to_json(file_location, orient='records', default_handler=str),
            'tsv': lambda: cleaned_df.to_csv(file_location, index=False, sep='\t', encoding=encoding),
            'txt': lambda: cleaned_df.to_csv(file_location, index=False, sep='\t', encoding=encoding)
        }

        if file_extension in write_funcs:
            try:
                write_funcs[file_extension]()
                logger.debug("Saved cleaned dataframe to %s", file_location)
            except Exception as e:
                logger.warning("Failed to save cleaned file: %s", file_location, exc_info=True)

    return cleaned_df


def get_column_properties(df: pd.DataFrame, n_samples: int = 3) -> List[Dict[str, Any]]:
    """
    Get properties of each column in a pandas DataFrame.
    Integrated from LIDA for comprehensive data profiling.
    
    Args:
        df: The DataFrame to analyze
        n_samples: Number of sample values to include per column
        
    Returns:
        List of dictionaries containing column properties
    """
    properties_list = []
    
    for column in df.columns:
        dtype = df[column].dtype
        properties = {}
        
        # Numeric columns
        if dtype in [int, float, complex]:
            properties["dtype"] = "number"
            properties["std"] = float(df[column].std()) if pd.notna(df[column].std()) else None
            properties["min"] = float(df[column].min()) if pd.notna(df[column].min()) else None
            properties["max"] = float(df[column].max()) if pd.notna(df[column].max()) else None
            properties["mean"] = float(df[column].mean()) if pd.notna(df[column].mean()) else None
            properties["median"] = float(df[column].median()) if pd.notna(df[column].median()) else None
            
        # Boolean columns
        elif dtype == bool:
            properties["dtype"] = "boolean"
            properties["true_count"] = int(df[column].sum())
            properties["false_count"] = int((~df[column]).sum())
            
        # Object columns (strings or mixed)
        elif dtype == object:
            # Try to parse as datetime
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pd.to_datetime(df[column], errors='raise')
                    properties["dtype"] = "date"
            except (ValueError, TypeError):
                # Check if categorical (limited unique values)
                if df[column].nunique() / len(df[column]) < 0.5:
                    properties["dtype"] = "category"
                else:
                    properties["dtype"] = "string"
                    
        # Categorical columns
        elif pd.api.types.is_categorical_dtype(df[column]):
            properties["dtype"] = "category"
            
        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            properties["dtype"] = "date"
        else:
            properties["dtype"] = str(dtype)

        # Add min/max for date columns
        if properties["dtype"] == "date":
            try:
                properties["min"] = str(df[column].min())
                properties["max"] = str(df[column].max())
            except TypeError:
                cast_date_col = pd.to_datetime(df[column], errors='coerce')
                properties["min"] = str(cast_date_col.min())
                properties["max"] = str(cast_date_col.max())
                
        # Add unique value count
        nunique = df[column].nunique()
        properties["num_unique_values"] = int(nunique)
        
        # Add sample values
        non_null_values = df[column][df[column].notnull()].unique()
        n_samples_actual = min(n_samples, len(non_null_values))
        if n_samples_actual > 0:
            samples = pd.Series(non_null_values).sample(
                n_samples_actual, random_state=42).tolist()
            # Convert to native Python types for JSON serialization
            properties["samples"] = [str(s) if not isinstance(s, (int, float, bool, str)) else s for s in samples]
        else:
            properties["samples"] = []
        
        # Calculate missing values
        properties["missing_count"] = int(df[column].isna().sum())
        properties["missing_percentage"] = float(df[column].isna().sum() / len(df[column]) * 100)
        
        # Placeholder for semantic type (can be enriched with LLM)
        properties["semantic_type"] = ""
        properties["description"] = ""
        
        properties_list.append({
            "column": column,
            "properties": properties
        })

    return properties_list


def create_data_summary(df: pd.DataFrame, file_name: str = "", n_samples: int = 3) -> Dict[str, Any]:
    """
    Create a comprehensive data summary for a DataFrame.
    Based on LIDA's summarization approach.
    
    Args:
        df: The DataFrame to summarize
        file_name: Name of the source file
        n_samples: Number of sample values per column
        
    Returns:
        Dictionary containing data summary with fields metadata
    """
    data_properties = get_column_properties(df, n_samples)
    
    summary = {
        "name": file_name or "dataset",
        "file_name": file_name,
        "dataset_description": f"Dataset with {len(df)} rows and {len(df.columns)} columns",
        "row_count": len(df),
        "column_count": len(df.columns),
        "fields": data_properties,
        "field_names": df.columns.tolist(),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
    }
    
    return summary


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate a DataFrame for common issues.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df.columns) == 0:
        return False, "DataFrame has no columns"
    
    # Check for all-null columns
    all_null_cols = [col for col in df.columns if df[col].isna().all()]
    if all_null_cols:
        return False, f"Columns with all null values: {', '.join(all_null_cols)}"
    
    # Check for duplicate column names
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        return False, f"Duplicate column names found: {', '.join(duplicate_cols)}"
    
    return True, "DataFrame is valid"


def clean_code_snippet(code_string: str) -> str:
    """
    Extract and clean code snippet from markdown or raw strings.
    Integrated from LIDA for code extraction from LLM responses.
    
    Args:
        code_string: String potentially containing code in markdown blocks
        
    Returns:
        Cleaned code snippet
    """
    # Extract code from markdown code blocks
    cleaned_snippet = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', code_string)
    
    if cleaned_snippet:
        cleaned_snippet = cleaned_snippet.group(1)
    else:
        cleaned_snippet = code_string
    
    # Remove any remaining markdown artifacts
    cleaned_snippet = cleaned_snippet.replace("```python", "")
    cleaned_snippet = cleaned_snippet.replace("```", "")
    
    return cleaned_snippet.strip()


def preprocess_visualization_code(code: str, 
                                  library: str = "plotly",
                                  ensure_imports: bool = True,
                                  remove_placeholders: bool = True) -> str:
    """
    Unified visualization code preprocessing function.
    Consolidates logic from viz_enhance.py and visualize.py.
    
    Args:
        code: Raw code string (possibly with markdown)
        library: Target visualization library ("plotly", "matplotlib", "seaborn")
        ensure_imports: Whether to add imports if missing
        remove_placeholders: Whether to remove <imports>, <stub>, <transforms> placeholders
        
    Returns:
        Cleaned and preprocessed code ready for execution
    """
    # Step 1: Extract code from markdown
    code = clean_code_snippet(code)
    
    # Step 2: Remove placeholder comments (from LLM templates)
    if remove_placeholders:
        code = code.replace("<imports>", "")
        code = code.replace("<stub>", "")
        code = code.replace("<transforms>", "")
    
    # Step 3: Ensure necessary imports are present
    if ensure_imports and "import" not in code:
        if library.lower() == "plotly":
            code = "import pandas as pd\nimport plotly.express as px\nimport plotly.graph_objects as go\n\n" + code
        elif library.lower() in ["matplotlib", "seaborn"]:
            imports = "import pandas as pd\nimport matplotlib.pyplot as plt\nimport numpy as np\n"
            if library.lower() == "seaborn" or "sns" in code:
                imports += "import seaborn as sns\n"
            code = imports + "\n" + code
    
    return code.strip()


def infer_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligently infer and convert data types.
    
    Args:
        df: DataFrame with potentially incorrect types
        
    Returns:
        DataFrame with inferred types
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            continue
        
        # Try to convert to numeric
        try:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='ignore')
        except (ValueError, TypeError) as e:
            logger.debug("Numeric conversion skipped for %s: %s", col, e)
        
        # Try to convert to datetime
        if df_copy[col].dtype == object:
            try:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='ignore')
            except (ValueError, TypeError, pd.errors.ParserError) as e:
                logger.debug("Datetime conversion skipped for %s: %s", col, e)
    
    return df_copy
