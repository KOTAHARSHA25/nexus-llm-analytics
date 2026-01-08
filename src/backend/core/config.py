# Centralized Configuration Module
# Single source of truth for all application settings

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import json
import logging

class Settings(BaseSettings):
    """Application settings with validation and defaults"""
    
    # Define Project Root (src/backend/core/config.py -> core -> backend -> src -> root)
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application metadata
    app_name: str = "Nexus LLM Analytics"
    app_version: str = "1.0.0"
    app_description: str = "Multi-Agent Data Analysis Assistant"
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    reload: bool = Field(default=True, env="RELOAD")
    
    # CORS settings - use str and parse in validator
    cors_allowed_origins: Union[str, List[str]] = Field(
        default="*",
        env="CORS_ALLOWED_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: Union[str, List[str]] = Field(default="*", env="CORS_ALLOW_METHODS")
    cors_allow_headers: Union[str, List[str]] = Field(default="*", env="CORS_ALLOW_HEADERS")
    
    # File upload settings
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    allowed_file_extensions: Union[str, List[str]] = Field(
        default="csv,json,txt,pdf,xlsx,xls",
        env="ALLOWED_FILE_EXTENSIONS"
    )
    upload_directory: str = Field(default=str(PROJECT_ROOT / "data" / "uploads"), env="UPLOAD_DIRECTORY")
    
    # Database settings
    chromadb_persist_directory: str = Field(
        default=str(PROJECT_ROOT / "chroma_db"),
        env="CHROMADB_PERSIST_DIRECTORY"
    )
    chromadb_collection_name: str = Field(
        default="nexus_documents",
        env="CHROMADB_COLLECTION_NAME"
    )
    
    # Ollama/LLM settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        env="OLLAMA_BASE_URL"
    )
    ollama_api_key: str = Field(default="not-needed", env="OLLAMA_API_KEY")
    primary_model: str = Field(default="ollama/llama3.1:8b", env="PRIMARY_MODEL")
    review_model: str = Field(default="ollama/phi3:mini", env="REVIEW_MODEL")
    embedding_model: str = Field(default="ollama/nomic-embed-text", env="EMBEDDING_MODEL")
    auto_model_selection: bool = Field(default=True, env="AUTO_MODEL_SELECTION")
    
    # User preference model settings (from .env file)
    preferred_primary_model: Optional[str] = Field(default=None, env="PREFERRED_PRIMARY_MODEL")
    preferred_review_model: Optional[str] = Field(default=None, env="PREFERRED_REVIEW_MODEL")
    
    # Memory management settings
    allow_swap_usage: bool = Field(default=False, env="ALLOW_SWAP_USAGE")
    memory_buffer_gb: float = Field(default=0.5, env="MEMORY_BUFFER_GB")
    
    # Security settings
    enable_code_sandbox: bool = Field(default=True, env="ENABLE_CODE_SANDBOX")
    sandbox_timeout: int = Field(default=60, env="SANDBOX_TIMEOUT")  # seconds
    sandbox_max_memory_mb: int = Field(default=256, env="SANDBOX_MAX_MEMORY_MB")
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    
    # Rate limiting settings
    rate_limit_requests_per_minute: int = Field(default=60, env="RATE_LIMIT_RPM")
    rate_limit_requests_per_hour: int = Field(default=1000, env="RATE_LIMIT_RPH")
    rate_limit_burst_size: int = Field(default=10, env="RATE_LIMIT_BURST")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=str(PROJECT_ROOT / "logs" / "nexus.log"), env="LOG_FILE")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    log_max_bytes: int = Field(default=10485760, env="LOG_MAX_BYTES")  # 10MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # Cache settings
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    cache_max_size: int = Field(default=100, env="CACHE_MAX_SIZE")
    
    # Performance settings
    max_concurrent_analyses: int = Field(default=3, env="MAX_CONCURRENT_ANALYSES")
    analysis_timeout_seconds: int = Field(default=600, env="ANALYSIS_TIMEOUT")  # 10 minutes
    enable_memory_optimization: bool = Field(default=True, env="ENABLE_MEMORY_OPTIMIZATION")
    
    # LLM timeout settings
    llm_timeout_seconds: int = Field(default=1200, env="LLM_TIMEOUT")  # 20 minutes
    llm_max_retries: int = Field(default=3, env="LLM_MAX_RETRIES")
    
    # Report generation settings
    report_template_path: str = Field(default="templates/report.html", env="REPORT_TEMPLATE")
    reports_directory: str = Field(default=str(PROJECT_ROOT / "reports"), env="REPORTS_DIRECTORY")
    max_report_size_mb: int = Field(default=50, env="MAX_REPORT_SIZE_MB")
    
    # API documentation
    docs_url: str = Field(default="/docs", env="DOCS_URL")
    redoc_url: str = Field(default="/redoc", env="REDOC_URL")
    openapi_url: str = Field(default="/openapi.json", env="OPENAPI_URL")
    
    # Feature flags
    enable_websockets: bool = Field(default=False, env="ENABLE_WEBSOCKETS")
    enable_export_excel: bool = Field(default=True, env="ENABLE_EXPORT_EXCEL")
    enable_export_pdf: bool = Field(default=True, env="ENABLE_EXPORT_PDF")
    enable_advanced_visualizations: bool = Field(default=True, env="ENABLE_ADV_VIZ")
    
    # Polars settings
    polars_skip_cpu_check: bool = Field(default=True, env="POLARS_SKIP_CPU_CHECK")
    
    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("allowed_file_extensions", mode="before")
    @classmethod
    def parse_file_extensions(cls, v):
        """Parse file extensions from string or list"""
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return v
    
    @field_validator("cors_allow_methods", "cors_allow_headers", mode="before")
    @classmethod
    def parse_cors_list(cls, v):
        """Parse CORS methods/headers from string or list"""
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [item.strip() for item in v.split(",")]
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()
    
    def get_upload_path(self) -> Path:
        """Get upload directory path"""
        path = Path(self.upload_directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_reports_path(self) -> Path:
        """Get reports directory path"""
        path = Path(self.reports_directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_chromadb_path(self) -> Path:
        """Get ChromaDB directory path"""
        path = Path(self.chromadb_persist_directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_log_path(self) -> Optional[Path]:
        """Get log file path"""
        if self.log_file:
            path = Path(self.log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return self.dict()
    
    def get_safe_dict(self) -> Dict[str, Any]:
        """Get settings dictionary with sensitive values masked"""
        data = self.dict()
        sensitive_keys = ["api_key", "password", "secret", "token"]
        
        for key in data:
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                data[key] = "***MASKED***"
        
        return data
    
    def setup_logging(self):
        """Configure standard Python logging based on settings."""
        import logging.handlers
        
        # Get or create root logger - Set to WARNING to suppress library noise
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler - Use minimal format for clean, attractive output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level)) # Allow configured level (INFO/DEBUG)
        
        try:
            import colorlog
            # Define colors for different levels
            console_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(message)s",
                log_colors={
                    'DEBUG':    'cyan',
                    'INFO':     'green',
                    'WARNING':  'yellow',
                    'ERROR':    'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        except ImportError:
            # Fallback if colorlog is missing (though it is in requirements)
            console_formatter = logging.Formatter("%(message)s")
            
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler if log file specified
        if self.log_file:
            log_path = self.get_log_path()
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=self.log_max_bytes,
                backupCount=self.log_backup_count,
                encoding='utf-8' # Force UTF-8 for file logs too
            )
            file_handler.setLevel(getattr(logging, self.log_level))
            file_formatter = logging.Formatter(self.log_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        # Silence specific noisy libraries
        for noisy_lib in ["uvicorn", "uvicorn.access", "watchfiles", "multipart", "httpcore", "httpx"]:
            logging.getLogger(noisy_lib).setLevel(logging.WARNING)
            
        # Explicitly set backend logger to configured level (INFO/DEBUG)
        # This ensures our app logs show up even if root is WARNING
        logging.getLogger("backend").setLevel(getattr(logging, self.log_level))
        logging.getLogger("src.backend").setLevel(getattr(logging, self.log_level))
        
        logging.info(f"Logging configured: level={self.log_level}, file={self.log_file}")
    
    def validate_environment(self) -> List[str]:
        """Validate environment and return any warnings"""
        warnings = []
        
        # Check Ollama connection (optional service)
        import requests
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                # Don't add warning - Ollama is optional
                logging.debug(f"Ollama API not accessible at {self.ollama_base_url}")
        except Exception as e:
            # Don't add warning - Ollama is optional during development
            logging.debug(f"Cannot connect to Ollama at {self.ollama_base_url}: {e}")
        
        # Check directories
        directories = [
            self.upload_directory,
            self.reports_directory,
            self.chromadb_persist_directory
        ]
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                warnings.append(f"Cannot create directory {directory}: {e}")
        
        # Check memory settings
        import psutil
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        if available_ram_gb < 2:
            warnings.append(f"Low memory warning: Only {available_ram_gb:.1f}GB available")
        
        return warnings

# Create singleton settings instance
@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    
    # Set environment variable for Polars
    if settings.polars_skip_cpu_check:
        os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
    
    return settings

# Export commonly used settings
settings = get_settings()

# Configuration presets for different environments
ENVIRONMENT_PRESETS = {
    "development": {
        "debug": True,
        "reload": True,
        "log_level": "DEBUG",
        "workers": 1
    },
    "production": {
        "debug": False,
        "reload": False,
        "log_level": "INFO",
        "workers": 4,
        "enable_rate_limiting": True
    },
    "testing": {
        "debug": True,
        "reload": False,
        "log_level": "DEBUG",
        "workers": 1,
        "enable_rate_limiting": False
    }
}

def apply_environment_preset(environment: str):
    """Apply environment-specific settings"""
    if environment in ENVIRONMENT_PRESETS:
        preset = ENVIRONMENT_PRESETS[environment]
        for key, value in preset.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        logging.info(f"Applied {environment} environment preset")

# Validation helper
def validate_config() -> bool:
    """Validate configuration and return True if valid"""
    try:
        settings = get_settings()
        warnings = settings.validate_environment()
        
        if warnings:
            for warning in warnings:
                # Use INFO level for environment warnings (not critical errors)
                logging.info(f"ℹ️ Environment note: {warning}")
        
        logging.info("Configuration validation completed")
        # Return True even with warnings - they're informational only
        return True
    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        return False
