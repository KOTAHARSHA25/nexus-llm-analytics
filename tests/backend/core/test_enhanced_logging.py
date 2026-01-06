import logging
import json
import pytest
from io import StringIO
from pathlib import Path
from src.backend.core.enhanced_logging import JsonFormatter, ColoredFormatter, setup_enhanced_logging

def test_json_formatter():
    formatter = JsonFormatter()
    record = logging.LogRecord("test", logging.INFO, "path/to/file.py", 10, "Test Message", None, None)
    log_output = formatter.format(record)
    
    data = json.loads(log_output)
    assert data["message"] == "Test Message"
    assert data["level"] == "INFO"
    assert data["logger"] == "test"
    assert "timestamp" in data

def test_json_formatter_exception():
    formatter = JsonFormatter()
    try:
        raise ValueError("Oops")
    except ValueError:
        exc_info = import_sys_exc_info()
        record = logging.LogRecord("test", logging.ERROR, "path.py", 10, "Error Msg", None, exc_info)
        log_output = formatter.format(record)
        data = json.loads(log_output)
        assert "exception" in data
        assert "ValueError: Oops" in data["exception"]

def import_sys_exc_info():
    import sys
    return sys.exc_info()

def test_colored_formatter():
    formatter = ColoredFormatter('%(levelname)s: %(message)s')
    record = logging.LogRecord("test", logging.INFO, "path.py", 10, "Info Message", None, None)
    output = formatter.format(record)
    # Check for color codes
    assert '\033[' in output
    assert "Info Message" in output

def test_setup_enhanced_logging():
    setup_enhanced_logging(level="DEBUG", use_colors=False)
    
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) >= 1
    assert isinstance(root.handlers[0], logging.StreamHandler)
    
    # Test with file
    tmp_path = Path("test_log.log")
    try:
        setup_enhanced_logging(level="INFO", log_file=tmp_path)
        assert len(root.handlers) >= 2 # Stream + File
        # Check if file handler is present
        file_handlers = [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(file_handlers) > 0
    finally:
        # Cleanup
        for h in root.handlers:
            h.close()
        root.handlers.clear()
        if tmp_path.exists():
            tmp_path.unlink()
