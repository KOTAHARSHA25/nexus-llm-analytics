
import sys
from pathlib import Path
import os

# Add src to path exactly as tests do
src_path = Path(os.getcwd()) / "src"
sys.path.insert(0, str(src_path))
print(f"Path inserted: {src_path}")

try:
    from backend.infra.circuit_breaker import CircuitBreaker
    print("✅ Successfully imported CircuitBreaker from backend.infra")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
