
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

from backend.utils.data_utils import DataPathResolver

print(f"Project Root: {DataPathResolver.get_project_root()}")
print(f"Samples Dir: {DataPathResolver.get_samples_dir()}")
print(f"Uploads Dir: {DataPathResolver.get_uploads_dir()}")

resolved = DataPathResolver.resolve_data_file("sales_data.csv")
print(f"Resolve 'sales_data.csv': {resolved}")
print(f"Exists? {resolved.exists() if resolved else 'N/A'}")
