import json
from pathlib import Path
from src.backend.utils.data_optimizer import DataOptimizer

# Load simple.json
filepath = Path("data/samples/simple.json")
with open(filepath, 'r') as f:
    data = json.load(f)

print("=" * 60)
print("ORIGINAL DATA:")
print(json.dumps(data, indent=2))
print()

optimizer = DataOptimizer()

# Test _is_nested
is_nested = optimizer._is_nested(data)
print(f"Is nested: {is_nested}")
print()

# Test _flatten_nested_json
flattened = optimizer._flatten_nested_json(data)
print("FLATTENED DATA:")
print(f"Type: {type(flattened)}")
print(f"Content: {flattened}")
