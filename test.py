# Setup logging before any tests
from backend.core.utils import setup_logging
setup_logging()


# Test the full review-and-execute flow using ControllerAgent and a sample JSON file
from backend.agents.controller_agent import ControllerAgent

controller = ControllerAgent()

# Use a sample JSON file from the data folder (e.g., 'analyze.json')
filename = 'analyze.json'

print("\n--- Test: summarize ---")
result = controller.handle_query('summarize', filename=filename)
print(result)

print("\n--- Test: describe ---")
result = controller.handle_query('describe', filename=filename)
print(result)

print("\n--- Test: value_counts (for first available column) ---")
import pandas as pd, os
try:
	# Use the correct backend/data/ path
	data_path = os.path.join(os.path.dirname(__file__), 'backend', 'data', filename)
	df = pd.read_json(data_path)
	columns = list(df.columns)
	if columns:
		col = columns[0]
		result = controller.handle_query('value_counts', filename=filename, column=col)
		print(f"Column used: {col}")
		print(result)
	else:
		print("No columns found in sample file for value_counts test.")
except Exception as e:
	print(f"Error loading file for value_counts test: {e}")