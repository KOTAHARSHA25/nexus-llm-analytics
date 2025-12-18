from src.backend.utils.data_optimizer import DataOptimizer

opt = DataOptimizer()
result = opt.optimize_for_llm('data/samples/sales_timeseries.json')

lines = result['preview'].split('\n')

print("Looking for SALES_AMOUNT lines with 'Total' or 'Sum':\n")
for i, line in enumerate(lines):
    if 'SALES_AMOUNT' in line.upper():
        print(f"Line {i}: {line}")
        # Print next 5 lines
        for j in range(i+1, min(i+6, len(lines))):
            if lines[j].strip():
                print(f"  {lines[j]}")
        print()
