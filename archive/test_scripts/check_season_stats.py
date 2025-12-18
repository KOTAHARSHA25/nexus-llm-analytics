from src.backend.utils.data_optimizer import DataOptimizer
import json

opt = DataOptimizer()
result = opt.optimize_for_llm('data/samples/sales_timeseries.json')

# Load actual data
data = json.load(open('data/samples/sales_timeseries.json'))
total_sales = sum(e['sales_amount'] for e in data)

print(f'Actual total sales: ${total_sales:,.2f}')
print(f'Total rows: {result["total_rows"]}')
print(f'Sample sent to LLM: {len(result.get("sample", []))} rows')
print(f'\nSEASON in preview: {"SEASON" in result["preview"].upper()}')
print(f'SALES_AMOUNT in preview: {"SALES_AMOUNT" in result["preview"].upper()}')
print(f'GROUPED AGGREGATIONS in preview: {"GROUPED AGGREGATIONS" in result["preview"]}')

# Check if seasonal grouping exists
if "SEASON" in result["preview"].upper():
    print("\n✅ Season column IS in the preview")
    # Find the SEASON section
    lines = result["preview"].split('\n')
    for i, line in enumerate(lines):
        if 'SEASON' in line.upper() and 'by' in line:
            print(f"\nFound: {line}")
            # Print next 20 lines
            for j in range(i+1, min(i+21, len(lines))):
                print(lines[j])
            break
else:
    print("\n❌ Season column NOT in preview - this is the problem!")
