"""Direct test of ML capabilities using AnalysisService"""
import sys
import os
sys.path.insert(0, 'src')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import pandas as pd
import asyncio
from backend.services.analysis_service import get_analysis_service

async def main():
    # Load test data
    df = pd.read_csv('data/samples/sales_data.csv')
    print(f"[OK] Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}\n")

    # Initialize service
    service = get_analysis_service()
    print("[OK] AnalysisService initialized\n")

    # Test ML queries
    test_queries = [
        ("K-means Clustering", "Perform K-means clustering with 3 clusters based on sales and revenue"),
        ("Correlation", "Find correlation between sales and revenue"),
        ("Random Forest", "Build a random forest classifier to predict if revenue > 5000 using sales and price"),
        ("Linear Regression", "Create a linear regression model to predict revenue from sales"),
        ("PCA", "Apply PCA to reduce dimensions of sales, revenue, price to 2 components")
    ]

    for name, query in test_queries:
        print(f"=== Test: {name} ===")
        print(f"Query: {query}")
        try:
            # Call async analyze method with proper context
            context = {
                'filename': 'sales_data.csv',
                'filepath': 'data/samples/sales_data.csv',
                'df': df
            }
            result = await service.analyze(query, context)
            print(f"Status: {result.get('status', 'unknown')}")
            success = result.get('status') == 'success' or result.get('success') == True
            if success:
                result_data = result.get('result', result.get('data', result.get('answer', '')))
                result_str = str(result_data)
                if len(result_str) > 200:
                    print(f"Result: {result_str[:200]}...")
                else:
                    print(f"Result: {result_str}")
                code = result.get('code')
                if code:
                    code_preview = code.replace('\n', ' ')[:150]
                    print(f"\nGenerated code: {code_preview}...")
            else:
                error = result.get('error', result.get('message', 'Unknown error'))
                print(f"Error: {error}")
        except Exception as e:
            print(f"Exception: {type(e).__name__}: {e}")
        print()

    print("[OK] All ML tests completed")

if __name__ == "__main__":
    asyncio.run(main())
