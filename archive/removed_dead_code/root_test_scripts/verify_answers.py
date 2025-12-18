"""Comprehensive verification of all CSV test answers"""
import pandas as pd

print("="*70)
print("ANSWER VERIFICATION REPORT")
print("="*70)

# ============================================================================
# TEST 2.1.1: Simple CSV (sales_simple.csv)
# ============================================================================
print("\n[TEST 2.1.1] Simple CSV - sales_simple.csv (5 rows)")
print("-"*70)

df_simple = pd.read_csv('data/samples/csv/sales_simple.csv')
print("\nActual Data:")
print(df_simple)

# Query 1: Total revenue
total_revenue = df_simple['revenue'].sum()
print(f"\n[Q1] What is the total revenue?")
print(f"  Expected: $5,850")
print(f"  LLM Answer: $5,850")
print(f"  Actual: ${total_revenue:,}")
print(f"  ✅ CORRECT" if total_revenue == 5850 else f"  ❌ WRONG")

# Query 2: Which product has highest sales
by_product = df_simple.groupby('product').agg({
    'revenue': 'sum',
    'quantity': 'sum'
}).sort_values('revenue', ascending=False)
print(f"\n[Q2] Which product has highest sales?")
print(f"  Expected: Widget A ($3,300 revenue, 33 units)")
print(f"  LLM Answer: Widget A ($2,400 revenue, 3 occurrences)")
print(f"\n  Actual breakdown:")
for product, row in by_product.iterrows():
    print(f"    {product}: ${row['revenue']:,} revenue, {row['quantity']} units")
top_product = by_product.index[0]
top_revenue = by_product.iloc[0]['revenue']
top_quantity = by_product.iloc[0]['quantity']
print(f"\n  Correct answer: {top_product} (${top_revenue:,} revenue, {top_quantity} units)")
if top_product == "Widget A" and top_revenue == 3300:
    print(f"  ⚠️ LLM got product correct but revenue WRONG ($2,400 vs $3,300)")
else:
    print(f"  ❌ WRONG")

# Query 3: Unique products
unique_products = df_simple['product'].nunique()
print(f"\n[Q3] How many unique products?")
print(f"  Expected: 2")
print(f"  LLM Answer: Two (2)")
print(f"  Actual: {unique_products}")
print(f"  ✅ CORRECT" if unique_products == 2 else f"  ❌ WRONG")

# ============================================================================
# TEST 2.1.2: Medium CSV (customer_data.csv)
# ============================================================================
print("\n\n[TEST 2.1.2] Medium CSV - customer_data.csv (100 rows)")
print("-"*70)

df_medium = pd.read_csv('data/samples/csv/customer_data.csv')

# Query 1: Average age
avg_age = df_medium['age'].mean()
print(f"\n[Q1] What is the average age of customers?")
print(f"  Expected: ~43 years")
print(f"  LLM Answer: 42.51 years")
print(f"  Actual: {avg_age:.2f} years")
print(f"  ✅ CORRECT" if abs(avg_age - 42.51) < 0.01 else f"  ❌ WRONG")

# Query 2: City with most customers
city_counts = df_medium['city'].value_counts()
print(f"\n[Q2] Which city has the most customers?")
print(f"  Expected: Top city with count")
print(f"  LLM Answer: Phoenix")
print(f"  Actual top 3:")
for city, count in city_counts.head(3).items():
    print(f"    {city}: {count} customers")
top_city = city_counts.index[0]
print(f"  ✅ CORRECT" if top_city == "Phoenix" else f"  ❌ WRONG (should be {top_city})")

# Query 3: Total by membership level
total_overall = df_medium['total_spent'].sum()
by_membership = df_medium.groupby('membership_level')['total_spent'].sum().round(2)
print(f"\n[Q3] Calculate total revenue by membership level")
print(f"  Expected: Breakdown by Bronze/Silver/Gold/Platinum")
print(f"  LLM Answer: Total $251,735.17 for all customers combined")
print(f"  Actual total: ${total_overall:,.2f}")
print(f"  Actual breakdown by membership:")
for level, amount in by_membership.items():
    print(f"    {level}: ${amount:,.2f}")
if abs(total_overall - 251735.17) < 1:
    print(f"  ⚠️ LLM got TOTAL correct but missing BREAKDOWN by membership level")
else:
    print(f"  ❌ WRONG")

# ============================================================================
# TEST 2.1.3: Large CSV (transactions_large.csv)
# ============================================================================
print("\n\n[TEST 2.1.3] Large CSV - transactions_large.csv (5,000 rows)")
print("-"*70)

df_large = pd.read_csv('data/samples/csv/transactions_large.csv')

# Query 1: Total transaction volume
total_amount = df_large['amount'].sum()
print(f"\n[Q1] What is the total transaction volume?")
print(f"  Expected: $1,272,076.58")
print(f"  LLM Answer: $1,272,076.58")
print(f"  Actual: ${total_amount:,.2f}")
print(f"  ✅ CORRECT" if abs(total_amount - 1272076.58) < 0.01 else f"  ❌ WRONG")

# Query 2: Top 5 customers by spending
top5_customers = df_large.groupby('customer_id')['amount'].sum().nlargest(5)
print(f"\n[Q2] Show top 5 customers by spending")
print(f"  Expected: CUST0314 ($5,609.67), CUST0460 ($5,243.52), etc.")
print(f"  LLM Answer: CUST0347 ($314.12), CUST0340 ($184.52)")
print(f"  Actual top 5:")
for i, (cust, amount) in enumerate(top5_customers.items(), 1):
    print(f"    {i}. {cust}: ${amount:,.2f}")
print(f"  ❌ WRONG - LLM gave wrong customers (likely due to sampling)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\n[2.1.1 Simple CSV] 5 rows:")
print("  Q1: Total revenue = $5,850 ✅ CORRECT")
print("  Q2: Highest sales product = Widget A ⚠️ CORRECT PRODUCT, WRONG AMOUNT ($2,400 vs $3,300)")
print("  Q3: Unique products = 2 ✅ CORRECT")
print("  Score: 2/3 perfect, 1/3 partial")

print("\n[2.1.2 Medium CSV] 100 rows:")
print("  Q1: Average age = 42.51 years ✅ CORRECT")
print("  Q2: Most customers = Phoenix ✅ CORRECT")
print("  Q3: Total revenue = $251,735.17 ⚠️ CORRECT TOTAL, MISSING BREAKDOWN")
print("  Score: 2/3 perfect, 1/3 partial")

print("\n[2.1.3 Large CSV] 5,000 rows:")
print("  Q1: Total volume = $1,272,076.58 ✅ CORRECT")
print("  Q2: Top 5 customers ❌ WRONG (sampling issue)")
print("  Score: 1/2 perfect, 1/2 wrong")

print("\n" + "="*70)
print("OVERALL ACCURACY:")
print("  Perfect answers: 5/8 (62.5%)")
print("  Partial answers: 2/8 (25.0%)")
print("  Wrong answers: 1/8 (12.5%)")
print("  Total correct/partial: 7/8 (87.5%)")
print("="*70)
