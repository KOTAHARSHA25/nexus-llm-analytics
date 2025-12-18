import pandas as pd
df = pd.read_csv('data/samples/csv/transactions_large.csv')
print(f'Total rows: {len(df)}')
print(f'Total amount: ${df["amount"].sum():,.2f}')
print(f'Average transaction: ${df["amount"].mean():.2f}')
print(f'\nTop 5 customers by spending:')
top5 = df.groupby('customer_id')["amount"].sum().nlargest(5)
for cust, amt in top5.items():
    print(f'  {cust}: ${amt:,.2f}')
