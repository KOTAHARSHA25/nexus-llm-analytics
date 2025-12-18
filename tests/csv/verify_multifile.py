import pandas as pd

c = pd.read_csv('data/samples/csv/customers.csv')
o = pd.read_csv('data/samples/csv/orders.csv')
m = o.merge(c, on='customer_id')

print('Merged data:')
print(f'  Total rows: {len(m)}')
print(f'  Total orders value: ${m["amount"].sum():.2f}')
print()

print('By Country:')
by_country = m.groupby('country')['amount'].sum().sort_values(ascending=False)
for country, total in by_country.items():
    print(f'  {country}: ${total:.2f}')

print()
print('By City:')
by_city = m.groupby('city')['amount'].agg(['sum', 'count']).sort_values('sum', ascending=False)
for city, row in by_city.iterrows():
    print(f'  {city}: ${row["sum"]:.2f} ({int(row["count"])} orders)')
