"""
Generate large transaction dataset for testing

Creates a JSON file with 10,000 transaction records to test
large dataset handling with the data optimizer.

Date: October 19, 2025
"""

import json
import random
from datetime import datetime, timedelta

def generate_large_transactions(num_records=10000):
    """Generate large transaction dataset"""
    
    print(f"Generating {num_records} transaction records...")
    
    categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home', 'Sports', 'Toys', 'Health', 'Beauty', 'Automotive']
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Bank Transfer']
    statuses = ['Completed', 'Pending', 'Cancelled', 'Refunded']
    
    transactions = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(1, num_records + 1):
        # Random date in 2024
        days_offset = random.randint(0, 365)
        transaction_date = start_date + timedelta(days=days_offset)
        
        # Random transaction
        transaction = {
            'transaction_id': f'TXN{i:06d}',
            'date': transaction_date.strftime('%Y-%m-%d'),
            'customer_id': f'CUST{random.randint(1, 2000):04d}',
            'category': random.choice(categories),
            'product_name': f'Product {random.randint(1, 500)}',
            'quantity': random.randint(1, 10),
            'unit_price': round(random.uniform(10, 500), 2),
            'total_amount': 0,  # Will calculate
            'payment_method': random.choice(payment_methods),
            'status': random.choice(statuses),
            'discount_percent': random.choice([0, 5, 10, 15, 20]),
            'tax_percent': 8.5,
            'shipping_cost': round(random.uniform(0, 25), 2),
            'region': random.choice(['North', 'South', 'East', 'West', 'Central'])
        }
        
        # Calculate total
        subtotal = transaction['quantity'] * transaction['unit_price']
        discount = subtotal * (transaction['discount_percent'] / 100)
        after_discount = subtotal - discount
        tax = after_discount * (transaction['tax_percent'] / 100)
        transaction['total_amount'] = round(after_discount + tax + transaction['shipping_cost'], 2)
        
        transactions.append(transaction)
        
        # Progress indicator
        if i % 1000 == 0:
            print(f"  Generated {i}/{num_records} records...")
    
    return transactions

def save_transactions(transactions, filename='data/samples/large_transactions.json'):
    """Save transactions to JSON file"""
    print(f"\nSaving to {filename}...")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(transactions, f, indent=2)
    
    # Calculate file size
    import os
    file_size = os.path.getsize(filename)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"✅ Saved {len(transactions)} transactions")
    print(f"   File size: {file_size_mb:.2f} MB")
    
    # Calculate statistics
    total_amount = sum(t['total_amount'] for t in transactions)
    avg_amount = total_amount / len(transactions)
    
    categories = {}
    for t in transactions:
        cat = t['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total transactions: {len(transactions):,}")
    print(f"   Total amount: ${total_amount:,.2f}")
    print(f"   Average amount: ${avg_amount:.2f}")
    print(f"   Categories: {len(categories)}")
    print(f"   Top category: {max(categories, key=categories.get)} ({categories[max(categories, key=categories.get)]} transactions)")

if __name__ == "__main__":
    # Generate 10,000 transactions
    transactions = generate_large_transactions(10000)
    
    # Save to file
    save_transactions(transactions)
    
    print("\n✅ Large transaction dataset created successfully!")
    print("   Ready for testing with data_optimizer.py")
