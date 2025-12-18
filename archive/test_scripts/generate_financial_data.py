#!/usr/bin/env python3
"""
Generate financial quarterly data for testing
Task 1.1.4: Financial Data JSON
"""

import json
import random
from datetime import datetime, timedelta

def generate_financial_data():
    """Generate quarterly financial data for 4 quarters"""
    
    quarters = {
        "Q1": {"months": ["January", "February", "March"], "start_month": 1},
        "Q2": {"months": ["April", "May", "June"], "start_month": 4},
        "Q3": {"months": ["July", "August", "September"], "start_month": 7},
        "Q4": {"months": ["October", "November", "December"], "start_month": 10},
    }
    
    financial_data = []
    
    for quarter, info in quarters.items():
        for month in info["months"]:
            # Generate realistic financial metrics for each month
            revenue = random.uniform(800000, 1500000)
            cost_of_goods = revenue * random.uniform(0.35, 0.50)
            operating_expenses = revenue * random.uniform(0.20, 0.30)
            marketing = revenue * random.uniform(0.08, 0.15)
            rd_expenses = revenue * random.uniform(0.05, 0.12)
            
            gross_profit = revenue - cost_of_goods
            operating_income = gross_profit - operating_expenses - marketing - rd_expenses
            taxes = operating_income * 0.21  # 21% tax rate
            net_income = operating_income - taxes
            
            profit_margin = (net_income / revenue) * 100
            gross_margin = (gross_profit / revenue) * 100
            
            entry = {
                "quarter": quarter,
                "month": month,
                "year": 2024,
                "revenue": round(revenue, 2),
                "cost_of_goods_sold": round(cost_of_goods, 2),
                "gross_profit": round(gross_profit, 2),
                "gross_margin_percent": round(gross_margin, 2),
                "operating_expenses": round(operating_expenses, 2),
                "marketing_expenses": round(marketing, 2),
                "rd_expenses": round(rd_expenses, 2),
                "operating_income": round(operating_income, 2),
                "taxes": round(taxes, 2),
                "net_income": round(net_income, 2),
                "profit_margin_percent": round(profit_margin, 2),
                "employees": random.randint(50, 150),
                "customers": random.randint(500, 2000),
                "region": random.choice(["North America", "Europe", "Asia-Pacific", "Latin America"])
            }
            
            financial_data.append(entry)
    
    return financial_data

def main():
    print("🏦 Generating financial quarterly data...")
    
    data = generate_financial_data()
    
    # Calculate overall statistics
    total_revenue = sum(entry['revenue'] for entry in data)
    total_net_income = sum(entry['net_income'] for entry in data)
    overall_margin = (total_net_income / total_revenue) * 100
    
    # Save to file
    output_file = "data/samples/financial_quarterly.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Generated {len(data)} monthly financial records")
    print(f"📁 Saved to: {output_file}")
    print(f"\n📊 Summary Statistics:")
    print(f"  • Total Revenue: ${total_revenue:,.2f}")
    print(f"  • Total Net Income: ${total_net_income:,.2f}")
    print(f"  • Overall Profit Margin: {overall_margin:.2f}%")
    print(f"  • Records per Quarter: 3 months")
    print(f"  • Total Quarters: 4 (Q1-Q4)")
    
    # Show Q1 total
    q1_data = [entry for entry in data if entry['quarter'] == 'Q1']
    q1_revenue = sum(entry['revenue'] for entry in q1_data)
    print(f"\n📈 Q1 2024 Revenue: ${q1_revenue:,.2f}")
    
    # Find quarter with highest profit margin
    quarter_margins = {}
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_data = [entry for entry in data if entry['quarter'] == quarter]
        q_revenue = sum(entry['revenue'] for entry in q_data)
        q_net_income = sum(entry['net_income'] for entry in q_data)
        quarter_margins[quarter] = (q_net_income / q_revenue) * 100
    
    best_quarter = max(quarter_margins.items(), key=lambda x: x[1])
    print(f"🏆 Highest Profit Margin: {best_quarter[0]} ({best_quarter[1]:.2f}%)")
    
    print("\n✨ Financial data generation complete!")

if __name__ == "__main__":
    main()
