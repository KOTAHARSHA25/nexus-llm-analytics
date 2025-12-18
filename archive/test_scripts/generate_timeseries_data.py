#!/usr/bin/env python3
"""
Generate time series sales data for testing
Task 1.1.5: Time Series JSON
"""

import json
import random
from datetime import datetime, timedelta

def generate_timeseries_data():
    """Generate daily sales data for a full year with seasonal patterns"""
    
    start_date = datetime(2024, 1, 1)
    data = []
    
    for day in range(366):  # 2024 is a leap year
        current_date = start_date + timedelta(days=day)
        
        # Seasonal multiplier (summer high, winter low)
        month = current_date.month
        if month in [6, 7, 8]:  # Summer
            seasonal_factor = 1.4
        elif month in [12, 1, 2]:  # Winter
            seasonal_factor = 0.7
        elif month in [3, 4, 5]:  # Spring
            seasonal_factor = 1.1
        else:  # Fall
            seasonal_factor = 1.0
        
        # Weekly pattern (weekends higher)
        day_of_week = current_date.weekday()
        if day_of_week >= 5:  # Saturday, Sunday
            weekly_factor = 1.3
        else:
            weekly_factor = 1.0
        
        # Base sales with random variation
        base_sales = 5000
        random_factor = random.uniform(0.8, 1.2)
        
        sales = base_sales * seasonal_factor * weekly_factor * random_factor
        units = int(sales / random.uniform(50, 150))
        
        entry = {
            "date": current_date.strftime("%Y-%m-%d"),
            "year": current_date.year,
            "month": current_date.month,
            "month_name": current_date.strftime("%B"),
            "quarter": f"Q{(current_date.month - 1) // 3 + 1}",
            "day_of_week": current_date.strftime("%A"),
            "day_of_year": day + 1,
            "week_of_year": current_date.isocalendar()[1],
            "sales_amount": round(sales, 2),
            "units_sold": units,
            "avg_price": round(sales / units if units > 0 else 0, 2),
            "customers": random.randint(50, 200),
            "season": "Summer" if month in [6,7,8] else "Winter" if month in [12,1,2] else "Spring" if month in [3,4,5] else "Fall"
        }
        
        data.append(entry)
    
    return data

def main():
    print("📈 Generating time series sales data...")
    
    data = generate_timeseries_data()
    
    # Calculate statistics
    total_sales = sum(entry['sales_amount'] for entry in data)
    total_units = sum(entry['units_sold'] for entry in data)
    avg_daily_sales = total_sales / len(data)
    
    # Seasonal analysis
    seasons = {}
    for entry in data:
        season = entry['season']
        if season not in seasons:
            seasons[season] = []
        seasons[season].append(entry['sales_amount'])
    
    # Save to file
    output_file = "data/samples/sales_timeseries.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Generated {len(data)} daily sales records")
    print(f"📁 Saved to: {output_file}")
    print(f"\n📊 Summary Statistics:")
    print(f"  • Total Sales: ${total_sales:,.2f}")
    print(f"  • Total Units: {total_units:,}")
    print(f"  • Average Daily Sales: ${avg_daily_sales:,.2f}")
    print(f"  • Date Range: 2024-01-01 to 2024-12-31")
    
    print(f"\n🌡️ Seasonal Patterns:")
    for season in ['Spring', 'Summer', 'Fall', 'Winter']:
        season_sales = seasons.get(season, [])
        if season_sales:
            avg = sum(season_sales) / len(season_sales)
            print(f"  • {season}: ${avg:,.2f} avg/day ({len(season_sales)} days)")
    
    print(f"\n📈 Expected Insights:")
    print(f"  • Highest: Summer months (June-August)")
    print(f"  • Lowest: Winter months (Dec-Feb)")
    print(f"  • Weekend effect: Higher sales on Sat/Sun")
    
    print("\n✨ Time series data generation complete!")

if __name__ == "__main__":
    main()
