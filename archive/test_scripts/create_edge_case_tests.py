"""
EDGE CASE TEST SUITE - Production Readiness Validation
Tests all potential failure scenarios with unseen data patterns
"""
import json
import os
from pathlib import Path

# Create test data directory
test_dir = Path("data/samples/edge_cases")
test_dir.mkdir(parents=True, exist_ok=True)

print("Creating edge case test files...")

# 1. NULL VALUES - Common in real data
null_test = {
    "records": [
        {"id": 1, "name": "Alice", "age": 25, "salary": 50000},
        {"id": 2, "name": "Bob", "age": None, "salary": 60000},  # Null age
        {"id": 3, "name": None, "age": 35, "salary": None},  # Multiple nulls
        {"id": 4, "name": "David", "age": 40, "salary": 70000}
    ]
}
with open(test_dir / "null_values.json", "w") as f:
    json.dump(null_test, f, indent=2)
print("✅ Created: null_values.json")

# 2. SPECIAL CHARACTERS IN KEYS
special_keys = {
    "user-data": [
        {"user-id": 1, "first.name": "Alice", "last name": "Smith", "email_address": "alice@test.com"},
        {"user-id": 2, "first.name": "Bob", "last name": "Jones", "email_address": "bob@test.com"}
    ]
}
with open(test_dir / "special_keys.json", "w") as f:
    json.dump(special_keys, f, indent=2)
print("✅ Created: special_keys.json")

# 3. UNICODE/INTERNATIONAL CHARACTERS
unicode_test = {
    "customers": [
        {"name": "李明", "city": "北京", "amount": 1000},  # Chinese
        {"name": "محمد", "city": "القاهرة", "amount": 2000},  # Arabic
        {"name": "José García", "city": "Madrid", "amount": 1500},  # Spanish accents
        {"name": "Müller", "city": "München", "amount": 1800},  # German umlauts
        {"name": "Test 😀 Emoji", "city": "Tokyo 🇯🇵", "amount": 2500}  # Emoji
    ]
}
with open(test_dir / "unicode_data.json", "w", encoding='utf-8') as f:
    json.dump(unicode_test, f, indent=2, ensure_ascii=False)
print("✅ Created: unicode_data.json")

# 4. BOOLEAN FIELDS
boolean_test = {
    "users": [
        {"id": 1, "name": "Alice", "active": True, "verified": True, "premium": False},
        {"id": 2, "name": "Bob", "active": False, "verified": True, "premium": True},
        {"id": 3, "name": "Charlie", "active": True, "verified": False, "premium": False}
    ]
}
with open(test_dir / "boolean_fields.json", "w") as f:
    json.dump(boolean_test, f, indent=2)
print("✅ Created: boolean_fields.json")

# 5. DATE/TIMESTAMP FORMATS
date_test = {
    "transactions": [
        {"id": 1, "date": "2024-01-15", "timestamp": "2024-01-15T10:30:00Z", "amount": 100},
        {"id": 2, "date": "2024-02-20", "timestamp": "2024-02-20T14:45:00Z", "amount": 200},
        {"id": 3, "date": "2024-03-10", "timestamp": "2024-03-10T09:15:00Z", "amount": 150}
    ]
}
with open(test_dir / "date_formats.json", "w") as f:
    json.dump(date_test, f, indent=2)
print("✅ Created: date_formats.json")

# 6. DEEPLY NESTED (3+ LEVELS)
deep_nest = {
    "company": {
        "departments": {
            "engineering": {
                "teams": {
                    "backend": {
                        "members": 5,
                        "budget": 100000
                    },
                    "frontend": {
                        "members": 3,
                        "budget": 75000
                    }
                }
            },
            "sales": {
                "teams": {
                    "enterprise": {
                        "members": 4,
                        "budget": 80000
                    }
                }
            }
        }
    }
}
with open(test_dir / "deep_nested.json", "w") as f:
    json.dump(deep_nest, f, indent=2)
print("✅ Created: deep_nested.json")

# 7. ARRAYS WITHIN ARRAYS
nested_arrays = {
    "data": {
        "matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "vectors": [[10, 20], [30, 40], [50, 60]]
    }
}
with open(test_dir / "nested_arrays.json", "w") as f:
    json.dump(nested_arrays, f, indent=2)
print("✅ Created: nested_arrays.json")

# 8. EMPTY DATA
empty_array = []
with open(test_dir / "empty_array.json", "w") as f:
    json.dump(empty_array, f, indent=2)
print("✅ Created: empty_array.json")

empty_object = {}
with open(test_dir / "empty_object.json", "w") as f:
    json.dump(empty_object, f, indent=2)
print("✅ Created: empty_object.json")

# 9. MIXED DATA TYPES IN COLUMNS
mixed_types = {
    "records": [
        {"id": 1, "value": 100},  # numeric
        {"id": 2, "value": "200"},  # string number
        {"id": 3, "value": "N/A"},  # string text
        {"id": 4, "value": 300.5}  # float
    ]
}
with open(test_dir / "mixed_types.json", "w") as f:
    json.dump(mixed_types, f, indent=2)
print("✅ Created: mixed_types.json")

# 10. LARGE NESTED ARRAY (Stress test)
large_nested = {
    "products": [
        {"id": i, "name": f"Product {i}", "price": 10.0 + i, "stock": 100 - i}
        for i in range(1, 151)  # 150 products
    ]
}
with open(test_dir / "large_nested_array.json", "w") as f:
    json.dump(large_nested, f, indent=2)
print("✅ Created: large_nested_array.json")

# 11. COMBINATION TEST - Real-world complexity
combo_test = {
    "company-info": {
        "name": "TechCorp™ 科技",  # Unicode + special chars
        "founded": "2020-01-15",  # Date
        "active": True,  # Boolean
        "employees": None,  # Null
        "departments": [
            {
                "dept-id": 1,
                "dept.name": "Engineering",
                "head": None,
                "budget": 500000,
                "active": True,
                "members": [
                    {"name": "Alice", "salary": None},
                    {"name": "李明", "salary": 80000}
                ]
            }
        ]
    }
}
with open(test_dir / "combo_test.json", "w", encoding='utf-8') as f:
    json.dump(combo_test, f, indent=2, ensure_ascii=False)
print("✅ Created: combo_test.json")

print("\n" + "="*70)
print("EDGE CASE TEST FILES CREATED SUCCESSFULLY")
print("="*70)
print(f"\nLocation: {test_dir}")
print(f"Total files: 11")
print("\nFiles created:")
print("  1. null_values.json - Tests null handling")
print("  2. special_keys.json - Tests dashes, dots, spaces in keys")
print("  3. unicode_data.json - Tests Chinese, Arabic, emoji")
print("  4. boolean_fields.json - Tests true/false values")
print("  5. date_formats.json - Tests ISO dates and timestamps")
print("  6. deep_nested.json - Tests 5-level deep nesting")
print("  7. nested_arrays.json - Tests arrays within arrays")
print("  8. empty_array.json - Tests empty [] handling")
print("  9. empty_object.json - Tests empty {} handling")
print(" 10. mixed_types.json - Tests numbers + strings in same column")
print(" 11. large_nested_array.json - Tests 150 nested objects")
print(" 12. combo_test.json - Tests everything combined")
print("\n✅ Ready for comprehensive edge case testing!")
