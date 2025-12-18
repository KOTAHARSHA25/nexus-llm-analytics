"""
Clear All Cache Script
======================
Clears all cached queries and file analysis to force fresh LLM analysis.
Run this when you need to invalidate stale cache entries.
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

try:
    from backend.core.advanced_cache import clear_all_caches, get_cache_status
    
    print("\n" + "="*60)
    print("🧹 CACHE CLEARING UTILITY")
    print("="*60)
    
    # Show cache status before clearing
    print("\n📊 Cache Status BEFORE Clearing:")
    status = get_cache_status()
    for cache_name, cache_stats in status.items():
        if cache_name != 'overall_performance':
            print(f"  {cache_name}:")
            print(f"    - Current Size: {cache_stats.get('current_size', 0)}")
            print(f"    - Total Requests: {cache_stats.get('total_requests', 0)}")
            print(f"    - Hit Rate: {cache_stats.get('hit_rate', 0):.2f}%")
    
    # Clear all caches
    print("\n🧹 Clearing all caches...")
    clear_all_caches()
    
    # Show cache status after clearing
    print("\n✅ Cache Status AFTER Clearing:")
    status = get_cache_status()
    for cache_name, cache_stats in status.items():
        if cache_name != 'overall_performance':
            print(f"  {cache_name}:")
            print(f"    - Current Size: {cache_stats.get('current_size', 0)}")
            print(f"    - Total Requests: {cache_stats.get('total_requests', 0)}")
            print(f"    - Hit Rate: {cache_stats.get('hit_rate', 0):.2f}%")
    
    print("\n" + "="*60)
    print("✅ All caches cleared successfully!")
    print("="*60)
    
except ImportError as e:
    print(f"\n❌ Error: Could not import cache modules: {e}")
    print("Make sure the backend is in the correct path.")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Error clearing cache: {e}")
    sys.exit(1)
