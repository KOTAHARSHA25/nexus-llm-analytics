# Memory Optimization Helper for Nexus LLM Analytics
# Helps free up RAM before running AI analysis

import psutil
import os
import subprocess
import logging
from typing import List, Dict, Tuple

class MemoryOptimizer:
    """
    Helps optimize system memory for LLM operations.
    Provides recommendations and utilities to free up RAM.
    """
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get detailed memory usage information"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "free_gb": memory.free / (1024**3),
            "percent_used": memory.percent,
            "cached_gb": getattr(memory, 'cached', 0) / (1024**3),
            "buffers_gb": getattr(memory, 'buffers', 0) / (1024**3)
        }
    
    @staticmethod
    def get_top_memory_processes(limit: int = 10) -> List[Dict[str, any]]:
        """Get processes using the most memory"""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                process_info = proc.info
                memory_mb = process_info['memory_info'].rss / (1024 * 1024)
                
                processes.append({
                    'pid': process_info['pid'],
                    'name': process_info['name'],
                    'memory_mb': memory_mb,
                    'memory_gb': memory_mb / 1024
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort by memory usage (descending)
        processes.sort(key=lambda x: x['memory_mb'], reverse=True)
        return processes[:limit]
    
    @staticmethod
    def estimate_available_after_cleanup() -> Tuple[float, List[str]]:
        """
        Estimate how much RAM could be freed and provide recommendations.
        
        Returns:
            Tuple[estimated_available_gb, recommendations]
        """
        memory_info = MemoryOptimizer.get_memory_usage()
        top_processes = MemoryOptimizer.get_top_memory_processes(15)
        
        recommendations = []
        potential_savings_gb = 0
        
        # Look for memory-heavy applications that could be closed
        browser_memory = 0
        ide_memory = 0
        other_heavy = 0
        
        for proc in top_processes:
            name = proc['name'].lower()
            memory_gb = proc['memory_gb']
            
            # Skip system-critical processes
            if name in ['system', 'svchost.exe', 'dwm.exe', 'explorer.exe', 'winlogon.exe']:
                continue
                
            # Browser processes
            if any(browser in name for browser in ['chrome', 'firefox', 'edge', 'brave', 'opera']):
                browser_memory += memory_gb
            # IDEs and editors
            elif any(ide in name for ide in ['code', 'devenv', 'pycharm', 'intellij', 'sublime']):
                ide_memory += memory_gb
            # Other heavy applications
            elif memory_gb > 0.5:
                other_heavy += memory_gb
        
        # Generate recommendations
        if browser_memory > 1.0:
            recommendations.append(f"🌐 Close browser tabs/windows: ~{browser_memory:.1f}GB could be freed")
            potential_savings_gb += browser_memory * 0.7  # Conservative estimate
            
        if ide_memory > 1.0:
            recommendations.append(f"💻 Close other IDEs/editors: ~{ide_memory:.1f}GB could be freed")
            potential_savings_gb += ide_memory * 0.8
            
        if other_heavy > 1.0:
            recommendations.append(f"📱 Close heavy applications: ~{other_heavy:.1f}GB could be freed")
            potential_savings_gb += other_heavy * 0.6
        
        # Add cached memory that could potentially be freed
        cached_gb = memory_info.get('cached_gb', 0)
        if cached_gb > 0.5:
            recommendations.append(f"🗂️ System cache: ~{cached_gb:.1f}GB could be cleared")
            potential_savings_gb += cached_gb * 0.3  # Very conservative for cache
        
        estimated_available = memory_info['available_gb'] + potential_savings_gb
        
        return estimated_available, recommendations
    
    @staticmethod
    def get_optimization_plan() -> Dict[str, any]:
        """Get a comprehensive memory optimization plan"""
        current_memory = MemoryOptimizer.get_memory_usage()
        top_processes = MemoryOptimizer.get_top_memory_processes(10)
        estimated_available, recommendations = MemoryOptimizer.estimate_available_after_cleanup()
        
        # Determine what models could run after optimization
        model_compatibility = {}
        models = {
            "llama3.1:8b": 6.0,
            "phi3:mini": 2.0,
            "nomic-embed-text": 0.5
        }
        
        for model, required_gb in models.items():
            current_compatible = current_memory['available_gb'] >= required_gb
            after_cleanup_compatible = estimated_available >= required_gb
            
            model_compatibility[model] = {
                "current": current_compatible,
                "after_cleanup": after_cleanup_compatible,
                "required_gb": required_gb
            }
        
        return {
            "current_memory": current_memory,
            "top_processes": top_processes,
            "estimated_available_after_cleanup": estimated_available,
            "optimization_recommendations": recommendations,
            "model_compatibility": model_compatibility
        }
    
    @staticmethod
    def clear_system_cache():
        """Clear system caches (Windows-specific)"""
        try:
            # Clear DNS cache
            subprocess.run(['ipconfig', '/flushdns'], capture_output=True, check=True)
            print("✅ DNS cache cleared")
            
            # Clear Windows temporary files (requires admin on some systems)
            temp_dirs = [
                os.environ.get('TEMP', ''),
                os.environ.get('TMP', ''),
                os.path.join(os.environ.get('WINDIR', ''), 'Temp')
            ]
            
            cleared_something = False
            for temp_dir in temp_dirs:
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        for file in os.listdir(temp_dir):
                            file_path = os.path.join(temp_dir, file)
                            try:
                                if os.path.isfile(file_path):
                                    os.unlink(file_path)
                                cleared_something = True
                            except (PermissionError, FileNotFoundError):
                                pass  # Skip files we can't delete
                    except PermissionError:
                        pass
            
            if cleared_something:
                print("✅ Temporary files cleared")
            else:
                print("ℹ️ No temporary files to clear or insufficient permissions")
                
        except Exception as e:
            print(f"⚠️ Cache clearing partially failed: {e}")

def main():
    """Main function to run memory optimization analysis"""
    print("🧠 Nexus LLM Analytics - Memory Optimizer")
    print("=" * 50)
    
    # Get optimization plan
    plan = MemoryOptimizer.get_optimization_plan()
    
    # Display current memory status
    memory = plan["current_memory"]
    print(f"💾 Current Memory Status:")
    print(f"   Total: {memory['total_gb']:.1f}GB")
    print(f"   Available: {memory['available_gb']:.1f}GB ({100-memory['percent_used']:.1f}% free)")
    print(f"   Used: {memory['used_gb']:.1f}GB ({memory['percent_used']:.1f}%)")
    
    # Show top memory-consuming processes
    print(f"\n🔍 Top Memory-Consuming Processes:")
    for i, proc in enumerate(plan["top_processes"][:5], 1):
        print(f"   {i}. {proc['name']} - {proc['memory_gb']:.1f}GB")
    
    # Show optimization recommendations
    if plan["optimization_recommendations"]:
        print(f"\n💡 Memory Optimization Recommendations:")
        for rec in plan["optimization_recommendations"]:
            print(f"   {rec}")
        
        print(f"\n📈 Estimated Available After Cleanup: {plan['estimated_available_after_cleanup']:.1f}GB")
    else:
        print(f"\n✅ Memory usage looks optimal!")
    
    # Show model compatibility
    print(f"\n🤖 Model Compatibility:")
    for model, compat in plan["model_compatibility"].items():
        current_status = "✅" if compat["current"] else "❌"
        cleanup_status = "✅" if compat["after_cleanup"] else "❌"
        
        print(f"   {model}:")
        print(f"      Current: {current_status} (needs {compat['required_gb']:.1f}GB)")
        if not compat["current"] and compat["after_cleanup"]:
            print(f"      After cleanup: {cleanup_status} (possible with optimization)")

if __name__ == "__main__":
    main()