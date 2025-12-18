"""
Nexus LLM Analytics - Consolidated Optimization System
======================================================

This module combines all optimization functionality:
- Memory optimization and analysis
- Performance optimization with advanced DSA techniques  
- Startup optimization with background loading
- Adaptive timeout management based on system resources

Replaces the separate optimizer files as suggested in DEAD_CODE_ANALYSIS.md
"""

import logging
import time
import psutil
import os
import subprocess
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from functools import lru_cache
from .crewai_import_manager import start_crewai_preloading, get_crewai_import_manager


# ====================================================================
# MEMORY OPTIMIZATION
# ====================================================================

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
            recommendations.append(f"Browser tabs/windows: ~{browser_memory:.1f}GB could be freed")
            potential_savings_gb += browser_memory * 0.7  # Conservative estimate
            
        if ide_memory > 1.0:
            recommendations.append(f"Other IDEs/editors: ~{ide_memory:.1f}GB could be freed")
            potential_savings_gb += ide_memory * 0.8
            
        if other_heavy > 1.0:
            recommendations.append(f"Heavy applications: ~{other_heavy:.1f}GB could be freed")
            potential_savings_gb += other_heavy * 0.6
        
        # Add cached memory that could potentially be freed
        cached_gb = memory_info.get('cached_gb', 0)
        if cached_gb > 0.5:
            recommendations.append(f"System cache: ~{cached_gb:.1f}GB could be cleared")
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
            logging.info("DNS cache cleared")
            
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
                                logging.debug("Operation failed (non-critical) - continuing")  # Skip files we can't delete
                    except PermissionError:
                        logging.debug("Operation failed (non-critical) - continuing")
            
            if cleared_something:
                logging.info("Temporary files cleared")
            else:
                logging.info("No temporary files to clear or insufficient permissions")
                
        except Exception as e:
            logging.warning(f"Cache clearing partially failed: {e}")


# ====================================================================
# PERFORMANCE OPTIMIZATION
# ====================================================================

class PerformanceOptimizer:
    """
    Advanced performance optimization system for all agents
    Uses sophisticated algorithms and data structures for maximum efficiency
    """
    
    def __init__(self):
        self.performance_metrics = defaultdict(list)
        self.optimization_cache = {}
        self.memory_threshold = 0.8  # 80% memory usage threshold
        
    @staticmethod
    @lru_cache(maxsize=128)
    def optimize_query_processing(query: str) -> Dict[str, Any]:
        """
        Optimize query processing using NLP techniques and caching
        Time Complexity: O(1) for cached queries, O(n) for new queries
        """
        words = query.lower().split()
        
        # Intent detection using keyword analysis
        intent_keywords = {
            'analysis': ['analyze', 'analysis', 'examine', 'study', 'investigate'],
            'visualization': ['plot', 'chart', 'graph', 'visualize', 'show'],
            'summary': ['summary', 'summarize', 'overview', 'brief'],
            'statistics': ['stats', 'statistics', 'mean', 'average', 'correlation'],
            'skills': ['skills', 'experience', 'qualifications', 'abilities']
        }
        
        detected_intents = []
        for intent, keywords in intent_keywords.items():
            if any(keyword in words for keyword in keywords):
                detected_intents.append(intent)
        
        # Complexity estimation
        complexity_factors = {
            'word_count': len(words),
            'unique_words': len(set(words)),
            'intents': len(detected_intents),
            'estimated_complexity': 'low' if len(words) < 10 else 'medium' if len(words) < 20 else 'high'
        }
        
        return {
            'processed_query': query,
            'detected_intents': detected_intents,
            'complexity': complexity_factors,
            'optimization_suggestions': _get_optimization_suggestions(complexity_factors)
        }
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """
        Real-time system resource monitoring for adaptive performance
        """
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        
        return {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'cpu_percent': cpu,
            'should_optimize': memory.percent > self.memory_threshold * 100 or cpu > 80
        }
    
    def optimize_data_processing(self, data: Any, operation: str) -> Dict[str, Any]:
        """
        Optimize data processing based on data size and operation type
        Uses adaptive algorithms based on data characteristics
        """
        if hasattr(data, 'shape'):  # pandas DataFrame
            rows, cols = data.shape
        elif hasattr(data, '__len__'):
            rows, cols = len(data), 1
        else:
            rows, cols = 1, 1
        
        # Choose optimal algorithm based on data size
        if rows > 100000:  # Large dataset
            strategy = 'streaming'
            chunk_size = 10000
        elif rows > 10000:  # Medium dataset
            strategy = 'batched'
            chunk_size = 5000
        else:  # Small dataset
            strategy = 'in_memory'
            chunk_size = rows
        
        return {
            'strategy': strategy,
            'chunk_size': chunk_size,
            'estimated_time': _estimate_processing_time(rows, cols, operation),
            'memory_requirement': _estimate_memory_requirement(rows, cols)
        }


class OptimizedAgentMixin:
    """
    Mixin class that adds performance optimizations to any agent
    Can be mixed into existing agent classes for immediate performance improvements
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_optimizer = PerformanceOptimizer()
        self.execution_cache = {}
        self.performance_stats = {
            'total_operations': 0,
            'cache_hits': 0,
            'avg_execution_time': 0.0
        }
    
    def optimized_execute(self, operation: str, *args, **kwargs):
        """
        Execute operation with performance optimizations
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{operation}_{hash(str(args))}{hash(str(sorted(kwargs.items())))}"
        if cache_key in self.execution_cache:
            self.performance_stats['cache_hits'] += 1
            return self.execution_cache[cache_key]
        
        # Monitor system resources
        resources = self.performance_optimizer.monitor_system_resources()
        
        # Execute operation
        try:
            result = self._execute_operation(operation, *args, **kwargs)
            
            # Cache successful results
            self.execution_cache[cache_key] = result
            
        except Exception as e:
            logging.error(f"Operation {operation} failed: {e}")
            result = {"error": str(e), "operation": operation}
        
        # Update performance stats
        execution_time = time.time() - start_time
        self.performance_stats['total_operations'] += 1
        self.performance_stats['avg_execution_time'] = (
            (self.performance_stats['avg_execution_time'] * (self.performance_stats['total_operations'] - 1) + 
             execution_time) / self.performance_stats['total_operations']
        )
        
        # Log performance metrics
        if execution_time > 5.0:  # Log slow operations
            logging.warning(f"Slow operation detected: {operation} took {execution_time:.2f}s")
        
        return result
    
    def _execute_operation(self, operation: str, *args, **kwargs):
        """Override this method in subclasses"""
        raise NotImplementedError("Subclasses must implement _execute_operation")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        cache_hit_rate = (
            self.performance_stats['cache_hits'] / self.performance_stats['total_operations']
            if self.performance_stats['total_operations'] > 0 else 0.0
        )
        
        return {
            **self.performance_stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.execution_cache)
        }


class AdaptiveTimeoutManager:
    """
    Manages adaptive timeouts based on model selection and system resources
    Addresses the user's question about timeout strategy
    """
    
    def __init__(self):
        self.timeout_history = deque(maxlen=100)  # Keep last 100 operations
        self.model_performance = defaultdict(list)
    
    def calculate_optimal_timeout(self, model: str, operation: str, system_resources: Dict[str, float]) -> int:
        """
        Calculate optimal timeout based on multiple factors
        
        Strategy Decision:
        - For high models with low RAM: INCREASE timeout (user accepts slower performance)
        - For efficient models: MAINTAIN standard timeout
        - For good systems: DECREASE timeout for faster failure detection
        """
        
        # Base timeouts by model
        base_timeouts = {
            "llama3.1:8b": 900,     # 15 minutes - complex model
            "phi3:mini": 300,       # 5 minutes - balanced model
            "tinyllama": 180,       # 3 minutes - lightweight model
            "nomic-embed-text": 120 # 2 minutes - embedding model
        }
        
        clean_model = model.replace("ollama/", "")
        base_timeout = base_timeouts.get(clean_model, 600)
        
        # System resource adjustments
        memory_gb = system_resources.get('memory_available_gb', 4.0)
        cpu_percent = system_resources.get('cpu_percent', 50.0)
        
        # ADAPTIVE TIMEOUT STRATEGY
        if memory_gb < 2.0:
            # Low RAM - User wants high model with swap
            # INCREASE timeout significantly to allow swap usage
            multiplier = 3.0
            logging.info(f"Low RAM detected ({memory_gb:.1f}GB) - Extending timeout for swap usage")
            
        elif memory_gb < 4.0:
            # Medium RAM - Some performance impact expected
            multiplier = 2.0
            logging.info(f"Medium RAM ({memory_gb:.1f}GB) - Moderate timeout extension")
            
        elif cpu_percent > 80:
            # High CPU usage - System under load
            multiplier = 1.5
            
        else:
            # Good resources - Use standard or reduced timeout
            multiplier = 1.0
            
        # Historical performance adjustment
        if clean_model in self.model_performance:
            avg_time = sum(self.model_performance[clean_model]) / len(self.model_performance[clean_model])
            if avg_time > base_timeout * 0.8:  # If model typically takes 80%+ of timeout
                multiplier *= 1.3  # Increase timeout preventively
        
        final_timeout = int(base_timeout * multiplier)
        
        # Cap maximum timeout (but allow longer for swap usage)
        max_timeout = 3600 if memory_gb < 2.0 else 1800  # 1 hour for swap, 30 min otherwise
        final_timeout = min(final_timeout, max_timeout)
        
        logging.info(f"Calculated timeout for {model}: {final_timeout}s (base: {base_timeout}s, multiplier: {multiplier:.1f}x)")
        
        return final_timeout
    
    def record_operation_time(self, model: str, operation_time: float):
        """Record operation time for future timeout calculations"""
        clean_model = model.replace("ollama/", "")
        self.model_performance[clean_model].append(operation_time)
        
        # Keep only recent history
        if len(self.model_performance[clean_model]) > 20:
            self.model_performance[clean_model] = self.model_performance[clean_model][-20:]


# ====================================================================
# STARTUP OPTIMIZATION
# ====================================================================

class StartupOptimizer:
    """
    Application startup optimization system
    Pre-loads expensive components during startup to improve API response times
    """
    
    @staticmethod
    def optimize_startup():
        """
        Optimize application startup by pre-loading expensive components
        Call this during application initialization
        """
        logging.info("Starting application startup optimization...")
        
        startup_start = time.perf_counter()
        
        # Start CrewAI background loading immediately 
        start_crewai_preloading()
        
        # Give the background loader a moment to start
        time.sleep(0.1)
        
        manager = get_crewai_import_manager()
        status = manager.get_status()
        
        if status['loading_in_progress']:
            logging.info("CrewAI background loading started successfully")
            logging.info("API requests will use pre-loaded components when ready")
            logging.info("Background loading will complete in ~30-40 seconds")
        else:
            logging.warning("CrewAI background loading may not have started properly")
        
        startup_duration = time.perf_counter() - startup_start
        logging.info(f"Startup optimization completed in {startup_duration:.3f}s")
        
        return {
            'startup_optimization_time': startup_duration,
            'crewai_background_loading': status['loading_in_progress'],
            'estimated_ready_time': 35.0  # Approximate time for CrewAI to load
        }

    @staticmethod
    def check_optimization_status():
        """Check if startup optimizations are complete"""
        manager = get_crewai_import_manager()
        status = manager.get_status()
        
        return {
            'crewai_loaded': status['crewai_loaded'],
            'load_duration': status.get('load_duration'),
            'loading_in_progress': status['loading_in_progress'],
            'ready_for_requests': status['crewai_loaded']
        }

    @staticmethod
    def wait_for_optimization_completion(timeout: float = 60.0):
        """
        Wait for startup optimizations to complete
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if optimization completed, False if timeout
        """
        logging.info("Waiting for startup optimization to complete...")
        
        start_wait = time.perf_counter()
        
        while (time.perf_counter() - start_wait) < timeout:
            status = StartupOptimizer.check_optimization_status()
            
            if status['ready_for_requests']:
                total_wait = time.perf_counter() - start_wait
                logging.info(f"Startup optimization completed in {total_wait:.2f}s")
                logging.info(f"CrewAI load duration: {status.get('load_duration', 'unknown'):.2f}s")
                return True
            
            # Log progress every 10 seconds
            elapsed = time.perf_counter() - start_wait
            if int(elapsed) % 10 == 0 and elapsed > 1:
                logging.info(f"Still waiting... ({elapsed:.0f}s elapsed)")
            
            time.sleep(1)
        
        logging.warning(f"Startup optimization timeout after {timeout}s")
        return False


# ====================================================================
# UNIFIED OPTIMIZATION MANAGER
# ====================================================================

class UnifiedOptimizer:
    """
    Unified interface for all optimization functionality
    Provides a single point of access for memory, performance, and startup optimizations
    """
    
    def __init__(self):
        self.memory_optimizer = MemoryOptimizer()
        self.performance_optimizer = PerformanceOptimizer()
        self.startup_optimizer = StartupOptimizer()
        self.timeout_manager = AdaptiveTimeoutManager()
    
    def get_system_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive system optimization report"""
        memory_plan = self.memory_optimizer.get_optimization_plan()
        system_resources = self.performance_optimizer.monitor_system_resources()
        startup_status = self.startup_optimizer.check_optimization_status()
        
        return {
            'timestamp': time.time(),
            'memory_optimization': memory_plan,
            'performance_metrics': system_resources,
            'startup_status': startup_status,
            'recommendations': self._generate_unified_recommendations(memory_plan, system_resources)
        }
    
    def _generate_unified_recommendations(self, memory_plan: Dict, resources: Dict) -> List[str]:
        """Generate unified optimization recommendations"""
        recommendations = []
        
        # Memory recommendations
        if memory_plan['optimization_recommendations']:
            recommendations.extend(memory_plan['optimization_recommendations'])
        
        # Performance recommendations
        if resources['should_optimize']:
            if resources['memory_percent'] > 80:
                recommendations.append("High memory usage detected - consider closing applications")
            if resources['cpu_percent'] > 80:
                recommendations.append("High CPU usage detected - system may be under load")
        
        # Model compatibility recommendations
        incompatible_models = [
            model for model, compat in memory_plan['model_compatibility'].items()
            if not compat['current'] and compat['after_cleanup']
        ]
        
        if incompatible_models:
            recommendations.append(f"Free up memory to enable: {', '.join(incompatible_models)}")
        
        return recommendations
    
    def optimize_for_model(self, model_name: str) -> Dict[str, Any]:
        """
        Optimize system for a specific model
        """
        memory_plan = self.memory_optimizer.get_optimization_plan()
        resources = self.performance_optimizer.monitor_system_resources()
        
        # Calculate optimal timeout for the model
        optimal_timeout = self.timeout_manager.calculate_optimal_timeout(
            model_name, "analysis", resources
        )
        
        # Check if model is compatible
        model_compat = memory_plan['model_compatibility'].get(model_name, {})
        
        return {
            'model': model_name,
            'optimal_timeout': optimal_timeout,
            'current_compatible': model_compat.get('current', False),
            'after_cleanup_compatible': model_compat.get('after_cleanup', False),
            'required_memory_gb': model_compat.get('required_gb', 'unknown'),
            'available_memory_gb': resources['memory_available_gb'],
            'optimization_needed': not model_compat.get('current', False),
            'recommendations': memory_plan.get('optimization_recommendations', [])
        }


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def _get_optimization_suggestions(complexity: Dict[str, Any]) -> List[str]:
    """Generate optimization suggestions based on query complexity"""
    suggestions = []
    
    if complexity['word_count'] > 30:
        suggestions.append("Consider breaking down the query into smaller parts")
    
    if complexity['estimated_complexity'] == 'high':
        suggestions.append("High complexity query - may take longer to process")
    
    if complexity['intents'] > 3:
        suggestions.append("Multiple intents detected - consider separate queries for better results")
    
    return suggestions


def _estimate_processing_time(rows: int, cols: int, operation: str) -> float:
    """Estimate processing time based on data size and operation"""
    base_times = {
        'analysis': 0.001,      # 1ms per row
        'visualization': 0.002,  # 2ms per row
        'summary': 0.0005,      # 0.5ms per row
        'statistics': 0.0015    # 1.5ms per row
    }
    
    base_time = base_times.get(operation, 0.001)
    return rows * cols * base_time


def _estimate_memory_requirement(rows: int, cols: int) -> float:
    """Estimate memory requirement in MB"""
    # Assume 8 bytes per numeric value + overhead
    bytes_per_cell = 12  # Including overhead
    total_bytes = rows * cols * bytes_per_cell
    return total_bytes / (1024 * 1024)  # Convert to MB


def should_increase_timeout_for_low_ram() -> Dict[str, Any]:
    """
    Answer the user's question about timeout strategy for low RAM + high models
    
    RECOMMENDATION: YES, increase timeout when user chooses high model with low RAM
    """
    return {
        "recommendation": "INCREASE_TIMEOUT",
        "reasoning": [
            "User explicitly chooses quality over speed when selecting high model on low RAM",
            "Swap usage can work but needs significantly more time",
            "Better to get correct results slowly than timeout failures",
            "System already warns user about performance impact"
        ],
        "implementation": "Adaptive timeout based on available RAM and model choice",
        "timeout_strategy": {
            "low_ram_high_model": "3x base timeout (up to 45 min for complex analysis)",
            "normal_ram": "1x base timeout (5-15 min depending on model)",
            "high_ram": "0.8x base timeout (faster failure detection)"
        },
        "user_control": "Allow users to override timeout in settings if needed"
    }


# ====================================================================
# LEGACY FUNCTION ALIASES (for backward compatibility)
# ====================================================================

# Maintain backward compatibility with existing imports
optimize_startup = StartupOptimizer.optimize_startup
check_optimization_status = StartupOptimizer.check_optimization_status
wait_for_optimization_completion = StartupOptimizer.wait_for_optimization_completion


# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    """Main function to run comprehensive optimization analysis"""
    optimizer = UnifiedOptimizer()
    
    print("Nexus LLM Analytics - Consolidated Optimization System")
    print("=" * 60)
    
    # Get comprehensive optimization report
    report = optimizer.get_system_optimization_report()
    
    # Display memory status
    memory = report["memory_optimization"]["current_memory"]
    print(f"Memory Status:")
    print(f"  Total: {memory['total_gb']:.1f}GB")
    print(f"  Available: {memory['available_gb']:.1f}GB ({100-memory['percent_used']:.1f}% free)")
    print(f"  Used: {memory['used_gb']:.1f}GB ({memory['percent_used']:.1f}%)")
    
    # Show top memory-consuming processes
    print(f"\nTop Memory-Consuming Processes:")
    for i, proc in enumerate(report["memory_optimization"]["top_processes"][:5], 1):
        print(f"  {i}. {proc['name']} - {proc['memory_gb']:.1f}GB")
    
    # Show unified recommendations
    if report["recommendations"]:
        print(f"\nOptimization Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
    else:
        print(f"\nSystem optimization looks good!")
    
    # Show model compatibility
    print(f"\nModel Compatibility:")
    for model, compat in report["memory_optimization"]["model_compatibility"].items():
        current_status = "✅" if compat["current"] else "❌"
        cleanup_status = "✅" if compat["after_cleanup"] else "❌"
        
        print(f"  {model}:")
        print(f"    Current: {current_status} (needs {compat['required_gb']:.1f}GB)")
        if not compat["current"] and compat["after_cleanup"]:
            print(f"    After cleanup: {cleanup_status} (possible with optimization)")
    
    # Show startup status
    startup = report["startup_status"]
    status = "✅ Ready" if startup["ready_for_requests"] else "⏳ Loading"
    print(f"\nStartup Status: {status}")
    if startup.get("load_duration"):
        print(f"  Load time: {startup['load_duration']:.1f}s")


if __name__ == "__main__":
    main()