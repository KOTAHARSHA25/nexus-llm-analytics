"""Memory optimisation helper for Nexus LLM Analytics.

Provides utilities to inspect system RAM usage, identify memory-hungry
processes, and give actionable recommendations before running LLM inference.

Enterprise v2.0 Additions
-------------------------
* **MemoryAlert** — Dataclass representing a memory-pressure alert with
  severity, timestamp, and actionable recommendation.
* **MemoryProfile** — Context-manager that snapshots RAM before/after a
  block and logs the delta, useful for profiling LLM invocations.
* **get_memory_optimizer()** — Thread-safe singleton accessor.

All v1.x APIs (``MemoryOptimizer``, ``main``) remain unchanged.

Author: Nexus Team
Since: v1.0 (Enterprise enhancements v2.0 — February 2026)
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import psutil

class MemoryOptimizer:
    """Helps optimise system memory for LLM operations.

    Provides static utility methods to inspect current RAM usage,
    identify the heaviest processes, estimate reclaimable memory,
    and build an actionable optimisation plan with per-model
    compatibility checks.

    All methods are ``@staticmethod`` so the class can be used
    without instantiation, but enterprise code should prefer the
    :func:`get_memory_optimizer` singleton for consistency.

    .. versionchanged:: 2.0
       Added enterprise singleton accessor and profiling helpers.
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
    def get_top_memory_processes(limit: int = 10) -> List[Dict[str, Any]]:
        """Get processes using the most memory."""
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
    def get_optimization_plan() -> Dict[str, Any]:
        """Get a comprehensive memory optimization plan."""
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
    def clear_system_cache() -> None:
        """Clear system caches (Windows-specific)."""
        try:
            # Clear DNS cache
            subprocess.run(['ipconfig', '/flushdns'], capture_output=True, check=True)
            logging.info("✅ DNS cache cleared")
            
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
                logging.info("✅ Temporary files cleared")
            else:
                logging.info("ℹ️ No temporary files to clear or insufficient permissions")
                
        except Exception as e:
            logging.warning(f"⚠️ Cache clearing partially failed: {e}")

def main() -> None:
    """Run interactive memory optimisation analysis and print results."""
    logging.info("🧠 Nexus LLM Analytics - Memory Optimizer")
    logging.info("=" * 50)
    
    # Get optimization plan
    plan = MemoryOptimizer.get_optimization_plan()
    
    # Display current memory status
    memory = plan["current_memory"]
    logging.info(f"💾 Current Memory Status:")
    logging.info(f"   Total: {memory['total_gb']:.1f}GB")
    logging.info(f"   Available: {memory['available_gb']:.1f}GB ({100-memory['percent_used']:.1f}% free)")
    logging.info(f"   Used: {memory['used_gb']:.1f}GB ({memory['percent_used']:.1f}%)")
    
    # Show top memory-consuming processes
    logging.info(f"\n🔍 Top Memory-Consuming Processes:")
    for i, proc in enumerate(plan["top_processes"][:5], 1):
        logging.info(f"   {i}. {proc['name']} - {proc['memory_gb']:.1f}GB")
    
    # Show optimization recommendations
    if plan["optimization_recommendations"]:
        logging.info(f"\n💡 Memory Optimization Recommendations:")
        for rec in plan["optimization_recommendations"]:
            logging.info(f"   {rec}")
        
        logging.info(f"\n📈 Estimated Available After Cleanup: {plan['estimated_available_after_cleanup']:.1f}GB")
    else:
        logging.info(f"\n✅ Memory usage looks optimal!")
    
    # Show model compatibility
    logging.info(f"\n🤖 Model Compatibility:")
    for model, compat in plan["model_compatibility"].items():
        current_status = "✅" if compat["current"] else "❌"
        cleanup_status = "✅" if compat["after_cleanup"] else "❌"
        
        logging.info(f"   {model}:")
        logging.info(f"      Current: {current_status} (needs {compat['required_gb']:.1f}GB)")
        if not compat["current"] and compat["after_cleanup"]:
            logging.info(f"      After cleanup: {cleanup_status} (possible with optimization)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()


# ============================================================================
# Enterprise v2.0 — MemoryAlert, MemoryProfile & Singleton
# ============================================================================


@dataclass
class MemoryAlert:
    """Structured memory-pressure alert.

    Attributes:
        severity: One of ``low``, ``moderate``, ``high``, ``critical``.
        available_gb: Available RAM at the time of the alert.
        percent_used: Percentage of total RAM in use.
        recommendation: Human-readable recommendation string.
        timestamp: ISO-8601 timestamp of alert creation.

    .. versionadded:: 2.0
    """

    severity: str
    available_gb: float
    percent_used: float
    recommendation: str
    timestamp: str = field(default_factory=lambda: __import__("datetime").datetime.now().isoformat())

    @classmethod
    def from_current_state(cls) -> "MemoryAlert":
        """Create an alert reflecting the current memory state.

        Returns:
            A :class:`MemoryAlert` with severity and recommendation
            computed from live ``psutil`` data.
        """
        info = MemoryOptimizer.get_memory_usage()
        pct = info["percent_used"]
        avail = info["available_gb"]
        if pct >= 95:
            sev, rec = "critical", "Immediately free memory or reduce model size."
        elif pct >= 85:
            sev, rec = "high", "Close heavy applications before running LLM inference."
        elif pct >= 70:
            sev, rec = "moderate", "Consider closing unused browser tabs."
        else:
            sev, rec = "low", "Memory usage is healthy."
        return cls(severity=sev, available_gb=avail, percent_used=pct, recommendation=rec)


class MemoryProfile:
    """Context manager that profiles memory usage across a code block.

    Records available RAM before and after the block executes and
    logs the delta at ``DEBUG`` level.

    Example::

        with MemoryProfile("llm_generate") as mp:
            result = llm.generate(prompt)
        print(f"RAM delta: {mp.delta_mb:.1f} MB")

    Attributes:
        label: Descriptive label for log messages.
        before_mb: Available RAM (MB) before the block.
        after_mb: Available RAM (MB) after the block.

    .. versionadded:: 2.0
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.before_mb: float = 0.0
        self.after_mb: float = 0.0

    def __enter__(self) -> "MemoryProfile":
        self.before_mb = psutil.virtual_memory().available / (1024 * 1024)
        return self

    def __exit__(self, *exc_info) -> None:
        self.after_mb = psutil.virtual_memory().available / (1024 * 1024)
        logging.debug(
            "MemoryProfile [%s]: before=%.1f MB, after=%.1f MB, delta=%.1f MB",
            self.label, self.before_mb, self.after_mb, self.delta_mb,
        )

    @property
    def delta_mb(self) -> float:
        """Change in available RAM in megabytes (negative = consumed)."""
        return self.after_mb - self.before_mb


# Thread-safe singleton
_memory_optimizer_instance: MemoryOptimizer | None = None
_memory_optimizer_lock = threading.Lock()


def get_memory_optimizer() -> MemoryOptimizer:
    """Return the global :class:`MemoryOptimizer` singleton (thread-safe).

    Uses double-checked locking to avoid contention after the first
    call.

    .. versionadded:: 2.0
    """
    global _memory_optimizer_instance
    if _memory_optimizer_instance is None:
        with _memory_optimizer_lock:
            if _memory_optimizer_instance is None:
                _memory_optimizer_instance = MemoryOptimizer()
    return _memory_optimizer_instance