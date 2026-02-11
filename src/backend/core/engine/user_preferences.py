"""Persistent user preference management.

Provides a Pydantic-backed model for user preferences (model choices,
memory buffers, routing flags) with JSON file persistence and a
thread-safe singleton accessor.

.. versionadded:: 2.0.0
   Added :class:`PreferenceVersion`, :class:`PreferenceMigrator`,
   :class:`PreferenceProfile`, and multi-tenant support.

Backward Compatibility
----------------------
All v1.x public names remain at the same import paths.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

__all__ = [
    # v1.x (backward compatible)
    "UserPreferences",
    "UserPreferencesManager",
    "get_preferences_manager",
    # v2.0 Enterprise additions
    "PreferenceVersion",
    "PreferenceMigrator",
    "PreferenceProfile",
    "MultiTenantPreferences",
    "get_multi_tenant_preferences",
]

class UserPreferences(BaseModel):
    """User preferences model with validation.

    Attributes:
        primary_model: Primary LLM model identifier.
        review_model: Secondary model used for CoT review passes.
        embedding_model: Model used for vector embeddings.
        auto_model_selection: Enable automatic model selection via routing.
        allow_swap_usage: Allow OS swap memory for large models.
        memory_buffer_gb: Reserved RAM buffer in gigabytes.
        preferred_performance: Performance tier ("high", "balanced", "efficient").
        enable_intelligent_routing: Enable semantic routing for queries.
        first_time_setup_complete: Whether initial setup wizard has run.
        last_updated: ISO-8601 timestamp of the last preference write.
        model_test_results: Cached per-model health-check outcomes.
    """

    primary_model: str = "llama3.1:8b"
    review_model: str = "phi3:mini"
    embedding_model: str = "nomic-embed-text"
    auto_model_selection: bool = True
    allow_swap_usage: bool = True
    memory_buffer_gb: float = 0.5
    preferred_performance: str = "balanced"  # "high", "balanced", "efficient"
    enable_intelligent_routing: bool = True  # ON by default - enables semantic routing for paper evaluation
    first_time_setup_complete: bool = False
    last_updated: str = ""
    model_test_results: Dict[str, Any] = {}

class UserPreferencesManager:
    """Manages user preferences with persistent JSON storage.

    Reads and writes ``user_preferences.json`` in the project ``config/``
    directory.  Caches the loaded :class:`UserPreferences` in memory so
    repeated calls to :meth:`load_preferences` are cheap.

    Args:
        config_dir: Override directory for the JSON file.  Defaults to
            ``<project>/config/``.
    """
    
    def __init__(self, config_dir: Optional[str] = None) -> None:
        if config_dir is None:
            # Store in project root config directory
            config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.preferences_file = self.config_dir / "user_preferences.json"
        self._preferences: Optional[UserPreferences] = None
    
    def load_preferences(self) -> UserPreferences:
        """Load user preferences from file or create defaults.

        Returns:
            UserPreferences: The current preferences instance.
        """
        if self._preferences is not None:
            return self._preferences
        
        try:
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r') as f:
                    data = json.load(f)
                    self._preferences = UserPreferences(**data)
                    logger.debug("User preferences loaded successfully")
            else:
                # Create default preferences
                self._preferences = UserPreferences()
                self.save_preferences()
                logger.debug("Created default user preferences")
        except Exception as e:
            logger.warning("Error loading preferences, using defaults: %s", e)
            self._preferences = UserPreferences()
        
        return self._preferences
    
    def save_preferences(self, preferences: UserPreferences = None) -> bool:
        """Save user preferences to file.

        Args:
            preferences: Instance to persist. Uses cached copy when *None*.

        Returns:
            bool: *True* on successful write, *False* on error.
        """
        try:
            if preferences is not None:
                self._preferences = preferences
            
            if self._preferences is None:
                return False
            
            # Update last modified timestamp
            self._preferences.last_updated = datetime.now().isoformat()
            
            # Save to file
            with open(self.preferences_file, 'w') as f:
                json.dump(self._preferences.model_dump(), f, indent=2)
            
            logger.debug("User preferences saved successfully")
            return True
        except Exception as e:
            logger.error("Error saving preferences: %s", e, exc_info=True)
            return False
    
    def update_preferences(self, **kwargs) -> bool:
        """Update specific preference fields.

        Args:
            **kwargs: Field-name / value pairs matching :class:`UserPreferences`.

        Returns:
            bool: *True* if all updates were saved successfully.
        """
        try:
            preferences = self.load_preferences()
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(preferences, key):
                    setattr(preferences, key, value)
                else:
                    logger.warning("Unknown preference field: %s", key)
            
            return self.save_preferences(preferences)
        except Exception as e:
            logger.error("Error updating preferences: %s", e, exc_info=True)
            return False
    
    def get_model_config(self) -> Dict[str, str]:
        """Get current model configuration.

        Returns:
            Dict: Keys ``primary_model``, ``review_model``, ``embedding_model``,
                ``auto_selection``, and ``allow_swap``.
        """
        preferences = self.load_preferences()
        return {
            "primary_model": preferences.primary_model,
            "review_model": preferences.review_model,
            "embedding_model": preferences.embedding_model,
            "auto_selection": preferences.auto_model_selection,
            "allow_swap": preferences.allow_swap_usage
        }
    
    def set_model_config(self, primary: str, review: str, embedding: str = None, auto_selection: bool = None) -> bool:
        """Set model configuration.

        Args:
            primary: Primary LLM model identifier.
            review: Review / CoT model identifier.
            embedding: Embedding model identifier (optional).
            auto_selection: Enable automatic model selection (optional).

        Returns:
            bool: *True* if the configuration was persisted.
        """
        update_data = {
            "primary_model": primary,
            "review_model": review
        }
        
        if embedding is not None:
            update_data["embedding_model"] = embedding
        
        if auto_selection is not None:
            update_data["auto_model_selection"] = auto_selection
        
        return self.update_preferences(**update_data)
    
    def mark_setup_complete(self) -> bool:
        """Mark first-time setup as complete.

        Returns:
            bool: *True* if the flag was persisted.
        """
        return self.update_preferences(first_time_setup_complete=True)
    
    def is_first_time_user(self) -> bool:
        """Check if this is a first-time user.

        Returns:
            bool: *True* when setup has not been completed yet.
        """
        preferences = self.load_preferences()
        return not preferences.first_time_setup_complete
    
    def save_model_test_result(self, model_name: str, success: bool, response_time: float = None, error: str = None) -> bool:
        """Save model test results.

        Args:
            model_name: Identifier of the tested model.
            success: Whether the health-check passed.
            response_time: Round-trip latency in seconds (optional).
            error: Error message on failure (optional).

        Returns:
            bool: *True* if the result was persisted.
        """
        preferences = self.load_preferences()
        
        test_result = {
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "response_time": response_time,
            "error": error
        }
        
        preferences.model_test_results[model_name] = test_result
        return self.save_preferences(preferences)
    
    def save_test_result(self, test_data: Dict[str, Any]) -> bool:
        """Save general test results.

        Args:
            test_data: Dict with at least a ``model`` key; a ``timestamp``
                is added automatically if absent.

        Returns:
            bool: *True* if the result was persisted.
        """
        preferences = self.load_preferences()
        
        # Add timestamp if not provided
        if "timestamp" not in test_data:
            test_data["timestamp"] = datetime.now().isoformat()
        
        # Store under model name or use "startup_test" as default
        model_name = test_data.get("model", "startup_test")
        preferences.model_test_results[model_name] = test_data
        
        return self.save_preferences(preferences)
    
    def get_model_test_results(self) -> Dict[str, Any]:
        """Get model test results.

        Returns:
            Dict: Per-model test outcome mappings.
        """
        preferences = self.load_preferences()
        return preferences.model_test_results
    
    def reset_preferences(self) -> bool:
        """Reset to default preferences.

        Returns:
            bool: *True* if defaults were restored and saved.
        """
        try:
            self._preferences = UserPreferences()
            return self.save_preferences()
        except Exception as e:
            logger.error("Error resetting preferences: %s", e, exc_info=True)
            return False
    
    def export_preferences(self) -> Dict[str, Any]:
        """Export preferences for backup or sharing.

        Returns:
            Dict: Full serialised preferences dict.
        """
        preferences = self.load_preferences()
        return preferences.model_dump()
    
    def import_preferences(self, data: Dict[str, Any]) -> bool:
        """Import preferences from backup.

        Args:
            data: Serialised preferences dict (as from :meth:`export_preferences`).

        Returns:
            bool: *True* if the import was validated and saved.
        """
        try:
            preferences = UserPreferences(**data)
            return self.save_preferences(preferences)
        except Exception as e:
            logger.error("Error importing preferences: %s", e, exc_info=True)
            return False

# Thread-safe global instance
_preferences_manager: Optional[UserPreferencesManager] = None
_preferences_lock = threading.Lock()


def get_preferences_manager() -> UserPreferencesManager:
    """Return the global :class:`UserPreferencesManager` singleton (thread-safe)."""
    global _preferences_manager
    if _preferences_manager is None:
        with _preferences_lock:
            if _preferences_manager is None:  # double-check locking
                _preferences_manager = UserPreferencesManager()
    return _preferences_manager


# =============================================================================
# ENTERPRISE: PREFERENCE VERSIONING
# =============================================================================

class PreferenceVersion:
    """Tracks preference schema versions for safe migrations.

    Attributes:
        CURRENT: The current schema version string.
    """
    CURRENT = "2.0.0"

    _HISTORY = [
        "1.0.0",  # Initial version
        "1.1.0",  # Added intelligent routing flag
        "2.0.0",  # Enterprise additions: profiles, tenant support
    ]

    @classmethod
    def needs_migration(cls, stored_version: str) -> bool:
        """Check if a stored version needs migration."""
        try:
            idx_stored = cls._HISTORY.index(stored_version)
            idx_current = cls._HISTORY.index(cls.CURRENT)
            return idx_stored < idx_current
        except ValueError:
            return True

    @classmethod
    def get_migration_path(cls, from_version: str) -> List[str]:
        """Get the ordered list of versions to migrate through."""
        try:
            start = cls._HISTORY.index(from_version) + 1
        except ValueError:
            start = 0
        try:
            end = cls._HISTORY.index(cls.CURRENT) + 1
        except ValueError:
            end = len(cls._HISTORY)
        return cls._HISTORY[start:end]


# =============================================================================
# ENTERPRISE: PREFERENCE MIGRATOR
# =============================================================================

class PreferenceMigrator:
    """Applies schema migrations to preference data.

    Register migration functions with :meth:`register` and execute
    them in sequence with :meth:`migrate`.

    .. code-block:: python

        migrator = PreferenceMigrator()
        migrator.register("1.0.0", "1.1.0", add_routing_flag)
        migrator.register("1.1.0", "2.0.0", add_profile_support)
        migrated = migrator.migrate(old_data, "1.0.0")
    """

    def __init__(self) -> None:
        self._migrations: Dict[str, Callable[[Dict], Dict]] = {}

    def register(
        self, from_version: str, to_version: str,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """Register a migration function.

        Args:
            from_version: Source schema version.
            to_version: Target schema version.
            fn: Function that transforms the data dict.
        """
        self._migrations[f"{from_version}->{to_version}"] = fn

    def migrate(
        self, data: Dict[str, Any], from_version: str,
    ) -> Dict[str, Any]:
        """Apply all necessary migrations.

        Args:
            data: Preference data to migrate.
            from_version: Current version of the data.

        Returns:
            Migrated data dict at :attr:`PreferenceVersion.CURRENT`.
        """
        path = PreferenceVersion.get_migration_path(from_version)
        result = copy.deepcopy(data)
        prev = from_version

        for target in path:
            key = f"{prev}->{target}"
            if key in self._migrations:
                try:
                    result = self._migrations[key](result)
                    logger.info("Migrated preferences %s", key)
                except Exception as e:
                    logger.error("Migration %s failed: %s", key, e)
                    break
            prev = target

        result["_schema_version"] = PreferenceVersion.CURRENT
        return result


# =============================================================================
# ENTERPRISE: PREFERENCE PROFILES
# =============================================================================

class PreferenceProfile:
    """Named preference profiles for quick switching.

    Stores multiple preference configurations (e.g., "development",
    "production", "demo") and allows switching between them.

    Args:
        storage_dir: Directory to store profile JSON files.
    """

    def __init__(self, storage_dir: Optional[str] = None) -> None:
        self._storage_dir = Path(
            storage_dir or os.path.join("data", "preference_profiles")
        )
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save_profile(self, name: str, preferences: UserPreferences) -> bool:
        """Save current preferences as a named profile.

        Args:
            name: Profile name.
            preferences: Preferences to save.

        Returns:
            True on success.
        """
        try:
            path = self._storage_dir / f"{name}.json"
            data = preferences.model_dump()
            data["_profile_name"] = name
            data["_saved_at"] = datetime.now().isoformat()
            with self._lock:
                path.write_text(json.dumps(data, indent=2))
            logger.info("Saved preference profile: %s", name)
            return True
        except Exception as e:
            logger.error("Failed to save profile %s: %s", name, e)
            return False

    def load_profile(self, name: str) -> Optional[UserPreferences]:
        """Load a named profile.

        Args:
            name: Profile name to load.

        Returns:
            :class:`UserPreferences` or None if not found.
        """
        path = self._storage_dir / f"{name}.json"
        if not path.exists():
            return None
        try:
            with self._lock:
                data = json.loads(path.read_text())
            data.pop("_profile_name", None)
            data.pop("_saved_at", None)
            return UserPreferences(**data)
        except Exception as e:
            logger.error("Failed to load profile %s: %s", name, e)
            return None

    def list_profiles(self) -> List[str]:
        """List all saved profile names."""
        return [p.stem for p in self._storage_dir.glob("*.json")]

    def delete_profile(self, name: str) -> bool:
        """Delete a saved profile."""
        path = self._storage_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False


# =============================================================================
# ENTERPRISE: MULTI-TENANT PREFERENCES
# =============================================================================

class MultiTenantPreferences:
    """Multi-tenant preference management.

    Maintains separate preference stores per tenant (user/org),
    with isolation and independent file persistence.

    Args:
        base_dir: Base directory for tenant preference files.
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self._base_dir = Path(base_dir or os.path.join("data", "tenant_preferences"))
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._managers: Dict[str, UserPreferencesManager] = {}

    def get_manager(self, tenant_id: str) -> UserPreferencesManager:
        """Get the preferences manager for a tenant.

        Args:
            tenant_id: Unique tenant identifier.

        Returns:
            Per-tenant :class:`UserPreferencesManager`.
        """
        with self._lock:
            if tenant_id not in self._managers:
                tenant_dir = self._base_dir / tenant_id
                tenant_dir.mkdir(parents=True, exist_ok=True)
                pref_file = str(tenant_dir / "user_preferences.json")
                self._managers[tenant_id] = UserPreferencesManager(preferences_file=pref_file)
                logger.info("Created preferences manager for tenant: %s", tenant_id)
            return self._managers[tenant_id]

    def list_tenants(self) -> List[str]:
        """List all known tenant IDs."""
        with self._lock:
            return list(self._managers.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Return multi-tenant statistics."""
        with self._lock:
            return {
                "total_tenants": len(self._managers),
                "tenants": list(self._managers.keys()),
            }


# =============================================================================
# ENTERPRISE SINGLETON
# =============================================================================

_multi_tenant: Optional[MultiTenantPreferences] = None
_multi_tenant_lock = threading.Lock()


def get_multi_tenant_preferences() -> MultiTenantPreferences:
    """Get or create the singleton multi-tenant manager (thread-safe)."""
    global _multi_tenant
    if _multi_tenant is None:
        with _multi_tenant_lock:
            if _multi_tenant is None:
                _multi_tenant = MultiTenantPreferences()
    return _multi_tenant