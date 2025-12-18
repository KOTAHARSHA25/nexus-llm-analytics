// Performance-optimized components with React.memo and advanced caching
// BEFORE: Components re-render on every state change (O(n) re-renders)
// AFTER: Memoized components with selective re-rendering (O(1) re-renders)

import React, { memo, useMemo, useCallback } from 'react';
import { FileInfo, PluginInfo } from '../hooks/useDashboardState';

// Memoized file list component - only re-renders when files change
export const FileList = memo<{
  files: FileInfo[];
  onRemoveFile: (fileId: string) => void;
  isLoading?: boolean;
}>(({ files, onRemoveFile, isLoading = false }) => {
  // Memoize sorted and filtered files
  const processedFiles = useMemo(() => {
    return files
      .filter(file => file.name && file.type)
      .sort((a, b) => b.uploadedAt - a.uploadedAt)
      .slice(0, 50); // Limit to 50 files for performance
  }, [files]);

  // Memoize remove handlers to prevent recreation
  const removeHandlers = useMemo(() => {
    const handlers = new Map<string, () => void>();
    processedFiles.forEach(file => {
      handlers.set(file.id, () => onRemoveFile(file.id));
    });
    return handlers;
  }, [processedFiles, onRemoveFile]);

  if (processedFiles.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No files uploaded yet
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {processedFiles.map((file) => (
        <FileItem
          key={file.id}
          file={file}
          onRemove={removeHandlers.get(file.id)!}
          isLoading={isLoading}
        />
      ))}
    </div>
  );
});

FileList.displayName = 'FileList';

// Memoized individual file item - prevents unnecessary re-renders
const FileItem = memo<{
  file: FileInfo;
  onRemove: () => void;
  isLoading: boolean;
}>(({ file, onRemove, isLoading }) => {
  // Memoize file type icon and styling
  const fileDisplay = useMemo(() => {
    const getFileIcon = (type: string) => {
      switch (type.toLowerCase()) {
        case 'csv': return '📊';
        case 'json': return '📋';
        case 'pdf': return '📄';
        case 'txt': return '📝';
        default: return '📁';
      }
    };

    const getFileColor = (type: string) => {
      switch (type.toLowerCase()) {
        case 'csv': return 'bg-green-50 border-green-200';
        case 'json': return 'bg-blue-50 border-blue-200';
        case 'pdf': return 'bg-red-50 border-red-200';
        case 'txt': return 'bg-gray-50 border-gray-200';
        default: return 'bg-purple-50 border-purple-200';
      }
    };

    return {
      icon: getFileIcon(file.type),
      colorClass: getFileColor(file.type)
    };
  }, [file.type]);

  // Memoize file size formatting
  const fileMetadata = useMemo(() => {
    const formatFileSize = (bytes?: number) => {
      if (!bytes) return 'Unknown size';
      const sizes = ['Bytes', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(1024));
      return `${Math.round(bytes / Math.pow(1024, i) * 100) / 100} ${sizes[i]}`;
    };

    const formatUploadTime = (timestamp: number) => {
      const now = Date.now();
      const diff = now - timestamp;
      const minutes = Math.floor(diff / 60000);
      const hours = Math.floor(minutes / 60);
      const days = Math.floor(hours / 24);

      if (days > 0) return `${days}d ago`;
      if (hours > 0) return `${hours}h ago`;
      if (minutes > 0) return `${minutes}m ago`;
      return 'Just now';
    };

    return {
      size: formatFileSize((file as any).size),
      uploadTime: formatUploadTime(file.uploadedAt)
    };
  }, [file.uploadedAt, (file as any).size]);

  return (
    <div className={`flex items-center justify-between p-3 rounded-lg border ${fileDisplay.colorClass}`}>
      <div className="flex items-center space-x-3">
        <span className="text-2xl">{fileDisplay.icon}</span>
        <div>
          <div className="font-medium text-gray-900">{file.name}</div>
          <div className="text-sm text-gray-500">
            {file.type.toUpperCase()} • {fileMetadata.size} • {fileMetadata.uploadTime}
          </div>
          {file.columns && file.columns.length > 0 && (
            <div className="text-xs text-gray-400 mt-1">
              {file.columns.length} columns: {file.columns.slice(0, 3).join(', ')}
              {file.columns.length > 3 && '...'}
            </div>
          )}
        </div>
      </div>
      <button
        onClick={onRemove}
        disabled={isLoading}
        className="text-red-500 hover:text-red-700 disabled:opacity-50 p-1"
        aria-label={`Remove ${file.name}`}
      >
        ❌
      </button>
    </div>
  );
});

FileItem.displayName = 'FileItem';

// Memoized plugin selector with optimized rendering
export const PluginSelector = memo<{
  plugins: PluginInfo[];
  selectedPlugin: string;
  onSelectPlugin: (pluginName: string) => void;
  isLoading?: boolean;
}>(({ plugins, selectedPlugin, onSelectPlugin, isLoading = false }) => {
  // Memoize plugin handlers
  const pluginHandlers = useMemo(() => {
    const handlers = new Map<string, () => void>();
    plugins.forEach(plugin => {
      handlers.set(plugin.name, () => onSelectPlugin(plugin.name));
    });
    return handlers;
  }, [plugins, onSelectPlugin]);

  // Memoize filtered and sorted plugins
  const processedPlugins = useMemo(() => {
    return plugins
      .filter(plugin => plugin.name && plugin.description)
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [plugins]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {processedPlugins.map((plugin) => (
        <PluginCard
          key={plugin.id}
          plugin={plugin}
          isSelected={plugin.name === selectedPlugin}
          onClick={pluginHandlers.get(plugin.name)!}
          isLoading={isLoading}
        />
      ))}
    </div>
  );
});

PluginSelector.displayName = 'PluginSelector';

// Memoized plugin card component
const PluginCard = memo<{
  plugin: PluginInfo;
  isSelected: boolean;
  onClick: () => void;
  isLoading: boolean;
}>(({ plugin, isSelected, onClick, isLoading }) => {
  // Memoize plugin styling
  const cardStyling = useMemo(() => {
    const baseClasses = "p-4 rounded-lg border cursor-pointer transition-all duration-200";
    const selectedClasses = isSelected 
      ? `${baseClasses} border-blue-500 bg-blue-50 shadow-md`
      : `${baseClasses} border-gray-200 hover:border-gray-300 hover:shadow-sm`;
    
    return {
      cardClasses: selectedClasses,
      iconColor: plugin.color || '#6B7280'
    };
  }, [isSelected, plugin.color]);

  // Memoize capabilities display
  const capabilitiesDisplay = useMemo(() => {
    if (!plugin.capabilities || plugin.capabilities.length === 0) {
      return null;
    }
    
    return plugin.capabilities.slice(0, 3).join(' • ') + 
           (plugin.capabilities.length > 3 ? '...' : '');
  }, [plugin.capabilities]);

  return (
    <div
      className={cardStyling.cardClasses}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
      aria-pressed={isSelected}
      aria-disabled={isLoading}
    >
      <div className="flex items-start space-x-3">
        {plugin.icon && (
          <div style={{ color: cardStyling.iconColor }}>
            <plugin.icon className="w-6 h-6" />
          </div>
        )}
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900">{plugin.name}</h3>
          <p className="text-sm text-gray-600 mt-1">{plugin.description}</p>
          {capabilitiesDisplay && (
            <p className="text-xs text-gray-500 mt-2">{capabilitiesDisplay}</p>
          )}
        </div>
        {isSelected && (
          <div className="text-blue-500">
            ✓
          </div>
        )}
      </div>
    </div>
  );
});

PluginCard.displayName = 'PluginCard';

// Memoized query history component
export const QueryHistory = memo<{
  history: string[];
  onSelectQuery: (query: string) => void;
  onClearHistory: () => void;
  maxItems?: number;
}>(({ history, onSelectQuery, onClearHistory, maxItems = 10 }) => {
  // Memoize limited history
  const displayHistory = useMemo(() => {
    return history.slice(0, maxItems);
  }, [history, maxItems]);

  // Memoize query handlers
  const queryHandlers = useMemo(() => {
    const handlers = new Map<string, () => void>();
    displayHistory.forEach((query, index) => {
      const key = `${query}_${index}`;
      handlers.set(key, () => onSelectQuery(query));
    });
    return handlers;
  }, [displayHistory, onSelectQuery]);

  if (displayHistory.length === 0) {
    return (
      <div className="text-center py-4 text-gray-500 text-sm">
        No recent queries
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <h3 className="text-sm font-medium text-gray-700">Recent Queries</h3>
        <button
          onClick={onClearHistory}
          className="text-xs text-red-500 hover:text-red-700"
        >
          Clear All
        </button>
      </div>
      <div className="space-y-1">
        {displayHistory.map((query, index) => {
          const key = `${query}_${index}`;
          return (
            <QueryHistoryItem
              key={key}
              query={query}
              onClick={queryHandlers.get(key)!}
            />
          );
        })}
      </div>
    </div>
  );
});

QueryHistory.displayName = 'QueryHistory';

// Memoized query history item
const QueryHistoryItem = memo<{
  query: string;
  onClick: () => void;
}>(({ query, onClick }) => {
  // Memoize truncated query
  const displayQuery = useMemo(() => {
    return query.length > 100 ? `${query.substring(0, 100)}...` : query;
  }, [query]);

  return (
    <button
      onClick={onClick}
      className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded border text-gray-700 transition-colors duration-150"
      title={query}
    >
      {displayQuery}
    </button>
  );
});

QueryHistoryItem.displayName = 'QueryHistoryItem';

// Memoized progress indicator with optimized animations
export const ProgressIndicator = memo<{
  progress: number;
  isLoading: boolean;
  statusMsg?: string;
  showDetails?: boolean;
}>(({ progress, isLoading, statusMsg, showDetails = true }) => {
  // Memoize progress styling
  const progressStyling = useMemo(() => {
    const progressPercentage = Math.min(Math.max(progress, 0), 100);
    const progressColor = progressPercentage < 30 ? 'bg-red-500' :
                         progressPercentage < 70 ? 'bg-yellow-500' : 'bg-green-500';
    
    return {
      percentage: progressPercentage,
      color: progressColor,
      width: `${progressPercentage}%`
    };
  }, [progress]);

  // Memoize status message display
  const statusDisplay = useMemo(() => {
    if (!statusMsg) return 'Processing...';
    return statusMsg.length > 80 ? `${statusMsg.substring(0, 80)}...` : statusMsg;
  }, [statusMsg]);

  if (!isLoading && progress === 0) {
    return null;
  }

  return (
    <div className="w-full space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-gray-700">
          {isLoading ? 'Processing' : 'Complete'}
        </span>
        <span className="text-sm text-gray-500">
          {Math.round(progressStyling.percentage)}%
        </span>
      </div>
      
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${progressStyling.color}`}
          style={{ width: progressStyling.width }}
        />
      </div>
      
      {showDetails && statusMsg && (
        <p className="text-xs text-gray-600">
          {statusDisplay}
        </p>
      )}
    </div>
  );
});

ProgressIndicator.displayName = 'ProgressIndicator';