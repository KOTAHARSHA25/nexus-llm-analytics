"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Loader2, Activity, Zap, TrendingUp, BarChart3, RefreshCw } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { getEndpoint } from '@/lib/config'

interface RoutingStats {
  total_decisions: number
  average_complexity: number
  average_routing_time_ms: number
  tier_distribution: {
    fast: { count: number; percentage: number }
    balanced: { count: number; percentage: number }
    full: { count: number; percentage: number }
  }
  recent_decisions?: Array<{
    query: string
    complexity: number
    tier: string
    model: string
    timestamp: string
  }>
}

interface RoutingStatsProps {
  isOpen: boolean
  onClose: () => void
}

export default function RoutingStats({ isOpen, onClose }: RoutingStatsProps) {
  const [stats, setStats] = useState<RoutingStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [routingEnabled, setRoutingEnabled] = useState(false)

  useEffect(() => {
    if (isOpen) {
      loadStats()
    }
  }, [isOpen])

  const loadStats = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const response = await fetch(getEndpoint('analyzeRoutingStats'))
      const data = await response.json()
      
      if (data.status === 'success') {
        setStats(data.statistics)
        setRoutingEnabled(data.routing_enabled)
      } else {
        setError(data.message || 'Failed to load routing statistics')
      }
    } catch (err) {
      console.error('Failed to load routing stats:', err)
      setError('Failed to load routing statistics')
    } finally {
      setLoading(false)
    }
  }

  const getTierColor = (tier: string) => {
    switch (tier.toLowerCase()) {
      case 'fast':
        return 'bg-green-100 text-green-800'
      case 'balanced':
        return 'bg-blue-100 text-blue-800'
      case 'full':
        return 'bg-purple-100 text-purple-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getTierIcon = (tier: string) => {
    switch (tier.toLowerCase()) {
      case 'fast':
        return <Zap className="h-4 w-4" />
      case 'balanced':
        return <Activity className="h-4 w-4" />
      case 'full':
        return <TrendingUp className="h-4 w-4" />
      default:
        return <BarChart3 className="h-4 w-4" />
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-6 w-6 text-gray-900" />
              <h2 className="text-2xl font-bold text-gray-900">Intelligent Routing Statistics</h2>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={loadStats}
                disabled={loading}
                className="bg-white text-gray-900 border-2 border-gray-300 hover:bg-gray-100 hover:border-gray-400 font-medium"
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
              <Button 
                variant="outline" 
                onClick={onClose}
                className="bg-white text-gray-900 border-2 border-gray-300 hover:bg-gray-100 hover:border-gray-400 font-medium"
              >
                Close
              </Button>
            </div>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin" />
              <span className="ml-2">Loading statistics...</span>
            </div>
          ) : error ? (
            <Alert>
              <AlertDescription>
                <p className="text-red-600">{error}</p>
                {!routingEnabled && (
                  <p className="text-sm text-gray-600 mt-2">
                    Intelligent routing is not yet initialized. It will start collecting data after the first analysis request.
                  </p>
                )}
              </AlertDescription>
            </Alert>
          ) : stats ? (
            <div className="space-y-6">
              {/* Status Banner */}
              {routingEnabled ? (
                <Alert className="bg-green-50 border-green-200">
                  <AlertDescription>
                    <div className="flex items-center gap-2">
                      <Activity className="h-4 w-4 text-green-600" />
                      <span className="font-medium text-green-900">
                        Intelligent routing is active and collecting data
                      </span>
                    </div>
                  </AlertDescription>
                </Alert>
              ) : (
                <Alert className="bg-blue-50 border-blue-200">
                  <AlertDescription>
                    <div className="flex items-center gap-2">
                      <Activity className="h-4 w-4 text-blue-600" />
                      <span className="font-medium text-blue-900">
                        Router initialized but intelligent routing is disabled (using manual model selection)
                      </span>
                    </div>
                  </AlertDescription>
                </Alert>
              )}

              {/* Overview Cards */}
              <div className="grid grid-cols-3 gap-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Total Decisions
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold">{stats.total_decisions}</div>
                    <p className="text-xs text-gray-600 mt-1">
                      Routing decisions made
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Avg Complexity
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold">
                      {stats.average_complexity?.toFixed(3) || '0.000'}
                    </div>
                    <p className="text-xs text-gray-600 mt-1">
                      Query complexity score (0-1)
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Routing Speed
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold">
                      {stats.average_routing_time_ms?.toFixed(2) || '0.00'}ms
                    </div>
                    <p className="text-xs text-gray-600 mt-1">
                      Average routing overhead
                    </p>
                  </CardContent>
                </Card>
              </div>

              {/* Tier Distribution */}
              <Card>
                <CardHeader>
                  <CardTitle>Model Tier Distribution</CardTitle>
                  <CardDescription>
                    How queries are distributed across different model tiers
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {stats.total_decisions > 0 ? (
                    <div className="space-y-4">
                      {Object.entries(stats.tier_distribution).map(([tier, data]) => (
                        <div key={tier} className="space-y-2">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              {getTierIcon(tier)}
                              <span className="font-medium capitalize">{tier} Tier</span>
                              <Badge className={getTierColor(tier)}>
                                {data.count} queries
                              </Badge>
                            </div>
                            <span className="text-sm font-semibold">
                              {data.percentage.toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                            <div
                              className={`h-full ${
                                tier === 'fast'
                                  ? 'bg-green-500'
                                  : tier === 'balanced'
                                  ? 'bg-blue-500'
                                  : 'bg-purple-500'
                              }`}
                              style={{ width: `${data.percentage}%` }}
                            />
                          </div>
                          <div className="text-xs text-gray-600">
                            {tier === 'fast' && '⚡ Simple queries - Fast models (2GB RAM)'}
                            {tier === 'balanced' && '⚖️ Medium queries - Balanced models (6GB RAM)'}
                            {tier === 'full' && '🚀 Complex queries - Powerful models (16GB RAM)'}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <Alert>
                      <AlertDescription>
                        No routing decisions yet. Statistics will appear after queries are processed.
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>

              {/* Performance Insights */}
              {stats.total_decisions > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Performance Insights</CardTitle>
                    <CardDescription>
                      Estimated resource savings with intelligent routing
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <Alert className="bg-blue-50 border-blue-200">
                        <AlertDescription>
                          <div className="space-y-2 text-sm">
                            <p className="font-medium text-blue-900">
                              📊 Routing Efficiency Analysis
                            </p>
                            <div className="space-y-1 text-blue-800">
                              <p>
                                • {stats.tier_distribution.fast?.percentage.toFixed(1)}% of queries using FAST tier 
                                → Saving ~14GB RAM per query
                              </p>
                              <p>
                                • {stats.tier_distribution.balanced?.percentage.toFixed(1)}% of queries using BALANCED tier 
                                → Saving ~10GB RAM per query
                              </p>
                              <p>
                                • Average routing overhead: {stats.average_routing_time_ms?.toFixed(2)}ms 
                                (negligible impact)
                              </p>
                            </div>
                            <p className="text-xs text-blue-700 mt-2">
                              💡 Estimated: {((stats.tier_distribution.fast?.percentage || 0) * 0.1 + 
                                (stats.tier_distribution.balanced?.percentage || 0) * 0.03).toFixed(0)}x faster average response time
                            </p>
                          </div>
                        </AlertDescription>
                      </Alert>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Info Card */}
              <Alert>
                <AlertDescription>
                  <div className="text-sm space-y-1">
                    <p className="font-medium">ℹ️ About Intelligent Routing</p>
                    <p className="text-gray-600">
                      Intelligent routing analyzes query complexity and automatically selects the optimal model tier. 
                      This feature is OFF by default and must be enabled in Model Settings.
                    </p>
                    <p className="text-gray-600 mt-2">
                      When disabled, all queries use your primary model from settings (default behavior).
                    </p>
                  </div>
                </AlertDescription>
              </Alert>
            </div>
          ) : (
            <Alert>
              <AlertDescription>
                No statistics available yet.
              </AlertDescription>
            </Alert>
          )}
        </div>
      </div>
    </div>
  )
}
