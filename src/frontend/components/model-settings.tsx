"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Loader2, CheckCircle, XCircle, Settings, Cpu, Zap, Gauge } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'

interface ModelInfo {
  name: string
  size_gb: number
  description: string
  capabilities: string[]
  compatible: boolean
  compatibility_message: string
}

interface UserPreferences {
  primary_model: string
  review_model: string
  embedding_model: string
  auto_model_selection: boolean
  allow_swap_usage: boolean
  preferred_performance: string
  first_time_setup_complete: boolean
}

interface ModelSettingsProps {
  isOpen: boolean
  onClose: () => void
}

export default function ModelSettings({ isOpen, onClose }: ModelSettingsProps) {
  const [preferences, setPreferences] = useState<UserPreferences | null>(null)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [systemInfo, setSystemInfo] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState<string | null>(null)
  const [testResults, setTestResults] = useState<Record<string, any>>({})

  useEffect(() => {
    if (isOpen) {
      loadData()
    }
  }, [isOpen])

  const loadData = async () => {
    try {
      setLoading(true)
      
      // Load user preferences
      const prefsResponse = await fetch('http://127.0.0.1:8000/models/preferences')
      const prefsData = await prefsResponse.json()
      if (prefsData.preferences) {
        setPreferences(prefsData.preferences)
      }

      // Load available models
      const modelsResponse = await fetch('http://127.0.0.1:8000/models/available')
      const modelsData = await modelsResponse.json()
      if (modelsData.models) {
        setModels(modelsData.models)
        setSystemInfo(modelsData.system_memory)
      }

      // Load test results
      const testResponse = await fetch('http://127.0.0.1:8000/models/test-results')
      const testData = await testResponse.json()
      if (testData.test_results) {
        setTestResults(testData.test_results)
      }
    } catch (error) {
      console.error('Failed to load model settings:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    if (!preferences) return

    try {
      setSaving(true)
      const response = await fetch('http://127.0.0.1:8000/models/preferences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          primary_model: preferences.primary_model,
          review_model: preferences.review_model,
          embedding_model: preferences.embedding_model,
          auto_selection: preferences.auto_model_selection,
          allow_swap: preferences.allow_swap_usage
        })
      })

      const result = await response.json()
      if (result.status === 'success') {
        alert('Settings saved successfully!')
        onClose()
      } else {
        alert('Failed to save settings: ' + result.error)
      }
    } catch (error) {
      console.error('Failed to save settings:', error)
      alert('Failed to save settings')
    } finally {
      setSaving(false)
    }
  }

  const testModel = async (modelName: string) => {
    try {
      setTesting(modelName)
      const response = await fetch('http://127.0.0.1:8000/models/test-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_name: modelName })
      })
      const result = await response.json()
      
      setTestResults(prev => ({
        ...prev,
        [modelName]: result
      }))
    } catch (error) {
      console.error('Failed to test model:', error)
    } finally {
      setTesting(null)
    }
  }

  const getPerformanceBadge = (performance: string) => {
    const variants = {
      'Excellent': 'bg-green-100 text-green-800',
      'Good': 'bg-blue-100 text-blue-800',
      'Fair': 'bg-yellow-100 text-yellow-800',
      'Slow': 'bg-red-100 text-red-800'
    }
    return variants[performance as keyof typeof variants] || 'bg-gray-100 text-gray-800'
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <div className="flex items-center gap-2">
              <Settings className="h-6 w-6" />
              <h2 className="text-2xl font-bold">Model Settings</h2>
            </div>
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin" />
              <span className="ml-2">Loading settings...</span>
            </div>
          ) : (
            <div className="space-y-6">
              {/* System Information */}
              {systemInfo && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Cpu className="h-5 w-5" />
                      System Information
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <Label>Total RAM</Label>
                        <p className="font-semibold">{systemInfo.total_gb.toFixed(1)} GB</p>
                      </div>
                      <div>
                        <Label>Available RAM</Label>
                        <p className="font-semibold">{systemInfo.available_gb.toFixed(1)} GB</p>
                      </div>
                      <div>
                        <Label>Usage</Label>
                        <p className="font-semibold">{systemInfo.percent_used.toFixed(1)}%</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Model Configuration */}
              {preferences && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Zap className="h-5 w-5" />
                      Model Configuration
                    </CardTitle>
                    <CardDescription>
                      Configure which AI models to use for different tasks
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Primary Model (Analysis & Code Generation)</Label>
                        <Select
                          value={preferences.primary_model}
                          onValueChange={(value) => 
                            setPreferences(prev => prev ? {...prev, primary_model: value} : null)
                          }
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {models.map(model => (
                              <SelectItem key={model.name} value={model.name}>
                                <div className="flex items-center justify-between w-full">
                                  <span>{model.name}</span>
                                  {model.compatible ? (
                                    <CheckCircle className="h-4 w-4 text-green-500" />
                                  ) : (
                                    <XCircle className="h-4 w-4 text-red-500" />
                                  )}
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label>Review Model (Code Review & QA)</Label>
                        <Select
                          value={preferences.review_model}
                          onValueChange={(value) => 
                            setPreferences(prev => prev ? {...prev, review_model: value} : null)
                          }
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {models.map(model => (
                              <SelectItem key={model.name} value={model.name}>
                                <div className="flex items-center justify-between w-full">
                                  <span>{model.name}</span>
                                  {model.compatible ? (
                                    <CheckCircle className="h-4 w-4 text-green-500" />
                                  ) : (
                                    <XCircle className="h-4 w-4 text-red-500" />
                                  )}
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Switch
                        id="auto-selection"
                        checked={preferences.auto_model_selection}
                        onCheckedChange={(checked) =>
                          setPreferences(prev => prev ? {...prev, auto_model_selection: checked} : null)
                        }
                      />
                      <Label htmlFor="auto-selection">
                        Enable automatic model selection based on system resources
                      </Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Switch
                        id="allow-swap"
                        checked={preferences.allow_swap_usage}
                        onCheckedChange={(checked) =>
                          setPreferences(prev => prev ? {...prev, allow_swap_usage: checked} : null)
                        }
                      />
                      <Label htmlFor="allow-swap">
                        Allow swap memory usage (slower but works with limited RAM)
                      </Label>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Model Testing */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Gauge className="h-5 w-5" />
                    Model Testing
                  </CardTitle>
                  <CardDescription>
                    Test models to verify they work on your system
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {models.map(model => {
                      const testResult = testResults[model.name]
                      const isTestingThis = testing === model.name
                      
                      return (
                        <div key={model.name} className="flex items-center justify-between p-3 border rounded-lg">
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{model.name}</span>
                              <Badge className={`text-xs ${model.compatible ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                                {model.compatible ? 'Compatible' : 'Incompatible'}
                              </Badge>
                              {testResult && (
                                <Badge className={`text-xs ${testResult.success ? 'bg-blue-100 text-blue-800' : 'bg-red-100 text-red-800'}`}>
                                  {testResult.success ? 'Tested ✓' : 'Failed ✗'}
                                </Badge>
                              )}
                            </div>
                            <p className="text-sm text-gray-600">{model.description}</p>
                            {testResult && testResult.response_time && (
                              <p className="text-xs text-gray-500">
                                Response time: {testResult.response_time.toFixed(2)}s
                              </p>
                            )}
                          </div>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => testModel(model.name)}
                            disabled={isTestingThis}
                          >
                            {isTestingThis ? (
                              <>
                                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                                Testing...
                              </>
                            ) : (
                              'Test Model'
                            )}
                          </Button>
                        </div>
                      )
                    })}
                  </div>
                </CardContent>
              </Card>

              {/* Save Button */}
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={onClose}>
                  Cancel
                </Button>
                <Button onClick={handleSave} disabled={saving}>
                  {saving ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin mr-2" />
                      Saving...
                    </>
                  ) : (
                    'Save Settings'
                  )}
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}