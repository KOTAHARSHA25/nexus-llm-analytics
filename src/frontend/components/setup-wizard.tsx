"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { getEndpoint } from '@/lib/config'
import { 
  Loader2, 
  CheckCircle, 
  XCircle, 
  Rocket, 
  Cpu, 
  Zap, 
  Settings, 
  ArrowRight, 
  ArrowLeft,
  Gauge
} from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'

interface SetupWizardProps {
  isOpen: boolean
  onComplete: () => void
}

interface ModelRecommendation {
  config: string
  primary: string
  review: string
  description: string
  performance: string
}

export default function SetupWizard({ isOpen, onComplete }: SetupWizardProps) {
  const [step, setStep] = useState(1)
  const [systemInfo, setSystemInfo] = useState<any>(null)
  const [recommendations, setRecommendations] = useState<ModelRecommendation[]>([])
  const [selectedConfig, setSelectedConfig] = useState<string>('')
  const [customConfig, setCustomConfig] = useState({
    primary_model: 'phi3:mini',
    review_model: 'phi3:mini',
    auto_selection: true,
    allow_swap: true
  })
  const [loading, setLoading] = useState(true)
  const [testing, setTesting] = useState(false)
  const [testResults, setTestResults] = useState<any>(null)
  const [saving, setSaving] = useState(false)

  const totalSteps = 4

  useEffect(() => {
    if (isOpen) {
      loadRecommendations()
    }
  }, [isOpen])

  const loadRecommendations = async () => {
    try {
      setLoading(true)
      const response = await fetch(getEndpoint('modelsRecommendations'))
      const data = await response.json()
      
      if (data.system_info) {
        setSystemInfo(data.system_info)
      }
      if (data.recommendations) {
        setRecommendations(data.recommendations)
        // Auto-select the first (best) recommendation
        if (data.recommendations.length > 0) {
          setSelectedConfig(data.recommendations[0].config)
        }
      }
    } catch (error) {
      console.error('Failed to load recommendations:', error)
    } finally {
      setLoading(false)
    }
  }

  const testSelectedConfiguration = async () => {
    try {
      setTesting(true)
      const selectedRec = recommendations.find(r => r.config === selectedConfig)
      if (!selectedRec) return

      // Test primary model
      const primaryTest = await fetch(getEndpoint('modelsTestModel'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_name: selectedRec.primary })
      })
      const primaryResult = await primaryTest.json()

      // Test review model (if different)
      let reviewResult = primaryResult
      if (selectedRec.review !== selectedRec.primary) {
        const reviewTest = await fetch(getEndpoint('modelsTestModel'), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ model_name: selectedRec.review })
        })
        reviewResult = await reviewTest.json()
      }

      setTestResults({
        primary: primaryResult,
        review: reviewResult,
        overall_success: primaryResult.success && reviewResult.success
      })
    } catch (error) {
      console.error('Testing failed:', error)
      setTestResults({
        primary: { success: false, error: 'Test failed' },
        review: { success: false, error: 'Test failed' },
        overall_success: false
      })
    } finally {
      setTesting(false)
    }
  }

  const saveConfiguration = async () => {
    try {
      setSaving(true)
      const selectedRec = recommendations.find(r => r.config === selectedConfig)
      
      const config = selectedRec ? {
        primary_model: selectedRec.primary,
        review_model: selectedRec.review,
        embedding_model: 'nomic-embed-text',
        auto_selection: true,
        allow_swap: customConfig.allow_swap
      } : {
        primary_model: customConfig.primary_model,
        review_model: customConfig.review_model,
        embedding_model: 'nomic-embed-text',
        auto_selection: customConfig.auto_selection,
        allow_swap: customConfig.allow_swap
      }

      // Save preferences
      const response = await fetch(getEndpoint('modelsPreferences'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })

      if (response.ok) {
        // Mark setup as complete
        await fetch(getEndpoint('modelsSetupComplete'), { method: 'POST' })
        onComplete()
      } else {
        throw new Error('Failed to save configuration')
      }
    } catch (error) {
      console.error('Failed to save configuration:', error)
      alert('Failed to save configuration. Please try again.')
    } finally {
      setSaving(false)
    }
  }

  const getStepTitle = () => {
    switch (step) {
      case 1: return 'Welcome to Nexus LLM Analytics'
      case 2: return 'System Analysis'
      case 3: return 'Configuration Testing'
      case 4: return 'Setup Complete'
      default: return 'Setup'
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          {/* Header */}
          <div className="text-center mb-6">
            <div className="flex items-center justify-center gap-2 mb-2">
              <Rocket className="h-8 w-8 text-blue-600" />
              <h1 className="text-3xl font-bold">Setup Wizard</h1>
            </div>
            <p className="text-gray-600">{getStepTitle()}</p>
          </div>

          {/* Progress Bar */}
          <div className="mb-8">
            <div className="flex justify-between text-sm text-gray-500 mb-2">
              <span>Step {step} of {totalSteps}</span>
              <span>{Math.round((step / totalSteps) * 100)}% Complete</span>
            </div>
            <Progress value={(step / totalSteps) * 100} className="h-2" />
          </div>

          {/* Step Content */}
          {loading && step === 1 ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin mr-2" />
              <span>Loading system information...</span>
            </div>
          ) : (
            <>
              {/* Step 1: Welcome */}
              {step === 1 && (
                <div className="text-center space-y-6">
                  <div className="max-w-2xl mx-auto">
                    <h2 className="text-2xl font-semibold mb-4">Welcome! Let's get you set up.</h2>
                    <p className="text-gray-600 mb-6">
                      This wizard will help you configure the best AI models for your system. 
                      We'll analyze your hardware and recommend optimal settings for the best performance.
                    </p>
                    
                    {systemInfo && (
                      <Card className="text-left">
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <Cpu className="h-5 w-5" />
                            Your System
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-3 gap-4">
                            <div>
                              <Label>Total RAM</Label>
                              <p className="font-semibold text-lg">{systemInfo.total_gb.toFixed(1)} GB</p>
                            </div>
                            <div>
                              <Label>Available RAM</Label>
                              <p className="font-semibold text-lg">{systemInfo.available_gb.toFixed(1)} GB</p>
                            </div>
                            <div>
                              <Label>Memory Usage</Label>
                              <p className="font-semibold text-lg">{systemInfo.percent_used.toFixed(1)}%</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </div>
                </div>
              )}

              {/* Step 2: Recommendations */}
              {step === 2 && (
                <div className="space-y-6">
                  <div className="text-center">
                    <h2 className="text-2xl font-semibold mb-2">Choose Your Configuration</h2>
                    <p className="text-gray-600">
                      Based on your system, here are our recommendations:
                    </p>
                  </div>

                  <div className="grid gap-4">
                    {recommendations.map((rec, index) => (
                      <Card 
                        key={rec.config}
                        className={`cursor-pointer transition-all ${
                          selectedConfig === rec.config ? 'ring-2 ring-blue-500 bg-blue-50' : 'hover:bg-gray-50'
                        }`}
                        onClick={() => setSelectedConfig(rec.config)}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-center justify-between">
                            <div className="flex-1">
                              <div className="flex items-center gap-3">
                                <div className={`w-4 h-4 rounded-full border-2 ${
                                  selectedConfig === rec.config 
                                    ? 'bg-blue-500 border-blue-500' 
                                    : 'border-gray-300'
                                }`} />
                                <h3 className="font-semibold">{rec.config}</h3>
                                <Badge className={`text-xs ${
                                  rec.performance === 'Excellent' ? 'bg-green-100 text-green-800' :
                                  rec.performance === 'Good' ? 'bg-blue-100 text-blue-800' :
                                  rec.performance === 'Fair' ? 'bg-yellow-100 text-yellow-800' :
                                  'bg-red-100 text-red-800'
                                }`}>
                                  {rec.performance}
                                </Badge>
                              </div>
                              <p className="text-sm text-gray-600 mt-1">{rec.description}</p>
                              <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                                <span>Primary: {rec.primary}</span>
                                <span>Review: {rec.review}</span>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>

                  <div className="mt-6 p-4 bg-yellow-50 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <Switch
                        id="allow-swap-setup"
                        checked={customConfig.allow_swap}
                        onCheckedChange={(checked) =>
                          setCustomConfig(prev => ({ ...prev, allow_swap: checked }))
                        }
                      />
                      <Label htmlFor="allow-swap-setup" className="text-sm">
                        Allow swap memory usage (slower but works with limited RAM)
                      </Label>
                    </div>
                    <p className="text-xs text-gray-600">
                      If enabled, models can use disk space when RAM is insufficient. This allows larger models to run but will be slower.
                    </p>
                  </div>
                </div>
              )}

              {/* Step 3: Testing */}
              {step === 3 && (
                <div className="space-y-6">
                  <div className="text-center">
                    <h2 className="text-2xl font-semibold mb-2">Testing Configuration</h2>
                    <p className="text-gray-600">
                      Let's test your selected configuration to make sure everything works.
                    </p>
                  </div>

                  {selectedConfig && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Selected Configuration: {selectedConfig}</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          {!testResults && !testing && (
                            <div className="text-center py-6">
                              <Button onClick={testSelectedConfiguration} size="lg">
                                <Gauge className="h-5 w-5 mr-2" />
                                Test Configuration
                              </Button>
                            </div>
                          )}

                          {testing && (
                            <div className="text-center py-6">
                              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-2" />
                              <p>Testing models... This may take a minute.</p>
                            </div>
                          )}

                          {testResults && (
                            <div className="space-y-3">
                              <div className="flex items-center gap-2">
                                {testResults.overall_success ? (
                                  <CheckCircle className="h-5 w-5 text-green-500" />
                                ) : (
                                  <XCircle className="h-5 w-5 text-red-500" />
                                )}
                                <span className="font-semibold">
                                  {testResults.overall_success ? 'Configuration Test Passed!' : 'Configuration Test Failed'}
                                </span>
                              </div>

                              <div className="grid grid-cols-2 gap-4 text-sm">
                                <div className="p-3 border rounded">
                                  <div className="flex items-center gap-2 mb-1">
                                    {testResults.primary.success ? (
                                      <CheckCircle className="h-4 w-4 text-green-500" />
                                    ) : (
                                      <XCircle className="h-4 w-4 text-red-500" />
                                    )}
                                    <span className="font-medium">Primary Model</span>
                                  </div>
                                  {testResults.primary.response_time && (
                                    <p className="text-gray-600">
                                      Response time: {testResults.primary.response_time.toFixed(2)}s
                                    </p>
                                  )}
                                </div>

                                <div className="p-3 border rounded">
                                  <div className="flex items-center gap-2 mb-1">
                                    {testResults.review.success ? (
                                      <CheckCircle className="h-4 w-4 text-green-500" />
                                    ) : (
                                      <XCircle className="h-4 w-4 text-red-500" />
                                    )}
                                    <span className="font-medium">Review Model</span>
                                  </div>
                                  {testResults.review.response_time && (
                                    <p className="text-gray-600">
                                      Response time: {testResults.review.response_time.toFixed(2)}s
                                    </p>
                                  )}
                                </div>
                              </div>

                              {!testResults.overall_success && (
                                <Alert>
                                  <AlertDescription>
                                    Some models failed to respond. You can continue with this configuration (it may work with your data) 
                                    or go back to choose a different one.
                                  </AlertDescription>
                                </Alert>
                              )}
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}

              {/* Step 4: Complete */}
              {step === 4 && (
                <div className="text-center space-y-6">
                  <div className="max-w-2xl mx-auto">
                    <CheckCircle className="h-16 w-16 text-green-500 mx-auto mb-4" />
                    <h2 className="text-2xl font-semibold mb-4">Setup Complete!</h2>
                    <p className="text-gray-600 mb-6">
                      Your Nexus LLM Analytics is now configured and ready to use. 
                      You can always change these settings later from the Settings menu.
                    </p>
                    
                    <div className="bg-blue-50 p-4 rounded-lg text-left">
                      <h3 className="font-semibold mb-2">Your Configuration:</h3>
                      <ul className="space-y-1 text-sm">
                        {selectedConfig && (
                          <>
                            <li>• Configuration: {selectedConfig}</li>
                            <li>• Primary Model: {recommendations.find(r => r.config === selectedConfig)?.primary}</li>
                            <li>• Review Model: {recommendations.find(r => r.config === selectedConfig)?.review}</li>
                            <li>• Swap Usage: {customConfig.allow_swap ? 'Enabled' : 'Disabled'}</li>
                          </>
                        )}
                      </ul>
                    </div>
                  </div>
                </div>
              )}

              {/* Navigation */}
              <div className="flex justify-between mt-8">
                <Button
                  variant="outline"
                  onClick={() => setStep(step - 1)}
                  disabled={step === 1}
                >
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Previous
                </Button>

                <div className="flex gap-2">
                  {step < totalSteps ? (
                    <Button
                      onClick={() => setStep(step + 1)}
                      disabled={step === 2 && !selectedConfig}
                    >
                      Next
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </Button>
                  ) : (
                    <Button
                      onClick={saveConfiguration}
                      disabled={saving}
                      className="bg-green-600 hover:bg-green-700"
                    >
                      {saving ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin mr-2" />
                          Saving...
                        </>
                      ) : (
                        <>
                          <CheckCircle className="h-4 w-4 mr-2" />
                          Complete Setup
                        </>
                      )}
                    </Button>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}